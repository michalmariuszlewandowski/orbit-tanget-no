from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from otno.config import load_config, recursive_update, set_by_path
from otno.data.datasets import TensorDictDataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.training.metrics import evaluate_model


class ZeroModel(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_all_experiment_configs_build_model_and_transform():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").glob("**/*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path)
        if "experiments" in cfg:
            for entry in cfg["experiments"]:
                base = load_config(root / entry["config"])
                updates = {}
                for key, value in entry.get("overrides", {}).items():
                    set_by_path(updates, key, value)
                merged = recursive_update(base, updates)
                assert "dataset" in merged and "model" in merged and "training" in merged
            continue
        if "adaptations" in cfg:
            for entry in cfg["adaptations"]:
                base = load_config(root / entry["config"])
                updates = {}
                for key, value in entry.get("overrides", {}).items():
                    set_by_path(updates, key, value)
                merged = recursive_update(base, updates)
                assert "adaptation" in merged and "runtime" in merged
            continue
        if "evaluations" in cfg:
            for entry in cfg["evaluations"]:
                transform = build_transform({"symmetry": entry.get("symmetry")})
                assert transform is not None
                assert entry.get("checkpoint")
                assert entry.get("out_dir")
            continue
        if "adaptation" in cfg:
            continue
        model = build_model(cfg)
        assert sum(p.numel() for p in model.parameters()) > 0
        transform = build_transform(cfg.get("symmetry"))
        assert transform is not None


def test_run_matrix_uses_valid_config_paths():
    root = Path(__file__).resolve().parents[1]
    matrix_path = root / "configs" / "experiments_june_september.yaml"
    matrix = yaml.safe_load(matrix_path.read_text())
    run_dirs = []
    for entry in matrix["experiments"]:
        assert (root / entry["config"]).exists()
        cfg = load_config(root / entry["config"])
        run_dir = entry.get("overrides", {}).get("runtime.run_dir", cfg.get("runtime", {}).get("run_dir"))
        assert run_dir is not None
        run_dirs.append(run_dir)
    assert len(run_dirs) == len(set(run_dirs))


def test_evaluate_model_seed_is_reproducible():
    dataset = TensorDictDataset(torch.randn(8, 16, 1), torch.randn(8, 16, 1))
    loader = DataLoader(dataset, batch_size=4)
    cfg = {"name": "translation1d", "max_shift": 0.2}
    transform = build_transform(cfg)
    model = ZeroModel()
    a = evaluate_model(model, loader, device=torch.device("cpu"), transform=transform, n_orbit_samples=3, seed=101)
    b = evaluate_model(model, loader, device=torch.device("cpu"), transform=transform, n_orbit_samples=3, seed=101)
    assert a["epsilon_mean"] == b["epsilon_mean"]
    assert a["oracle_canonical_ood_relative_l2"] == b["oracle_canonical_ood_relative_l2"]


def test_evaluate_model_reports_force_mae_for_molecular_outputs():
    dataset = TensorDictDataset(torch.randn(6, 4, 3), torch.randn(6, 4, 3))
    loader = DataLoader(dataset, batch_size=3)
    model = ZeroModel()
    metrics = evaluate_model(model, loader, device=torch.device("cpu"), transform=None)
    assert "mae" in metrics
    assert "force_mae" in metrics
