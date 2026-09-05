import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from otno.config import load_config, recursive_update, set_by_path, validate_config
from otno.data.datasets import TensorDictDataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.training.metrics import evaluate_model


def test_config_checks_can_require_checkpoints_without_requiring_training(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    checker = importlib.import_module("check_configs")
    matrix = tmp_path / "evaluation.yaml"
    matrix.write_text(
        "evaluations:\n"
        "  - checkpoint: runs/untrained/checkpoints/best.pt\n"
        "    out_dir: runs/evaluation\n"
        "    symmetry: {name: translation1d, max_shift: 0.1}\n",
        encoding="utf-8",
    )
    assert checker._validate_evaluation_matrix(matrix) == []
    assert any(
        "missing checkpoint" in error
        for error in checker._validate_evaluation_matrix(matrix, require_checkpoints=True)
    )
    matrix.write_text("evaluations:\n  - out_dir: runs/evaluation\n", encoding="utf-8")
    assert any(
        "missing checkpoint" in error for error in checker._validate_evaluation_matrix(matrix)
    )


class ZeroModel(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_all_experiment_configs_build_model_and_transform():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "configs").glob("**/*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path)
        if "release_version" in cfg:
            # Workflow registries are covered by test_release_reproduction.py.
            continue
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
    matrix_path = root / "configs" / "experiments_core.yaml"
    matrix = yaml.safe_load(matrix_path.read_text())
    run_dirs = []
    for entry in matrix["experiments"]:
        assert (root / entry["config"]).exists()
        cfg = load_config(root / entry["config"])
        run_dir = entry.get("overrides", {}).get("runtime.run_dir", cfg.get("runtime", {}).get("run_dir"))
        assert run_dir is not None
        run_dirs.append(run_dir)
    assert len(run_dirs) == len(set(run_dirs))


def test_run_matrix_skip_completed_only_skips_finalized_job(tmp_path):
    root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "seed_23"
    run_dir.mkdir()
    metrics = dict.fromkeys(
        ["relative_l2", "orbit_ood_relative_l2", "equivariance_defect_relative",
         "latency_ms_per_sample", "best_val_relative_l2", "parameters"], 1.0,
    )
    metrics["method"] = "baseline"
    metrics_path = run_dir / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        yaml.safe_dump(
            {
                "experiments": [
                    {
                        "config": "unused.yaml",
                        "seeds": [23],
                        "overrides": {"runtime.run_dir": str(run_dir.parent)},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_matrix.py",
            "--matrix",
            str(matrix_path),
            "--skip-completed",
            "--dry-run",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"skipping completed matrix job: {run_dir}" in completed.stdout
    for invalid in ("{}", "null", "not JSON", json.dumps({**metrics, "relative_l2": float("nan")})):
        metrics_path.write_text(invalid, encoding="utf-8")
        rejected = subprocess.run(completed.args, cwd=root, capture_output=True, text=True)
        assert rejected.returncode != 0
        assert "Invalid completed metrics" in rejected.stderr
        assert "skipping completed" not in rejected.stdout
        assert metrics_path.read_text(encoding="utf-8") == invalid


def test_aug_tangent_method_config_validates():
    cfg = {
        "seed": 1,
        "dataset": {
            "kind": "navier_stokes_vorticity2d_boosted",
            "path": "data/test_aug_tangent.pt",
            "n": 16,
            "num_train": 4,
            "num_val": 2,
            "num_test": 2,
            "final_time": 0.1,
            "dt": 0.01,
            "viscosity": 0.001,
            "max_boost": 0.2,
            "solver_batch_size": 2,
        },
        "model": {"name": "fno2d", "in_channels": 3, "out_channels": 1},
        "symmetry": {"name": "navier_stokes2d_galilean", "max_boost": 0.1},
        "training": {"method": "aug_tangent", "lambda_tangent": 0.1},
        "runtime": {"run_dir": "runs/test_aug_tangent"},
    }
    validate_config(cfg)


@pytest.mark.parametrize(("key", "value"), [
    ("training.lr", float("nan")),
    ("training.lr", float("inf")),
    ("training.epochs", 1.5),
    ("training.epochs", True),
    ("training.data_fraction", float("nan")),
    ("training.orbit_data_fraction", -0.1),
    ("training.orbit_batch_size", 0),
    ("training.orbit_steps_per_epoch", 0),
    ("model.branch_channels", [4, 2.5]),
    ("symmetry.enabled", False),
    ("symmetry.transforms", []),
])
def test_invalid_training_settings_are_rejected(key, value):
    cfg = {
        "dataset": {"kind": "advection1d", "path": "unused.pt"},
        "training": {"method": "orbit"},
        "symmetry": {"name": "translation1d", "max_shift": 0.1},
    }
    set_by_path(cfg, key, value)
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("probabilities", [[-1, 2], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_transform_probabilities_are_rejected(probabilities):
    cfg = {
        "dataset": {"kind": "advection1d", "path": "unused.pt"},
        "symmetry": {
            "transforms": [{"name": "translation1d"}, {"name": "translation1d"}],
            "probabilities": probabilities,
        },
    }
    with pytest.raises(ValueError, match="probabilities"):
        validate_config(cfg)


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


@pytest.mark.parametrize("evaluation", ["standard", "observable"])
def test_seeded_cpu_evaluation_preserves_all_cuda_rng_streams(monkeypatch, evaluation):
    cuda_generators = [torch.Generator().manual_seed(100 + index) for index in range(2)]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cuda_generators))
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: cuda_generators[device].get_state())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda state, device: cuda_generators[device].set_state(state))

    def seed_cuda(seed):
        for generator in cuda_generators:
            generator.manual_seed(seed)

    monkeypatch.setattr(torch.cuda, "manual_seed_all", seed_cuda)
    if evaluation == "observable":
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
        from run_observable_canonicalization_matrix import evaluate_observable_canonicalization

        a = torch.ones(2, 8, 8, 3)
        a[..., 1:] = 0
        dataset = TensorDictDataset(a, a[..., :1])
        model = nn.Linear(3, 1)
    else:
        dataset = TensorDictDataset(torch.ones(2, 8, 1), torch.ones(2, 8, 1))
        model = ZeroModel()
    loader = DataLoader(dataset, batch_size=2)
    cpu_state = torch.get_rng_state()
    cuda_states = [generator.get_state() for generator in cuda_generators]

    if evaluation == "observable":
        evaluate_observable_canonicalization(
            model, loader, device=torch.device("cpu"), seed=202, n_orbit_samples=1,
            transform=build_transform({"name": "navier_stokes2d_galilean", "max_boost": 0.01}),
        )
    else:
        evaluate_model(model, loader, device=torch.device("cpu"), seed=202)

    assert torch.equal(torch.get_rng_state(), cpu_state)
    for generator, expected in zip(cuda_generators, cuda_states):
        assert torch.equal(generator.get_state(), expected)


def test_evaluate_model_reports_force_mae_for_molecular_outputs():
    dataset = TensorDictDataset(torch.randn(6, 4, 3), torch.randn(6, 4, 3))
    loader = DataLoader(dataset, batch_size=3)
    model = ZeroModel()
    metrics = evaluate_model(model, loader, device=torch.device("cpu"), transform=None)
    assert "mae" in metrics
    assert "force_mae" in metrics
