import copy
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from otno.training import trainer
from otno.utils import capture_rng_state, restore_rng_state


def _config(tmp_path):
    return {
        "seed": 17,
        "dataset": {"kind": "advection1d", "path": str(tmp_path / "data.pt"),
                    "n": 16, "num_train": 6, "num_val": 2, "num_test": 2,
                    "final_time": 0.25, "velocity": 0.7, "modes": 3, "seed": 7},
        "model": {"name": "fno1d", "in_channels": 1, "out_channels": 1,
                  "width": 4, "modes": 3, "depth": 1, "add_grid": True},
        "symmetry": {"name": "translation1d", "max_shift": 0.1},
        "training": {"method": "aug_orbit", "epochs": 3, "batch_size": 2,
                     "lr": 0.002, "lambda_orbit": 0.05, "lambda_aug": 0.5,
                     "eval_every": 1, "eval_orbit_samples": 2,
                     "latency_repeats": 1, "latency_warmup": 0, "num_workers": 0},
        "runtime": {"device": "cpu", "run_dir": str(tmp_path / "full"), "overwrite": False},
    }


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_rng_roundtrip_restores_all_available_streams():
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))
    cuda_expected = torch.rand(3, device="cuda") if torch.cuda.is_available() else None
    restore_rng_state(state)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(3), expected[2])
    if cuda_expected is not None:
        assert torch.equal(torch.rand(3, device="cuda"), cuda_expected)


@pytest.mark.parametrize(("steps", "method"), [(1, "aug_orbit"), (5, "aug_orbit"), (5, "semi_aug_orbit")])
def test_resume_matches_uninterrupted_training(tmp_path, steps, method):
    config = _config(tmp_path)
    config["training"]["steps_per_epoch"] = steps
    config["training"]["method"] = method
    if method == "semi_aug_orbit":
        config["training"].update(data_fraction=0.5, orbit_data_fraction=1.0, orbit_steps_per_epoch=steps)
    full = trainer.train_from_config(config)
    partial_config = copy.deepcopy(config)
    partial_config["runtime"]["run_dir"] = str(tmp_path / "resumed")
    partial_config["training"]["stop_after_epochs"] = 1
    partial = trainer.train_from_config(partial_config)
    assert partial["status"] == "partial"
    partial_config["runtime"]["resume"] = True
    partial_config["training"].pop("stop_after_epochs")
    resumed = trainer.train_from_config(partial_config)
    for filename in ("best.pt", "last.pt"):
        full_ckpt = _load(tmp_path / "full" / "checkpoints" / filename)
        resumed_ckpt = _load(tmp_path / "resumed" / "checkpoints" / filename)
        assert full_ckpt["epoch"] == resumed_ckpt["epoch"]
        for key, value in full_ckpt["model"].items():
            assert torch.equal(value, resumed_ckpt["model"][key]), key
    for key in ("relative_l2", "orbit_ood_relative_l2", "equivariance_defect_relative"):
        assert full[key] == resumed[key]
    assert resumed["train_wall_seconds"] > partial["train_wall_seconds"]


def test_final_checkpoint_keeps_last_weights_and_cli_reproduces_best_metrics(tmp_path, monkeypatch):
    config = _config(tmp_path)
    original_evaluate = trainer.evaluate_model
    calls = 0

    def increasing_validation(*args, **kwargs):
        nonlocal calls
        metrics = original_evaluate(*args, **kwargs)
        calls += 1
        if calls <= config["training"]["epochs"]:
            metrics["relative_l2"] = float(calls)
        return metrics

    monkeypatch.setattr(trainer, "evaluate_model", increasing_validation)
    results = trainer.train_from_config(config)
    run_dir = Path(config["runtime"]["run_dir"])
    best_path = run_dir / "checkpoints" / "best.pt"
    best = _load(best_path)
    last = _load(run_dir / "checkpoints" / "last.pt")
    assert best["epoch"] == 1 and last["epoch"] == 3
    assert any(not torch.equal(value, last["model"][key]) for key, value in best["model"].items())
    assert {"python", "numpy", "torch_cpu", "torch_cuda"} <= last["rng_state"].keys()
    out = tmp_path / "reevaluation.json"
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / "scripts" / "evaluate.py"),
               "--checkpoint", str(best_path), "--device", "cpu", "--out", str(out)]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    reevaluated = json.loads(out.read_text())
    for key in ("relative_l2", "orbit_ood_relative_l2", "equivariance_defect_relative", "epsilon_mean"):
        assert reevaluated[key] == results[key]
    assert reevaluated["eval_orbit_samples"] == 2
    assert reevaluated["eval_seed"] == config["seed"] + 200_000
    different = tmp_path / "different.pt"
    different.write_bytes(b"incorrect dataset")
    rejected = subprocess.run(command + ["--dataset", str(different)], cwd=root,
                              capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "SHA-256 does not match" in rejected.stderr


def test_semi_supervised_orbit_uses_full_inputs_without_their_labels(tmp_path, monkeypatch):
    config = _config(tmp_path)
    config["training"].update(method="semi_aug_orbit", epochs=1, data_fraction=1 / 3,
                              orbit_data_fraction=1.0, orbit_steps_per_epoch=4, steps_per_epoch=1)
    original = trainer.train_from_config(config)
    full_model = _load(tmp_path / "full/checkpoints/last.pt")["model"]
    original_loader = trainer._loader
    sizes = {}

    def poison_unlabeled_targets(*args, **kwargs):
        loader = original_loader(*args, **kwargs)
        if kwargs.get("seed_offset_override") == 40_000:
            sizes["orbit"] = len(loader.dataset)
            loader.dataset.u.fill_(float("nan"))
        elif args[1] == "train":
            sizes["supervised"] = len(loader.dataset)
        return loader

    monkeypatch.setattr(trainer, "_loader", poison_unlabeled_targets)
    config["runtime"]["run_dir"] = str(tmp_path / "poisoned")
    poisoned = trainer.train_from_config(config)
    assert sizes == {"supervised": 2, "orbit": 6}
    assert original["relative_l2"] == poisoned["relative_l2"]
    poisoned_checkpoint = _load(tmp_path / "poisoned/checkpoints/last.pt")
    for key, value in full_model.items():
        assert torch.equal(value, poisoned_checkpoint["model"][key])
    assert "orbit" in poisoned_checkpoint["loader_rng_states"]
    record = json.loads((tmp_path / "poisoned/train_metrics.jsonl").read_text())
    assert record["train_supervised_steps"] == 4
    assert "train_aug_loss" in record and "train_orbit_loss" in record


@pytest.mark.parametrize("method", ["mixed_semi_aug_orbit", "split_semi_aug_orbit"])
def test_unsupported_semi_methods_fail_before_writing_artifacts(tmp_path, method):
    config = _config(tmp_path)
    config["training"]["method"] = method
    with pytest.raises(ValueError, match="Unsupported training method"):
        trainer.train_from_config(config)
    assert not Path(config["runtime"]["run_dir"]).exists()
    assert not Path(config["dataset"]["path"]).exists()
