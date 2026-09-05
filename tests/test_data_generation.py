from pathlib import Path

import numpy as np
import pytest
import torch

from otno.data.datasets import load_tensor_dataset
from otno.data.generators import generate_dataset_from_config, validate_existing_dataset


def test_generate_tiny_advection_dataset(tmp_path: Path):
    path = tmp_path / "tiny.pt"
    cfg = {
        "dataset": {
            "kind": "advection1d",
            "path": str(path),
            "n": 16,
            "num_train": 4,
            "num_val": 2,
            "num_test": 2,
            "final_time": 0.1,
            "velocity": 1.0,
            "modes": 3,
            "seed": 123,
        }
    }
    out = generate_dataset_from_config(cfg, path)
    assert out.exists()
    ds, meta = load_tensor_dataset(path, "train")
    item = ds[0]
    assert item["a"].shape == (16, 1)
    assert item["u"].shape == (16, 1)
    assert meta["kind"] == "advection1d"


def test_generate_tiny_boosted_navier_stokes_dataset(tmp_path: Path):
    path = tmp_path / "tiny_boosted_ns.pt"
    cfg = {
        "dataset": {
            "kind": "navier_stokes_vorticity2d_boosted",
            "path": str(path),
            "n": 8,
            "num_train": 2,
            "num_val": 1,
            "num_test": 1,
            "final_time": 0.01,
            "dt": 0.005,
            "viscosity": 0.01,
            "smoothness": 4.0,
            "amplitude": 0.1,
            "max_boost": 0.2,
            "solver_batch_size": 2,
            "seed": 123,
        }
    }
    out = generate_dataset_from_config(cfg, path)
    assert out.exists()
    ds, meta = load_tensor_dataset(path, "train")
    item = ds[0]
    assert item["a"].shape == (8, 8, 3)
    assert item["u"].shape == (8, 8, 1)
    assert meta["kind"] == "navier_stokes_vorticity2d_boosted"


def test_generate_tiny_heat1d_dirichlet_dataset(tmp_path: Path):
    path = tmp_path / "tiny_heat_dirichlet.pt"
    cfg = {
        "dataset": {
            "kind": "heat1d_dirichlet",
            "path": str(path),
            "n": 16,
            "num_train": 4,
            "num_val": 2,
            "num_test": 2,
            "final_time": 0.1,
            "diffusivity": 0.01,
            "modes": 4,
            "seed": 123,
        }
    }
    out = generate_dataset_from_config(cfg, path)
    assert out.exists()
    ds, meta = load_tensor_dataset(path, "train")
    item = ds[0]
    assert item["a"].shape == (16, 1)
    assert item["u"].shape == (16, 1)
    assert meta["kind"] == "heat1d_dirichlet"
    assert abs(float(item["a"][0, 0])) < 1e-6
    assert abs(float(item["a"][-1, 0])) < 1e-6


def test_generate_tiny_rmd17_force_dataset(tmp_path: Path):
    source = tmp_path / "rmd17_ethanol.npz"
    coords = np.arange(12 * 3 * 3, dtype=np.float32).reshape(12, 3, 3)
    forces = np.ones_like(coords)
    charges = np.array([6, 1, 8], dtype=np.int64)
    np.savez(
        source,
        nuclear_charges=charges,
        coords=coords,
        forces=forces,
        energies=np.arange(12, dtype=np.float32),
        old_indices=np.arange(12, dtype=np.int64),
    )
    path = tmp_path / "rmd17_ethanol.pt"
    cfg = {
        "dataset": {
            "kind": "rmd17_force",
            "path": str(path),
            "source_path": str(source),
            "molecule": "ethanol",
            "num_train": 5,
            "num_val": 3,
            "num_test": 2,
            "split_seed": 123,
        }
    }
    out = generate_dataset_from_config(cfg, path)
    assert out.exists()
    ds, meta = load_tensor_dataset(path, "train")
    item = ds[0]
    assert item["a"].shape == (3, 4)
    assert item["u"].shape == (3, 3)
    assert meta["kind"] == "rmd17_force"
    assert meta["n_atoms"] == 3


def test_existing_dataset_config_mismatch_raises(tmp_path: Path):
    path = tmp_path / "tiny.pt"
    cfg = {
        "dataset": {
            "kind": "advection1d",
            "path": str(path),
            "n": 16,
            "num_train": 4,
            "num_val": 2,
            "num_test": 2,
            "final_time": 0.1,
            "velocity": 1.0,
            "modes": 3,
            "seed": 123,
        }
    }
    generate_dataset_from_config(cfg, path)
    changed = {"dataset": {**cfg["dataset"], "velocity": 2.0}}
    try:
        generate_dataset_from_config(changed, path)
    except ValueError as exc:
        assert "metadata does not match" in str(exc)
    else:
        raise AssertionError("expected metadata mismatch")


@pytest.mark.parametrize("fraction", [-0.1, float("nan"), 1.1])
def test_dataset_fraction_must_select_a_valid_subset(tmp_path, fraction):
    path = tmp_path / "data.pt"
    values = torch.arange(4.0).reshape(4, 1, 1)
    torch.save({"splits": {"train": {"a": values, "u": values}}}, path)
    with pytest.raises(ValueError, match="fraction"):
        load_tensor_dataset(path, "train", fraction=fraction)


def test_generator_rejects_fractional_sample_counts_before_writing(tmp_path):
    path = tmp_path / "invalid.pt"
    config = {"kind": "advection1d", "n": 8, "num_train": 2.5,
              "num_val": 1, "num_test": 1}
    with pytest.raises(ValueError, match="num_train"):
        generate_dataset_from_config(config, path)
    assert not path.exists()


def test_dataset_validation_accepts_nested_allow_stale_option(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"metadata": {}}, path)
    validate_existing_dataset(path, {"dataset": {"allow_stale": True}})
