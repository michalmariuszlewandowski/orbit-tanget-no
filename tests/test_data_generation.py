from pathlib import Path


from otno.data.generators import generate_dataset_from_config
from otno.data.datasets import load_tensor_dataset


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
