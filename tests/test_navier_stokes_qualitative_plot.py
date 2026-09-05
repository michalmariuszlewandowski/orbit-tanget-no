from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "plot_navier_stokes_qualitative.py"
    spec = importlib.util.spec_from_file_location("plot_navier_stokes_qualitative", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gradient_magnitude_is_zero_for_a_constant_periodic_field():
    module = _load_module()
    target = torch.ones(3, 8, 8, 1)
    gradient = module._gradient_magnitude(target)
    assert gradient.shape == (3, 8, 8)
    assert torch.count_nonzero(gradient) == 0


def test_gradient_binned_mae_uses_per_example_absolute_errors():
    module = _load_module()
    generator = torch.Generator().manual_seed(7)
    target = torch.randn(4, 8, 8, 1, generator=generator)
    prediction = target + 2.0
    result = module._gradient_binned_mae(target, prediction, bins=5)
    assert result["percentile_midpoints"] == pytest.approx([10, 30, 50, 70, 90])
    assert result["mean_absolute_error"] == pytest.approx([2.0] * 5)
    assert result["normal_approximation_ci95_half_width"] == pytest.approx([0.0] * 5)
    assert len(result["gradient_edges"]) == 6


def test_relative_defects_compare_direct_and_transformed_predictions():
    module = _load_module()
    fields = {
        "aug_ood": torch.full((2, 4, 4, 1), 2.0),
        "aug_equiv": torch.ones(2, 4, 4, 1),
        "loco_ood": torch.ones(2, 4, 4, 1),
        "loco_equiv": torch.ones(2, 4, 4, 1),
    }
    defects = module._relative_defects(fields)
    assert defects["aug"].tolist() == pytest.approx([1.0, 1.0])
    assert defects["loco"].tolist() == pytest.approx([0.0, 0.0])


def test_frozen_input_validation_rejects_protocol_drift(tmp_path):
    module = _load_module()
    aug = tmp_path / "aug.pt"
    loco = tmp_path / "loco.pt"
    dataset = tmp_path / "dataset.pt"
    aug.write_bytes(b"aug")
    loco.write_bytes(b"loco")
    dataset.write_bytes(b"dataset")
    run_spec = {"seed": 23, "data_fraction": 0.02}
    figure_config = {
        "augmentation_checkpoint_sha256": module.file_sha256(aug),
        "loco_checkpoint_sha256": module.file_sha256(loco),
        "dataset_sha256": module.file_sha256(dataset),
        "expected_run_spec": {"seed": 23, "data_fraction": 0.02},
    }
    module._validate_frozen_figure_inputs(
        figure_config,
        aug_checkpoint=aug,
        loco_checkpoint=loco,
        dataset_path=dataset,
        run_spec=run_spec,
    )
    figure_config["expected_run_spec"]["seed"] = 31
    with pytest.raises(ValueError, match="protocol mismatch"):
        module._validate_frozen_figure_inputs(
            figure_config,
            aug_checkpoint=aug,
            loco_checkpoint=loco,
            dataset_path=dataset,
            run_spec=run_spec,
        )
