from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "make_finite_objective_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("make_finite_objective_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _raw_spec(module):
    return module.ObjectiveSpec(
        objective="raw finite",
        method_label="Augmentation + raw finite",
        run_root=Path("unused"),
        expected_seeds=(23, 31, 47, 59, 71),
        training_method="aug_orbit",
        steps_per_epoch=4,
        weight_name="lambda_raw",
        weight_config_key="config.training.lambda_orbit",
        weight=2.4,
        normalization="none (raw finite MSE)",
        normalized_by_epsilon=False,
        normalization_eta=None,
        uses_jvp=False,
        batch_forward_evaluations_per_epoch=12,
    )


def test_seed_validation_rejects_incomplete_objective():
    module = _load_module()
    with pytest.raises(module.ComparisonValidationError, match=r"missing \[47, 59, 71\]"):
        module._validate_seed_set(pd.DataFrame({"seed": [23, 31]}), _raw_spec(module))


def test_seed_validation_rejects_duplicate_objective():
    module = _load_module()
    with pytest.raises(module.ComparisonValidationError, match=r"duplicate.*23: 2"):
        module._validate_seed_set(
            pd.DataFrame({"seed": [23, 23, 31, 47]}),
            _raw_spec(module),
        )


def test_tex_table_records_finite_labels_and_jvp_status():
    module = _load_module()
    rows = []
    for objective, method_label, uses_jvp, seed_count in (
        ("augmentation baseline", "Augmentation", False, 5),
        ("tangent", "Augmentation + tangent", True, 5),
        ("normalized finite", "Augmentation + normalized finite", False, 5),
        ("raw finite", "Augmentation + raw finite", False, 5),
    ):
        rows.append(
            {
                "objective": objective,
                "method_label": method_label,
                "uses_jvp": uses_jvp,
                "seed_count": seed_count,
                "relative_l2_mean": 0.4,
                "relative_l2_std": 0.01,
                "orbit_ood_relative_l2_mean": 0.5,
                "orbit_ood_relative_l2_std": 0.02,
                "equivariance_defect_relative_mean": 0.3,
                "equivariance_defect_relative_std": 0.03,
            }
        )
    tex = module._tex_table(pd.DataFrame(rows))
    assert r"Augmentation + normalized finite & $\lambda_{\rm orb}=0.1$" in tex
    assert r"Augmentation + raw finite & $\lambda_{\rm raw}=2.4$" in tex
    assert "Augmentation + tangent &" in tex
    assert "n/a & yes" in tex
