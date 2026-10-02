from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_finite_objective_comparison.py"
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


def test_csv_summary_records_finite_labels_weights_and_jvp_status(monkeypatch):
    module = _load_module()
    monkeypatch.setattr(sys, "argv", ["make_finite_objective_comparison.py"])
    specs = module._build_specs(module.parse_args())
    rows = [
        {"objective": spec.objective, "seed": seed, "relative_l2": 0.4}
        for spec in specs
        for seed in spec.expected_seeds
    ]
    aggregate = module._aggregate(pd.DataFrame(rows), specs).set_index("objective")
    assert aggregate.loc["normalized finite", "method_label"] == "Augmentation + normalized finite"
    assert aggregate.loc["normalized finite", "weight"] == 0.1
    assert aggregate.loc["raw finite", "weight"] == 2.4
    assert aggregate.loc["normalized finite", "normalized_by_epsilon"] == "true"
    assert aggregate.loc["raw finite", "normalized_by_epsilon"] == "false"
    assert bool(aggregate.loc["tangent", "uses_jvp"])
    assert not bool(aggregate.loc["normalized finite", "uses_jvp"])
    assert set(aggregate["seed_count"]) == {5}
