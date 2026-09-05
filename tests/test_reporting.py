import json
import importlib.util
import math
from pathlib import Path

import pandas as pd
import pytest
from scipy.stats import t as student_t

from otno.reporting import RunValidation, aggregate_runs, collect_run_rows, drop_empty_config_columns


def test_collect_run_rows_adds_label_fraction_from_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "test_metrics.json").write_text(
        json.dumps({"method": "aug_orbit", "relative_l2": 0.1}),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        "\n".join(
            [
                "training:",
                "  data_fraction: 0.25",
                "  lambda_orbit: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    rows = collect_run_rows(tmp_path)

    assert rows[0]["data_fraction"] == 0.25
    assert rows[0]["lambda_orbit"] == 0.5
    assert rows[0]["config.training.data_fraction"] == 0.25


def test_aggregate_runs_groups_by_fraction_and_flattens_columns():
    df = pd.DataFrame(
        [
            {
                "method": "aug",
                "model_name": "fno1d",
                "dataset_kind": "burgers1d",
                "data_fraction": 0.05,
                "relative_l2": 0.2,
            },
            {
                "method": "aug",
                "model_name": "fno1d",
                "dataset_kind": "burgers1d",
                "data_fraction": 0.25,
                "relative_l2": 0.1,
            },
        ]
    )

    agg = aggregate_runs(df)

    assert "relative_l2_mean" in agg.columns
    assert len(agg.columns) == len(set(agg.columns))
    assert set(agg["data_fraction"]) == {0.05, 0.25}


def test_filtered_tables_drop_unrelated_config_fields_but_keep_metric_and_provenance_schema():
    all_runs = pd.DataFrame([
        {"method": "fno", "config.dataset.n": 64, "environment.cuda_version": None,
         "relative_l2": None},
        {"method": "molecular", "config.dataset.molecule": "ethanol"},
    ])
    fno_runs = all_runs[all_runs["method"] == "fno"]
    table = drop_empty_config_columns(fno_runs)
    assert "config.dataset.molecule" not in table
    pd.testing.assert_frame_equal(table, fno_runs.drop(columns="config.dataset.molecule"))


@pytest.mark.parametrize("bad_seed", [23.5, True, float("nan"), float("inf")])
def test_protocol_rejects_noninteger_seeds(bad_seed):
    expected = (1,) if bad_seed is True else (23,)
    with pytest.raises(ValueError, match="seed"):
        RunValidation().seeds(pd.DataFrame({"seed": [bad_seed]}), "test", expected)


def _table_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"make_{name}_tables.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _comparison_rows(module_name, reference, candidate):
    module = _table_script(module_name)
    if module_name == "reviewer":
        return module._paired_rows_from_frames(
            reference, candidate, setting="test", reference_label="aug", candidate_label="aug_orbit"
        )
    runs = pd.concat([reference.assign(method="aug"), candidate.assign(method="aug_orbit")])
    return module._paired_rows(runs).to_dict("records")


def _paired_input(values):
    return pd.DataFrame([
        {"seed": 23 + index, "dataset_sha256": "same-dataset", "relative_l2": value,
         "orbit_ood_relative_l2": value, "equivariance_defect_relative": value,
         "force_mae": value, "orbit_ood_force_mae": value}
        for index, value in enumerate(values)
    ])


@pytest.mark.parametrize("module_name", ["reviewer", "rmd17"])
@pytest.mark.parametrize("problem", ["duplicate", "missing", "nan", "inf", "dataset", "single"])
def test_paired_reports_reject_invalid_evidence(module_name, problem):
    reference = _paired_input([1.0, 2.0])
    candidate = _paired_input([0.8, 1.6])
    if problem == "duplicate":
        candidate = pd.concat([candidate, candidate.iloc[[0]]])
    elif problem == "missing":
        candidate = candidate.iloc[[0]]
    elif problem in {"nan", "inf"}:
        candidate.loc[0, "relative_l2"] = float(problem)
    elif problem == "dataset":
        candidate["dataset_sha256"] = "other-dataset"
    else:
        reference, candidate = reference.iloc[[0]], candidate.iloc[[0]]
    with pytest.raises(ValueError):
        _comparison_rows(module_name, reference, candidate)


@pytest.mark.parametrize("module_name", ["reviewer", "rmd17"])
def test_paired_reports_use_student_t_beyond_ten_degrees_of_freedom(module_name):
    reference = _paired_input([1.0] * 12)
    candidate = _paired_input([1.0 - index / 100 for index in range(12)])
    result = _comparison_rows(module_name, reference, candidate)[0]
    delta = candidate["relative_l2"] - reference["relative_l2"]
    expected_half_width = student_t.ppf(0.975, 11) * delta.std(ddof=1) / math.sqrt(12)
    assert result["paired_delta_ci95_high"] == pytest.approx(delta.mean() + expected_half_width)


@pytest.mark.parametrize("module_name", ["reviewer", "rmd17"])
def test_zero_reference_mean_keeps_absolute_comparison(module_name):
    result = _comparison_rows(module_name, _paired_input([0.0, 0.0]), _paired_input([0.1, 0.2]))[0]
    assert result["paired_delta_mean"] == pytest.approx(0.15)
    assert math.isnan(result["relative_reduction_pct"])


@pytest.mark.parametrize("problem", ["nan", "inf", "duplicate", "dataset"])
def test_rmd17_aggregate_rejects_invalid_evidence(problem):
    runs = _paired_input([1.0, 2.0, 3.0]).assign(method="aug")
    if problem == "duplicate":
        runs = pd.concat([runs, runs.iloc[[0]]])
    elif problem == "dataset":
        runs.loc[0, "dataset_sha256"] = "other-dataset"
    else:
        runs.loc[0, "force_mae"] = float(problem)
    with pytest.raises(ValueError):
        _table_script("rmd17")._aggregate(runs)


def test_rmd17_run_root_filter_excludes_similarly_named_campaigns(tmp_path):
    root = tmp_path / "labels_500"
    df = pd.DataFrame({"run_dir": [str(root / "aug" / "seed_23"),
                                    str(tmp_path / "labels_5000" / "aug" / "seed_31")]})
    selected = _table_script("rmd17")._filter_runs(df, str(root))
    assert list(selected["run_dir"]) == [str(root / "aug" / "seed_23")]


def test_rmd17_single_seed_summary_does_not_claim_zero_variance():
    runs = _paired_input([1.0]).assign(method="aug")
    result = _table_script("rmd17")._aggregate(runs).iloc[0]
    assert result["seed_count"] == 1
    assert result["relative_l2_mean"] == 1.0
    assert math.isnan(result["relative_l2_std"])
