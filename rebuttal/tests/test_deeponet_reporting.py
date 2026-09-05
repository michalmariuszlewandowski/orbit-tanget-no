from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_deeponet_backbone_table.py"
    spec = importlib.util.spec_from_file_location("make_deeponet_backbone_table", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_method_rows(module, spec):
    rows = []
    for seed in module.EXPECTED_SEEDS:
        row = dict(module._expected_protocol(spec))
        row.update(
            {
                "seed": seed,
                "config.seed": seed,
                "environment.git_commit": "0123456789abcdef",
                "environment.platform": "test-platform",
                "environment.python": "3.12.0",
                "environment.torch": "test-torch",
                "environment.cuda_available": False,
                "environment.cuda_version": None,
                "environment.cudnn_version": None,
                "relative_l2": 0.4,
                "orbit_ood_relative_l2": 0.5,
                "equivariance_defect_relative": 0.3,
                "best_val_relative_l2": 0.4,
                "latency_ms_per_sample": 5.0,
                "train_wall_seconds": 300.0,
            }
        )
        rows.append(row)
    return rows


def test_deeponet_seed_validation_rejects_incomplete_runs():
    module = _load_module()
    with pytest.raises(module.DeepONetProtocolError, match="expected seeds"):
        module._validate_seeds(pd.DataFrame({"seed": [23, 31, 47, 59]}), "test")


def _fno_reference_rows(module):
    return [
        {"method": spec.method, "model_name": "fno2d", "seed": seed}
        for spec in module.METHOD_SPECS
        for seed in module.EXPECTED_SEEDS
    ]


def test_deeponet_loads_exact_shared_fno_seed_set(tmp_path):
    module = _load_module()
    path = tmp_path / "fno.runs.csv"
    pd.DataFrame(_fno_reference_rows(module)).to_csv(path, index=False)
    assert module.load_fno_reference_seeds(path) == module.EXPECTED_SEEDS


def test_deeponet_rejects_inconsistent_fno_seed_sets(tmp_path):
    module = _load_module()
    rows = _fno_reference_rows(module)
    rows = [
        row
        for row in rows
        if not (row["method"] == "aug_orbit" and row["seed"] == 71)
    ]
    path = tmp_path / "fno.runs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(module.DeepONetProtocolError, match="shared FNO seeds"):
        module.load_fno_reference_seeds(path)


def test_deeponet_primary_pair_uses_candidate_minus_reference():
    module = _load_module()
    rows = []
    for method, offset in (("aug", 0.0), ("aug_orbit", -0.1)):
        for index, seed in enumerate(module.EXPECTED_SEEDS):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "relative_l2": 0.5 + index * 0.01 + offset,
                    "orbit_ood_relative_l2": 0.6 + index * 0.01 + offset,
                    "equivariance_defect_relative": 0.4 + index * 0.01 + offset,
                }
            )
    paired = module.paired_primary(pd.DataFrame(rows)).set_index("metric")
    assert paired.loc["relative_l2", "paired_delta_mean"] == pytest.approx(-0.1)
    assert paired.loc["relative_l2", "relative_reduction_pct"] > 0
    assert bool(paired.loc["relative_l2", "ci_excludes_zero"])


def test_deeponet_table_discloses_forward_and_backward_counts():
    module = _load_module()
    rows = []
    for spec in module.METHOD_SPECS:
        row = {
            "method": spec.method,
            "method_label": spec.label,
            "steps_per_epoch": spec.steps_per_epoch,
            "model_forwards_total": spec.model_forwards_per_epoch * 150,
            "backward_passes_total": spec.steps_per_epoch * 150,
            "seeds": ",".join(str(seed) for seed in module.EXPECTED_SEEDS),
        }
        for metric in module.AGGREGATE_METRICS:
            row[f"{metric}_mean"] = 0.4
            row[f"{metric}_std"] = 0.01
        rows.append(row)
    tex = module.latex_table(pd.DataFrame(rows))
    assert "Model fwds., total" in tex
    assert "Bwd. passes, total" in tex
    assert "DeepONet + aug. + normalized LOCO" in tex
    assert "same training seeds as the FNO comparison" in tex
    assert r"\(\{23, 31, 47, 59, 71\}\)" in tex
    assert r"\mathbf{0.4000\pm0.0100}" in tex
    assert "Latency ms/sample" not in tex


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("config.symmetry.final_time", 1.0),
        ("config.symmetry.boost_x_channel", 2),
        ("config.dataset.dt", 0.002),
        ("config.dataset.dealias", False),
        ("config.training.latency_repeats", 10),
        ("environment.omp_num_threads", "4"),
        ("environment.torch", "different"),
    ],
)
def test_deeponet_method_validation_rejects_protocol_drift(
    monkeypatch, column, bad_value
):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0][column] = bad_value
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.DeepONetProtocolError, match=column):
        module._load_method(Path("unused"), spec)


def test_deeponet_method_validation_accepts_absent_or_false_resume(monkeypatch):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0]["config.runtime.resume"] = False
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    result = module._load_method(Path("unused"), spec)
    assert len(result) == len(module.EXPECTED_SEEDS)


def test_deeponet_method_validation_accepts_consistent_alternate_software(monkeypatch):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    for row in rows:
        row["environment.platform"] = "Linux-test"
        row["environment.python"] = "3.13.0"
        row["environment.torch"] = "future-test-build"
        row["environment.cuda_available"] = True
        row["environment.cuda_version"] = "test-cuda"
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    result = module._load_method(Path("unused"), spec)
    assert len(result) == len(module.EXPECTED_SEEDS)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("config.runtime.resume", True),
        ("config.training.stop_after_epochs", 20),
    ],
)
def test_deeponet_method_validation_rejects_resumed_or_chunked_runs(
    monkeypatch, column, value
):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0][column] = value
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.DeepONetProtocolError, match=column):
        module._load_method(Path("unused"), spec)


def test_deeponet_campaign_rejects_mixed_git_commits(monkeypatch):
    module = _load_module()
    method_rows = {}
    for spec in module.METHOD_SPECS:
        rows = _valid_method_rows(module, spec)
        if spec is module.METHOD_SPECS[-1]:
            rows[0]["environment.git_commit"] = "different-commit"
        method_rows[spec.relative_dir] = rows

    def fake_collect(path):
        return method_rows[Path(path).name]

    monkeypatch.setattr(module, "collect_run_rows", fake_collect)
    with pytest.raises(module.DeepONetProtocolError, match="git_commit"):
        module.build_run_frame(Path("unused"))
