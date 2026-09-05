from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_cno2d_backbone_table.py"
    spec = importlib.util.spec_from_file_location("make_cno2d_backbone_table", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_method_rows(module, spec):
    rows = []
    for seed in module.EXPECTED_SEEDS:
        row = dict(module._expected_protocol(spec))
        resume_spec = module.AUTHORIZED_RESUMES.get((spec.method, seed))
        config_hash = (
            resume_spec.resumed_config_hash if resume_spec is not None else f"config-{seed}"
        )
        row.update(
            {
                "run_dir": f"unused/seed_{seed}",
                "config_path": f"unused/seed_{seed}/config.yaml",
                "seed": seed,
                "config.seed": seed,
                "config_hash": config_hash,
                "environment.config_hash": config_hash,
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
        if resume_spec is not None:
            row["config.runtime.resume"] = True
        rows.append(row)
    return rows


def _fno_reference_rows(module):
    return [
        {"method": spec.method, "model_name": "fno2d", "seed": seed}
        for spec in module.METHOD_SPECS
        for seed in module.EXPECTED_SEEDS
    ]


def test_cno2d_resume_allowlist_matches_pause_record():
    module = _load_module()
    expected = {
        ("baseline", 23): (
            108,
            "2bd7a1e0f067fe3e5975816c69a5547991a34bc9b0a179871ed057f64e28b208",
            "9b9457c84e7e6925e842e2cfe0e38ac928340c9586ad50d86cda31b56019339e",
            "57d7580e4a21",
        ),
        ("aug_orbit", 59): (
            103,
            "37579559adb2f66305a4502aa1a913eafacf042fe40aef7cc49d45ce7e387a70",
            "a3031d0f375ff34ce1090203121b2c593f7ac47f8bcddd3390ebd6e78096eb02",
            "c3963fd57bc2",
        ),
    }
    actual = {
        key: (
            value.last_completed_epoch,
            value.last_checkpoint_sha256,
            value.best_checkpoint_sha256,
            value.resumed_config_hash,
        )
        for key, value in module.AUTHORIZED_RESUMES.items()
    }
    assert actual == expected


def test_cno2d_seed_validation_rejects_incomplete_runs():
    module = _load_module()
    with pytest.raises(module.CNO2dProtocolError, match="expected seeds"):
        module._validate_seeds(pd.DataFrame({"seed": [23, 31, 47, 59]}), "test")


def test_cno2d_loads_exact_shared_fno_seed_set(tmp_path):
    module = _load_module()
    path = tmp_path / "fno.runs.csv"
    pd.DataFrame(_fno_reference_rows(module)).to_csv(path, index=False)
    assert module.load_fno_reference_seeds(path) == module.EXPECTED_SEEDS


def test_cno2d_rejects_inconsistent_fno_seed_sets(tmp_path):
    module = _load_module()
    rows = _fno_reference_rows(module)
    rows = [
        row
        for row in rows
        if not (row["method"] == "aug_orbit" and row["seed"] == 71)
    ]
    path = tmp_path / "fno.runs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(module.CNO2dProtocolError, match="shared FNO seeds"):
        module.load_fno_reference_seeds(path)


def test_cno2d_primary_pair_uses_candidate_minus_reference():
    module = _load_module()
    rows = []
    for method, offset in (("aug", 0.0), ("aug_orbit", -0.1)):
        for index, seed in enumerate(module.EXPECTED_SEEDS):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "checkpoint_resumed": method == "aug_orbit" and seed == 59,
                    "relative_l2": 0.5 + index * 0.01 + offset,
                    "orbit_ood_relative_l2": 0.6 + index * 0.01 + offset,
                    "equivariance_defect_relative": 0.4 + index * 0.01 + offset,
                }
            )
    paired = module.paired_primary(pd.DataFrame(rows)).set_index("metric")
    assert paired.loc["relative_l2", "paired_delta_mean"] == pytest.approx(-0.1)
    assert paired.loc["relative_l2", "relative_reduction_pct"] > 0
    assert bool(paired.loc["relative_l2", "ci_excludes_zero"])
    assert paired.loc["relative_l2", "reference_checkpoint_resumed_seeds"] == ""
    assert paired.loc["relative_l2", "candidate_checkpoint_resumed_seeds"] == "59"


def test_cno2d_table_discloses_forward_and_backward_counts():
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
    assert "CNO2d + aug. + normalized LOCO" in tex
    assert "same training seeds as the FNO comparison" in tex
    assert r"\(\{23, 31, 47, 59, 71\}\)" in tex
    assert r"\mathbf{0.4000\pm0.0100}" in tex
    assert "Latency ms/sample" not in tex
    assert "Baseline seed 23" in tex
    assert "augmentation-plus-LOCO seed 59" in tex
    assert "segment-local wall times are excluded" in tex


def test_cno2d_aggregate_excludes_resumed_segment_wall_times():
    module = _load_module()
    rows = []
    for spec in module.METHOD_SPECS:
        for seed in module.EXPECTED_SEEDS:
            resumed = (spec.method, seed) in module.AUTHORIZED_RESUMES
            rows.append(
                {
                    "method": spec.method,
                    "method_label": spec.label,
                    "seed": seed,
                    "steps_per_epoch": spec.steps_per_epoch,
                    "model_forwards_total": spec.model_forwards_per_epoch * 150,
                    "backward_passes_total": spec.steps_per_epoch * 150,
                    "parameters": module.EXPECTED_PARAMETERS,
                    "checkpoint_resumed": resumed,
                    "relative_l2": 0.0 if resumed else 0.5,
                    "orbit_ood_relative_l2": 0.5,
                    "equivariance_defect_relative": 0.5,
                    "best_val_relative_l2": 0.5,
                    "latency_ms_per_sample": 5.0,
                    "train_wall_seconds": 1.0 if resumed else 300.0,
                }
            )

    aggregate = module.aggregate_runs(pd.DataFrame(rows)).set_index("method")
    assert aggregate.loc["baseline", "relative_l2_mean"] == pytest.approx(0.4)
    assert aggregate.loc["aug_orbit", "relative_l2_mean"] == pytest.approx(0.4)
    assert aggregate.loc["baseline", "train_wall_seconds_uninterrupted_mean"] == 300.0
    assert aggregate.loc["aug_orbit", "train_wall_seconds_uninterrupted_mean"] == 300.0
    assert aggregate.loc["baseline", "train_wall_seconds_uninterrupted_count"] == 4
    assert aggregate.loc["aug_orbit", "train_wall_seconds_uninterrupted_count"] == 4
    assert aggregate.loc["orbit", "train_wall_seconds_uninterrupted_count"] == 5
    assert aggregate.loc["aug", "train_wall_seconds_uninterrupted_count"] == 5
    assert "train_wall_seconds_mean" not in aggregate.columns


def test_cno2d_manifest_discloses_resume_provenance(monkeypatch):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == "baseline")
    rows = _valid_method_rows(module, spec)
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    run_df = module._load_method(Path("unused"), spec)
    manifest = module._manifest(run_df).set_index("seed")
    resumed = manifest.loc[23]
    assert bool(resumed["checkpoint_resumed"])
    assert resumed["resume_last_completed_epoch"] == 108
    assert resumed["resume_authorization_source"] == module.RESUME_AUTHORIZATION_SOURCE
    assert resumed["train_wall_seconds_scope"] == "post_resume_segment"
    assert manifest.loc[31, "train_wall_seconds_scope"] == "full_run"


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("config.symmetry.final_time", 1.0),
        ("config.symmetry.boost_x_channel", 2),
        ("config.dataset.dt", 0.002),
        ("config.dataset.dealias", False),
        ("config.model.channel_multiplier", 36),
        ("config.model.resample_halo", 4),
        ("config.model.use_batch_norm", True),
        ("parameters_total", 2_667_742),
        ("config.training.latency_repeats", 10),
        ("environment.omp_num_threads", "4"),
        ("environment.torch", "different"),
    ],
)
def test_cno2d_method_validation_rejects_protocol_drift(
    monkeypatch, column, bad_value
):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0][column] = bad_value
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match=column):
        module._load_method(Path("unused"), spec)


def test_cno2d_method_validation_accepts_absent_or_false_resume(monkeypatch):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0]["config.runtime.resume"] = False
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    result = module._load_method(Path("unused"), spec)
    assert len(result) == len(module.EXPECTED_SEEDS)


@pytest.mark.parametrize(
    ("method", "seed", "last_completed_epoch"),
    [("baseline", 23, 108), ("aug_orbit", 59, 103)],
)
def test_cno2d_method_validation_accepts_only_declared_resumes(
    monkeypatch, method, seed, last_completed_epoch
):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == method)
    rows = _valid_method_rows(module, spec)
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    result = module._load_method(Path("unused"), spec).set_index("seed")
    resumed = result.loc[seed]
    authorization = module.AUTHORIZED_RESUMES[(method, seed)]
    assert bool(resumed["checkpoint_resumed"])
    assert resumed["resume_last_completed_epoch"] == last_completed_epoch
    assert (
        resumed["resume_source_last_checkpoint_sha256"]
        == authorization.last_checkpoint_sha256
    )
    assert (
        resumed["resume_source_best_checkpoint_sha256"]
        == authorization.best_checkpoint_sha256
    )
    assert resumed["train_wall_seconds_scope"] == "post_resume_segment"
    assert set(result.index[result["checkpoint_resumed"]]) == {seed}
    assert set(result.loc[~result["checkpoint_resumed"], "train_wall_seconds_scope"]) == {
        "full_run"
    }


@pytest.mark.parametrize("method", ["baseline", "aug_orbit"])
def test_cno2d_method_validation_requires_declared_resume(monkeypatch, method):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == method)
    rows = _valid_method_rows(module, spec)
    resumed_seed = next(
        seed
        for authorized_method, seed in module.AUTHORIZED_RESUMES
        if authorized_method == method
    )
    resumed_row = next(row for row in rows if row["seed"] == resumed_seed)
    resumed_row.pop("config.runtime.resume")
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match="expected authorized checkpoint"):
        module._load_method(Path("unused"), spec)


@pytest.mark.parametrize(
    ("method", "seed"),
    [("baseline", 31), ("orbit", 23), ("aug", 71), ("aug_orbit", 71)],
)
def test_cno2d_method_validation_rejects_undeclared_resume(
    monkeypatch, method, seed
):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == method)
    rows = _valid_method_rows(module, spec)
    next(row for row in rows if row["seed"] == seed)["config.runtime.resume"] = True
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match="unauthorized checkpoint resume"):
        module._load_method(Path("unused"), spec)


def test_cno2d_method_validation_rejects_non_boolean_resume(monkeypatch):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == "aug")
    rows = _valid_method_rows(module, spec)
    rows[0]["config.runtime.resume"] = "true"
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match="boolean or absent"):
        module._load_method(Path("unused"), spec)


def test_cno2d_method_validation_rejects_wrong_authorized_config_hash(monkeypatch):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == "aug_orbit")
    rows = _valid_method_rows(module, spec)
    resumed = next(row for row in rows if row["seed"] == 59)
    resumed["config_hash"] = "internally-consistent-but-wrong"
    resumed["environment.config_hash"] = "internally-consistent-but-wrong"
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match="resumed config hash mismatch"):
        module._load_method(Path("unused"), spec)


def test_cno2d_method_validation_rejects_chunking_on_authorized_resume(monkeypatch):
    module = _load_module()
    spec = next(spec for spec in module.METHOD_SPECS if spec.method == "baseline")
    rows = _valid_method_rows(module, spec)
    resumed = next(row for row in rows if row["seed"] == 23)
    resumed["config.training.stop_after_epochs"] = 20
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match="stop_after_epochs"):
        module._load_method(Path("unused"), spec)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("config.runtime.resume", True),
        ("config.training.stop_after_epochs", 20),
    ],
)
def test_cno2d_method_validation_rejects_resumed_or_chunked_runs(
    monkeypatch, column, value
):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0][column] = value
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match=column):
        module._load_method(Path("unused"), spec)


def test_cno2d_method_validation_rejects_config_hash_mismatch(monkeypatch):
    module = _load_module()
    spec = module.METHOD_SPECS[-1]
    rows = _valid_method_rows(module, spec)
    rows[0]["environment.config_hash"] = "different"
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match="config hash mismatch"):
        module._load_method(Path("unused"), spec)


def test_cno2d_campaign_rejects_mixed_git_commits(monkeypatch):
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
    with pytest.raises(module.CNO2dProtocolError, match="git_commit"):
        module.build_run_frame(Path("unused"))


def _fresh_method_rows(module, spec):
    rows = _valid_method_rows(module, spec)
    for row in rows:
        row.update({"environment.git_commit": None, "environment.omp_num_threads": "1",
                    "environment.mkl_num_threads": "1", "source_sha256": "new-source",
                    "dataset_sha256": "regenerated-container", "config.runtime.resume": False})
        row.pop("config.runtime.source_snapshot")
        row.pop("config.runtime.source_snapshot_sha256")
    return rows


def test_cno2d_fresh_protocol_uses_current_provenance_not_historical_resume(monkeypatch):
    module = _load_module()
    spec = module.METHOD_SPECS[0]
    rows = _fresh_method_rows(module, spec)
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    result = module._load_method(Path("unused"), spec, fresh_runs=True)
    assert not result["checkpoint_resumed"].any()
    assert set(result["train_wall_seconds_scope"]) == {"full_run"}
    # Historical mode remains strict about the original campaign.
    with pytest.raises(module.CNO2dProtocolError):
        module._load_method(Path("unused"), spec)


@pytest.mark.parametrize("column,value", [
    ("config.runtime.source_snapshot", "historical-snapshot.zip"),
    ("config.runtime.source_snapshot_sha256", "historical-hash"),
    ("config.runtime.resume", True),
])
def test_cno2d_fresh_protocol_rejects_historical_provenance_claims(monkeypatch, column, value):
    module = _load_module()
    spec = module.METHOD_SPECS[0]
    rows = _fresh_method_rows(module, spec)
    rows[0][column] = value
    monkeypatch.setattr(module, "collect_run_rows", lambda _: rows)
    with pytest.raises(module.CNO2dProtocolError, match=column):
        module._load_method(Path("unused"), spec, fresh_runs=True)
