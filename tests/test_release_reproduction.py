from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    spec = importlib.util.spec_from_file_location("reproduce_release", ROOT / "scripts/reproduce_release.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fresh_release_plan_has_all_training_and_report_dependencies(release, tmp_path):
    registry = release.load_registry()
    plan = release.build_plan(registry, release.selected_suites(registry, ["all"]))
    # The checkout includes the fixed rMD17 split.
    # Other data, checkpoints and intermediate tables must all have producers.
    preserved = next(step for step in plan if step.immutable)
    target = tmp_path / preserved.produces[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test historical input")
    preserved.sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    assert release.preflight(plan, tmp_path) == []
    assert all((ROOT / path).is_file() for path in release.source_files(registry))


def test_release_plan_includes_controls_and_excludes_unreported_pilots(release):
    registry = release.load_registry()
    plan = release.build_plan(registry, release.selected_suites(registry, ["all"]))
    runs = [step.protected_dir for step in plan if step.stage == "train"]
    assert len(runs) == len(set(runs))
    assert sum("2pct_cno2d_5seed" in run for run in runs) == 20
    assert sum("2pct_deeponet_5seed" in run for run in runs) == 20
    assert sum("10pct_lambda_0p1" in run for run in runs) == 6
    assert sum("rmd17_ethanol/labels_500" in run for run in runs) == 15
    assert not any("fraction_0.005" in run for run in runs)
    assert not any("labels_100/" in run or "lpsda" in run for run in runs)
    assert all("--overwrite" not in step.command for step in plan)


def test_report_manifest_accepts_best_checkpoint_without_last(release, tmp_path, monkeypatch):
    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "paper_artifact_report", ROOT / "scripts/reproduce_paper_artifacts.py"
    )
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    run_dir = tmp_path / "runs/example/seed_23"
    for name in release.TRAIN_FILES:
        path = run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    rows = pd.DataFrame([{"run_dir": "runs/example/seed_23", "seed": 23}])
    assert report._manifest_for_training(rows)[0]["complete"]
    (run_dir / "checkpoints/best.pt").unlink()
    missing = report._manifest_for_training(rows)[0]
    assert not missing["complete"]
    assert missing["missing_files"] == "checkpoints/best.pt"


def test_partial_training_is_rejected_before_starting_other_jobs(release, tmp_path):
    (tmp_path / "runs/partial").mkdir(parents=True)
    (tmp_path / "runs/partial/config.yaml").write_text("seed: 23", encoding="utf-8")
    plan = [release.Step("new", "train", [], produces=["runs/new/test_metrics.json"]),
            release.Step("partial", "train", [], produces=["runs/partial/test_metrics.json"],
                         protected_dir="runs/partial")]
    assert "Partial protected run" in release.preflight(plan, tmp_path)[0]
    assert not (tmp_path / "runs/new").exists()


def test_existing_report_does_not_hide_missing_raw_evidence(release, tmp_path):
    table = tmp_path / "cached.csv"
    table.write_text("preserved table", encoding="utf-8")
    step = release.Step("report", "reports", [], requires=["missing/test_metrics.json"],
                        produces=["cached.csv"])
    assert "missing prerequisite" in release.preflight([step], tmp_path)[0]
    assert table.read_text(encoding="utf-8") == "preserved table"


def test_preserved_split_is_required_and_hash_verified(release, tmp_path):
    step = release.Step("data:rmd17", "data", [], produces=["historical.pt"],
                        immutable=True, sha256=hashlib.sha256(b"correct").hexdigest())
    assert "cannot regenerate" in release.preflight([step], tmp_path)[0]
    (tmp_path / "historical.pt").write_bytes(b"wrong")
    assert "SHA-256 mismatch" in release.preflight([step], tmp_path)[0]
    assert (tmp_path / "historical.pt").read_bytes() == b"wrong"


def test_nonfinite_completed_metrics_do_not_count_as_a_completed_run(release, tmp_path):
    outputs = ["run/" + file for file in release.TRAIN_FILES]
    for output in outputs:
        target = tmp_path / output
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    metrics = dict.fromkeys(release.METRICS, 1.0)
    metrics.update(method="aug", relative_l2=float("nan"))
    (tmp_path / "run/test_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    step = release.Step("train", "train", [], produces=outputs, protected_dir="run")
    assert "Invalid completed metrics" in release.preflight([step], tmp_path)[0]


def test_registry_rejects_unknown_suite_cycles_and_escaping_paths(release, tmp_path):
    registry = {"suites": {"a": {"depends_on": ["b"]}, "b": {"depends_on": ["a"]}}}
    with pytest.raises(ValueError, match="Unknown suite"):
        release.selected_suites(registry, ["missing"])
    with pytest.raises(ValueError, match="Cyclic"):
        release.selected_suites(registry, ["a"])
    with pytest.raises(ValueError, match="repository-relative"):
        release._path(tmp_path, "../outside")


def test_existing_dataset_with_wrong_generation_parameters_is_rejected(release, tmp_path):
    import torch

    target = tmp_path / "data.pt"
    torch.save({"metadata": {"config_fingerprint": "different-parameters"}}, target)
    original = target.read_bytes()
    step = release.Step("data", "data", [], produces=["data.pt"],
                        expected_config={"dataset": {"n": 64, "seed": 31}})
    assert "does not match requested config" in release.preflight([step], tmp_path)[0]
    assert target.read_bytes() == original


def test_default_dry_run_never_invokes_a_child_command(release, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["reproduce_release.py", "--suite", "main", "--stage", "train"])

    def unexpected(*args, **kwargs):
        raise AssertionError("dry-run attempted to execute a child command")

    monkeypatch.setattr(release.subprocess, "run", unexpected)
    release.main()
    assert "scripts/train.py" in capsys.readouterr().out


def test_backbone_report_selects_fresh_audit_and_rejects_mixed_campaign(release, tmp_path):
    step = release.Step("cno", "reports", ["python", "table.py"],
                        fresh_report_inputs=["seed_23/meta.json", "seed_31/meta.json"])
    assert release.report_command(step, tmp_path)[-1] == "--fresh-runs"
    target = tmp_path / step.fresh_report_inputs[0]
    target.parent.mkdir()
    target.write_text('{"config_hash": "legacy"}', encoding="utf-8")
    with pytest.raises(ValueError, match="historical and fresh"):
        release.report_command(step, tmp_path)
    target.write_text('{"source_sha256": "new"}', encoding="utf-8")
    assert release.report_command(step, tmp_path)[-1] == "--fresh-runs"


@pytest.mark.parametrize("module_name", ["run_matrix", "run_adapt_matrix"])
def test_matrix_overrides_keep_yaml_types_and_literal_ellipses(release, module_name):
    import importlib
    from otno.config import parse_overrides

    runner = importlib.import_module(module_name)
    values = {"a": "false", "b": "001", "c": ["a...b"], "d": "", "e": None, "f": True}
    command = runner._command("base.yaml", values)
    parsed = parse_overrides([command[index + 1] for index, arg in enumerate(command) if arg == "--override"])
    assert parsed == values
    assert type(parsed["a"]) is str
    assert type(parsed["b"]) is str


def test_suites_execute_dependencies_before_dependents_regardless_of_registry_order(release):
    registry = {"suites": {"dependent": {"depends_on": ["base"]}, "base": {}}}
    assert release.selected_suites(registry, ["dependent"]) == ["base", "dependent"]


@pytest.mark.parametrize("module_name", ["run_ood_severity_matrix", "run_observable_canonicalization_matrix"])
def test_evaluation_dry_run_does_not_write_or_replace_reports(release, tmp_path, monkeypatch, module_name):
    import importlib

    runner = importlib.import_module(module_name)
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text("evaluations: []\n", encoding="utf-8")
    for directory in (tmp_path, tmp_path / "missing"):
        prefix = directory / "report"
        if directory.exists():
            for suffix in (".runs.csv", ".aggregate.csv", ".aggregate.tex"):
                prefix.with_suffix(suffix).write_bytes(b"published table")
        before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        monkeypatch.setattr(sys, "argv", [module_name, "--matrix", str(matrix), "--out-prefix", str(prefix), "--dry-run"])
        runner.main()
        after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
        assert after == before
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("module_name", ["run_ood_severity_matrix", "run_observable_canonicalization_matrix"])
def test_evaluation_cache_reuse_requires_unchanged_job_checkpoint_data_and_metrics(
    release, tmp_path, monkeypatch, module_name, request
):
    import importlib
    import torch
    import yaml
    from otno.models import build_model
    from otno.utils import file_sha256

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    request.addfinalizer(lambda: torch.set_num_threads(threads))
    runner = importlib.import_module(module_name)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    cfg = {
        "model": {"name": "fno2d", "in_channels": 3, "out_channels": 1, "width": 4, "modes": 2, "depth": 1},
        "dataset": {"path": "data.pt"},
        "training": {"batch_size": 2},
    }
    a = torch.randn(2, 8, 8, 3)
    a[..., 1:] = 0
    torch.save({"splits": {"test": {"a": a, "u": a[..., :1]}}}, tmp_path / "data.pt")
    checkpoint = {"model": build_model(cfg).state_dict(), "config": cfg, "meta": {"dataset_sha256": "0" * 64}}
    torch.save(checkpoint, tmp_path / "checkpoint.pt")
    job = {
        "checkpoint": "checkpoint.pt", "out_dir": "evaluation", "seed": 23, "method": "baseline",
        "severity": "small", "severity_scale": 1.0, "orbit_samples": 1,
        "latency_repeats": 1, "latency_warmup": 0,
        "symmetry": {"name": "navier_stokes2d_galilean", "max_boost": 0.01},
    }
    matrix = tmp_path / "matrix.yaml"
    matrix.write_text(yaml.safe_dump({"evaluations": [job]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [module_name, "--matrix", str(matrix), "--out-prefix", str(tmp_path / "table"), "--device", "cpu"])
    with pytest.raises(ValueError, match="Dataset SHA-256 does not match checkpoint"):
        runner.main()
    assert not (tmp_path / "evaluation").exists()
    checkpoint["meta"]["dataset_sha256"] = file_sha256(tmp_path / "data.pt")
    torch.save(checkpoint, tmp_path / "checkpoint.pt")
    runner.main()
    relative = "evaluation/severity_metrics.json" if module_name == "run_ood_severity_matrix" else "evaluation/observable_canonicalization/metrics.json"
    output = tmp_path / relative
    original_metrics = output.read_bytes()

    def no_recompute(*args, **kwargs):
        raise AssertionError("Unchanged cache should be reused")

    evaluator = "evaluate_model" if module_name == "run_ood_severity_matrix" else "evaluate_observable_canonicalization"
    monkeypatch.setattr(runner, evaluator, no_recompute)
    runner.main()
    assert output.read_bytes() == original_metrics
    for changed in (matrix, tmp_path / "checkpoint.pt", tmp_path / "data.pt", output):
        original = changed.read_bytes()
        if changed == matrix:
            job["symmetry"]["max_boost"] = 0.02
            changed.write_text(yaml.safe_dump({"evaluations": [job]}), encoding="utf-8")
        else:
            changed.write_bytes(original + b" ")
        with pytest.raises(ValueError, match="inputs or metrics changed"):
            runner.main()
        changed.write_bytes(original)


def test_execute_rejects_unverified_partial_evaluation_before_launching_training(
    release, tmp_path, monkeypatch
):
    output = tmp_path / "evaluation/metrics.json"
    output.parent.mkdir()
    output.write_text('{"relative_l2": 0.1}', encoding="utf-8")
    plan = [
        release.Step("train", "train", ["training"], produces=["new/checkpoints/best.pt"]),
        release.Step("evaluate", "evaluate", ["evaluation"],
                     produces=["evaluation/metrics.json", "missing.csv"],
                     evaluations=[({"checkpoint": "new/checkpoints/best.pt"}, "evaluation/metrics.json")]),
    ]
    assert release.preflight(plan, tmp_path) == []
    monkeypatch.setattr(release, "ROOT", tmp_path)
    monkeypatch.setattr(release, "load_registry", lambda *_: {"suites": {"test": {}}})
    monkeypatch.setattr(release, "source_files", lambda *_: set())
    monkeypatch.setattr(release, "build_plan", lambda *_: plan)
    monkeypatch.setattr(sys, "argv", ["reproduce_release.py", "--execute"])

    def no_training(*args, **kwargs):
        raise AssertionError("Preflight must reject stale caches before training starts")

    monkeypatch.setattr(release.subprocess, "run", no_training)
    with pytest.raises(SystemExit, match="Cached evaluation has no input record"):
        release.main()
    assert output.read_text(encoding="utf-8") == '{"relative_l2": 0.1}'


@pytest.mark.parametrize("module_name", ["run_matrix", "run_adapt_matrix"])
def test_continue_on_error_runs_remaining_jobs_but_reports_failure(release, monkeypatch, module_name):
    import importlib
    from types import SimpleNamespace

    runner = importlib.import_module(module_name)
    monkeypatch.setattr(runner, "_jobs", lambda _: [("first.yaml", {}), ("second.yaml", {})])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", [module_name, "--matrix", "unused.yaml", "--continue-on-error"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 1
    assert len(calls) == 2
