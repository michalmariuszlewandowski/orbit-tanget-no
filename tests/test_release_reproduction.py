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
    # A fresh extraction contains only this irreproducible historical split.
    # Other data, checkpoints and intermediate tables must all have producers.
    preserved = next(step for step in plan if step.immutable)
    target = tmp_path / preserved.produces[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(b"test historical input")
    preserved.sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    assert release.preflight(plan, tmp_path) == []
    assert all((ROOT / path).is_file() for path in release.source_files(registry))


def test_v11_plan_includes_controls_and_excludes_unreported_pilots(release):
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
