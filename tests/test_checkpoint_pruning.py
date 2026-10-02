from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture
def pruning(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prune_release_checkpoints", scripts / "prune_release_checkpoints.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def completed_run(pruning, tmp_path, monkeypatch):
    from reproduce_release import Step

    run = "runs/example/seed_23"
    checkpoint = tmp_path / run / "checkpoints/last.pt"
    checkpoint.parent.mkdir(parents=True)
    best = checkpoint.with_name("best.pt")
    best.write_bytes(b"validation-selected weights")
    metrics = {"relative_l2": 0.25}
    metrics_path = tmp_path / run / "test_metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    torch.save({"epoch": 3, "results": metrics}, checkpoint)
    step = Step(
        "train",
        "train",
        [],
        protected_dir=run,
        produces=[f"{run}/checkpoints/best.pt", f"{run}/test_metrics.json"],
        expected_config={"training": {"epochs": 3}},
    )
    monkeypatch.setattr(pruning, "completed", lambda *_: True)
    return step, checkpoint, best, metrics_path


def test_preview_and_execute_preserve_selected_weights_and_metrics(
    pruning, completed_run, tmp_path
):
    step, checkpoint, best, metrics = completed_run
    originals = {path: path.read_bytes() for path in (best, metrics)}
    selected, skipped = pruning.candidates(tmp_path, [step])
    assert len(selected) == 1 and skipped == []
    assert checkpoint.exists()  # Preview is read-only.
    size = checkpoint.stat().st_size
    manifest = tmp_path / "cleanup.json"
    assert pruning.remove_candidates(tmp_path, selected, manifest) == size
    assert not checkpoint.exists()
    assert all(path.read_bytes() == data for path, data in originals.items())
    report = json.loads(manifest.read_text())
    assert report["removed"] == [selected[0]["path"]]
    assert report["removed_bytes"] == size
    assert pruning.candidates(tmp_path, [step]) == ([], [])


@pytest.mark.parametrize("case", ["incomplete", "referenced", "unfinished_epoch", "stale_results"])
def test_unfinished_or_required_checkpoints_are_kept(
    pruning, completed_run, tmp_path, monkeypatch, case
):
    step, checkpoint, _, _ = completed_run
    if case == "incomplete":
        monkeypatch.setattr(pruning, "completed", lambda *_: False)
    elif case == "referenced":
        step.requires.append(checkpoint.relative_to(tmp_path).as_posix())
    else:
        torch.save({"epoch": 2 if case == "unfinished_epoch" else 3, "results": {}}, checkpoint)
    selected, skipped = pruning.candidates(tmp_path, [step])
    assert selected == [] and len(skipped) == 1
    assert checkpoint.exists()


@pytest.mark.parametrize("changed", ["last.pt", "best.pt", "test_metrics.json"])
def test_changed_files_abort_before_deletion(pruning, completed_run, tmp_path, changed):
    step, checkpoint, best, metrics = completed_run
    selected, _ = pruning.candidates(tmp_path, [step])
    target = {path.name: path for path in (checkpoint, best, metrics)}[changed]
    with target.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed since inspection"):
        pruning.remove_candidates(tmp_path, selected, tmp_path / "cleanup.json")
    assert checkpoint.exists()
    assert not (tmp_path / "cleanup.json").exists()


def test_existing_audit_is_never_overwritten(pruning, completed_run, tmp_path):
    step, checkpoint, _, _ = completed_run
    selected, _ = pruning.candidates(tmp_path, [step])
    manifest = tmp_path / "cleanup.json"
    manifest.write_text("previous cleanup")
    with pytest.raises(FileExistsError):
        pruning.remove_candidates(tmp_path, selected, manifest)
    assert checkpoint.exists()
    assert manifest.read_text() == "previous cleanup"


def test_escaping_path_is_rejected(pruning, tmp_path):
    with pytest.raises(ValueError, match="repository-relative"):
        pruning._local_file(tmp_path, "../outside/checkpoints/last.pt")


def test_windows_reparse_point_is_rejected(pruning, tmp_path, monkeypatch):
    from types import SimpleNamespace

    target = tmp_path / "last.pt"
    target.write_bytes(b"checkpoint")
    original = Path.lstat

    def lstat(path, *args, **kwargs):
        if path == target:
            return SimpleNamespace(st_mode=0o100644, st_file_attributes=0x400)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="linked cleanup path"):
        pruning._local_file(tmp_path, "last.pt")
