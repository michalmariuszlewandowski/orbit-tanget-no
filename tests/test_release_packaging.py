from __future__ import annotations

import importlib
import hashlib
import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def packager(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("make_anonymous_submission")
    monkeypatch.setattr(module, "ALLOWED_FILES", {"README.md"})
    monkeypatch.setattr(module, "ALLOWED_GLOBS", {"src/otno/**/*.py", "src/otno/*.py"})
    monkeypatch.setattr(module, "FORCE_INCLUDE_FILES", set())
    return module


def test_release_is_reproducible_complete_and_independent_of_workspace_notes(tmp_path, packager):
    (tmp_path / "README.md").write_text("Run scripts/train.py.\n", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/train.py").write_text(
        '# default: configs/base.yaml\nprint("ready")\n', encoding="utf-8"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/base.yaml").write_text("seed: 23\n", encoding="utf-8")
    (tmp_path / "src/otno/data").mkdir(parents=True)
    (tmp_path / "src/otno/data/generators.py").write_text("VALUE = 1\n", encoding="utf-8")
    # An excluded local note must neither enter nor block the selected release.
    (tmp_path / "notes.md").write_text("local.user" + "@" + "example.org", encoding="utf-8")
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    packager.build_archive(tmp_path, first)
    os.utime(tmp_path / "README.md", (1_700_000_000, 1_700_000_000))
    packager.build_archive(tmp_path, second)
    assert first.read_bytes() == second.read_bytes()
    assert packager.verify_release(first) == 4
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(first) as archive:
        assert set(archive.namelist()) == {
            "README.md",
            "scripts/train.py",
            "configs/base.yaml",
            "src/otno/data/generators.py",
            packager.MANIFEST_NAME,
        }
        archive.extractall(extracted)
    assert packager.verify_release(extracted) == 4
    (extracted / "configs/base.yaml").write_text("seed: 31\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch: configs/base.yaml"):
        packager.verify_release(extracted)


def test_release_fails_when_required_input_or_dependency_is_absent(tmp_path, packager):
    with pytest.raises(ValueError, match="Missing required release files: README.md"):
        packager.build_archive(tmp_path, tmp_path / "out.zip")
    (tmp_path / "README.md").write_text("Run scripts/train.py.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing release dependency scripts/train.py"):
        packager.build_archive(tmp_path, tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_release_rejects_private_paths_in_payload(tmp_path, packager):
    (tmp_path / "README.md").write_text("/" + "home/" + "researcher/project", encoding="utf-8")
    with pytest.raises(SystemExit):
        packager.build_archive(tmp_path, tmp_path / "out.zip")
    assert not (tmp_path / "out.zip").exists()


def test_release_rejects_unrecorded_archive_entries(tmp_path, packager):
    (tmp_path / "README.md").write_text("Release\n", encoding="utf-8")
    path = tmp_path / "out.zip"
    packager.build_archive(tmp_path, path)
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("unrecorded.txt", "extra bytes")
    with pytest.raises(ValueError, match="files absent from its manifest"):
        packager.verify_release(path)


def test_release_rejects_parent_path_in_manifest(packager):
    manifest = {"schema_version": 1, "files": [{"path": "../outside"}]}
    with pytest.raises(ValueError, match="Invalid or repeated manifest path"):
        packager._verify_contents(manifest, lambda _: pytest.fail("must validate before reading"))


def test_data_archive_requires_the_historical_dataset_hash(tmp_path, packager):
    import yaml

    (tmp_path / "data").mkdir()
    dataset = tmp_path / "data/evidence.pt"
    dataset.write_bytes(b"historical tensor")
    (tmp_path / "configs").mkdir()
    registry = {
        "datasets": {
            "evidence": {
                "path": "data/evidence.pt",
                "paper_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        }
    }
    (tmp_path / "configs/release.yaml").write_text(yaml.safe_dump(registry), encoding="utf-8")
    output = tmp_path / "data.zip"
    packager.build_data_archive(tmp_path, output)
    original_archive = output.read_bytes()
    assert packager.verify_release(output) == 1
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"data/evidence.pt", packager.DATA_MANIFEST_NAME}
    dataset.write_bytes(b"regenerated tensor with different bytes")
    with pytest.raises(ValueError, match="differs from paper SHA-256"):
        packager.build_data_archive(tmp_path, output)
    assert output.read_bytes() == original_archive


def test_anonymity_reference_exception_is_an_exact_repository_url(tmp_path, packager):
    scanner = importlib.import_module("check_anonymity")
    public_url = "https://github.com/" + "camlab-ethz/ConvolutionalNeuralOperator"
    path = tmp_path / "references.md"
    path.write_text(public_url + "\n" + public_url + "-private\n", encoding="utf-8")
    findings = scanner.check(tmp_path, [(path, "references.md")])
    assert len(findings) == 1
    assert "references.md:2:" in findings[0]


def test_evidence_archive_preserves_selected_runs_and_rejects_missing_records(
    tmp_path, packager, monkeypatch
):
    workflow = importlib.import_module("reproduce_release")
    monkeypatch.setattr(workflow, "load_registry", lambda _: {})
    monkeypatch.setattr(workflow, "selected_suites", lambda _, requested: requested)
    run_dir = "runs/selected/seed_23"
    produces = [
        f"{run_dir}/{name}"
        for name in (
            "config.yaml",
            "test_metrics.json",
            "checkpoints/best.pt",
            "checkpoints/last.pt",
        )
    ]
    plan = [SimpleNamespace(stage="train", produces=produces, protected_dir=run_dir)]
    monkeypatch.setattr(workflow, "build_plan", lambda *_: plan)
    for rel in produces:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original bytes")
    pilot = tmp_path / "runs/pilot/test_metrics.json"
    pilot.parent.mkdir(parents=True)
    pilot.write_bytes(b"unselected")
    output = tmp_path / "evidence.zip"
    packager.build_evidence_archive(tmp_path, output, ["main"])
    assert packager.verify_release(output) == 3
    with zipfile.ZipFile(output) as archive:
        assert f"{run_dir}/checkpoints/last.pt" not in archive.namelist()
        assert "runs/pilot/test_metrics.json" not in archive.namelist()
        assert archive.read(f"{run_dir}/checkpoints/best.pt") == b"original bytes"
    assert len(packager.evidence_files(tmp_path, ["main"], include_last=True)[0]) == 4
    (tmp_path / produces[0]).unlink()
    with pytest.raises(ValueError, match="Missing required evidence files"):
        packager.evidence_files(tmp_path, ["main"])


def test_config_quality_checks_work_before_training_but_can_require_checkpoints(tmp_path, packager):
    checker = importlib.import_module("check_configs")
    matrix = tmp_path / "evaluation.yaml"
    matrix.write_text(
        "evaluations:\n"
        "  - checkpoint: runs/untrained/checkpoints/best.pt\n"
        "    out_dir: runs/evaluation\n"
        "    symmetry: {name: translation1d, max_shift: 0.1}\n",
        encoding="utf-8",
    )
    assert checker._validate_evaluation_matrix(matrix) == []
    assert any(
        "missing checkpoint" in error
        for error in checker._validate_evaluation_matrix(matrix, require_checkpoints=True)
    )
    matrix.write_text("evaluations:\n  - out_dir: runs/evaluation\n", encoding="utf-8")
    assert any(
        "missing checkpoint" in error for error in checker._validate_evaluation_matrix(matrix)
    )
