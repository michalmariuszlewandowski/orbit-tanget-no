from __future__ import annotations

import importlib
import pickle
import subprocess
import zipfile
from pathlib import Path

import pytest


@pytest.fixture
def scanner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("check_anonymity")


def test_git_candidates_include_new_and_ignored_tracked_files_but_not_deleted_files(
    tmp_path, scanner
):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    for name in ("tracked.txt", "deleted.txt", "later_ignored.txt"):
        (tmp_path / name).write_text("text", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    (tmp_path / "deleted.txt").unlink()
    (tmp_path / ".gitignore").write_text("*_ignored.txt\n", encoding="utf-8")
    (tmp_path / "new file.txt").write_text("text", encoding="utf-8")
    (tmp_path / "new_ignored.txt").write_text("text", encoding="utf-8")
    names = {rel for _, rel in scanner.repository_files(tmp_path)}
    assert names == {"tracked.txt", "later_ignored.txt", ".gitignore", "new file.txt"}


def test_scans_unknown_extensions_and_printable_binary_metadata(tmp_path, scanner):
    files = []
    for name in ("metadata.custom", "tensor.pt"):
        path = tmp_path / name
        path.write_bytes(b"\x00\xff" + ("/" + "home/" + "researcher/project").encode())
        files.append((path, name))
    findings = scanner.check(tmp_path, files)
    assert len(findings) == 2
    assert all("local user path" in finding for finding in findings)


def test_torch_zip_scans_pickle_metadata_and_member_names_without_reading_tensors(
    tmp_path, scanner, monkeypatch
):
    path = tmp_path / "checkpoint.pt"
    private_path = "/" + "home/" + "researcher/project"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", pickle.dumps({"source": private_path}))
        archive.writestr("checkpoint/version", "3\n")
        archive.writestr("checkpoint/data/0", b"tensor" + private_path.encode())
        archive.writestr("checkpoint/researcher-note", b"opaque storage")
    original_read = zipfile.ZipFile.read
    reads = []

    def read_metadata_only(archive, member, *args, **kwargs):
        name = member.filename if isinstance(member, zipfile.ZipInfo) else member
        assert name.endswith(".pkl"), "scanner attempted to read tensor storage"
        reads.append(name)
        return original_read(archive, member, *args, **kwargs)

    monkeypatch.setattr(scanner.zipfile.ZipFile, "read", read_metadata_only)
    findings = scanner.check(tmp_path, [(path, path.name)], terms=["researcher"])
    assert reads == ["checkpoint/data.pkl"]
    assert any(
        "checkpoint.pt!checkpoint/data.pkl:" in item and "local user path" in item
        for item in findings
    )
    assert any(
        "checkpoint.pt!checkpoint/researcher-note:" in item and "identifying search term" in item
        for item in findings
    )
    assert all("data/0" not in item for item in findings)


def test_reference_exception_does_not_allow_a_different_repository(tmp_path, scanner):
    public_url = "https://github.com/" + "camlab-ethz/ConvolutionalNeuralOperator"
    path = tmp_path / "references.md"
    path.write_text(public_url + "\n" + public_url + "-private\n", encoding="utf-8")
    findings = scanner.check(tmp_path, [(path, "references.md")])
    assert findings == ["references.md:2: public code host URL"]


def test_explicit_identity_terms_match_content_and_filenames(tmp_path, scanner):
    path = tmp_path / "researcher.txt"
    path.write_text("RESEARCHER", encoding="utf-8")
    findings = scanner.check(tmp_path, [(path, path.name)], terms=["researcher"])
    assert len(findings) == 2
    assert "filename" in findings[0]


@pytest.mark.parametrize("name", ["private.pem", "notes.zip", ".env"])
def test_flags_sensitive_or_nested_files(tmp_path, scanner, name):
    path = tmp_path / name
    path.write_bytes(b"contents")
    assert len(scanner.check(tmp_path, [(path, name)])) == 1


def test_fails_outside_git_instead_of_scanning_an_empty_payload(tmp_path, scanner):
    with pytest.raises(subprocess.CalledProcessError):
        scanner.repository_files(tmp_path)


def test_git_metadata_finds_names_remotes_and_full_or_abbreviated_commits(tmp_path, scanner):
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    path = tmp_path / "README.md"
    path.write_text("Researcher Name\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Researcher Name",
            "-c",
            "user.email=researcher" + "@" + "example.org",
            "commit",
            "--quiet",
            "-m",
            "initial",
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "remote",
            "add",
            "origin",
            "https://github.com/" + "research-group/example",
        ],
        check=True,
    )
    terms, commits = scanner.git_identities(tmp_path)
    assert "Researcher Name" in terms
    assert "research-group" in terms
    assert len(commits) == 1
    commit = next(iter(commits))
    path.write_text(f"Researcher Name\n{commit}\n{commit[:12]}\n", encoding="utf-8")
    findings = scanner.check(tmp_path, [(path, path.name)], terms=terms, commits=commits)
    assert len(findings) == 3
    assert "identifying search term" in findings[0]
    assert all("repository commit ID" in finding for finding in findings[1:])
