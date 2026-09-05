#!/usr/bin/env python
from __future__ import annotations

import argparse
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path


# This external baseline attribution does not identify the project authors.
PUBLIC_REFERENCE_URLS = {
    "https://github.com/camlab-ethz/ConvolutionalNeuralOperator",
}

PATTERNS = [
    (
        "local user path",
        re.compile(r"(?:[A-Za-z]:)?[/\\](?:Users|home)[/\\][^/\\\s]+", re.I),
    ),
    ("email address", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)),
    (
        "LaTeX author",
        re.compile(r"\\author\{(?!Anonymous Authors\})[^}]+\}", re.I),
    ),
    (
        "author or affiliation metadata",
        re.compile(r"^\s*(?:authors?|affiliations?|orcid)\s*[:=]\s*\S+", re.I),
    ),
    (
        "PDF author metadata",
        re.compile(r"/Author\s*\([^)]*[^\s)][^)]*\)", re.I),
    ),
    (
        "public code host URL",
        re.compile(
            r"\b(?:(?:https?://)?(?:www\.)?|git@)(?:github|gitlab)\.com"
            r"[/:][A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            re.I,
        ),
    ),
]
COMMIT_PATTERN = re.compile(r"\b[0-9a-f]{7,40}\b", re.I)


def repository_files(root: Path) -> list[tuple[Path, str]]:
    """Read existing tracked files and untracked files allowed by Git's ignore rules."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    names = sorted(set(result.stdout.decode("utf-8").split("\0")) - {""})
    return [
        (root / name, name)
        for name in names
        if (root / name).is_file() or (root / name).is_symlink()
    ]


def git_identities(root: Path) -> tuple[set[str], set[str]]:
    """Read identity terms and commit IDs without storing them in repository files."""
    history = subprocess.check_output(
        ["git", "-C", str(root), "log", "--all", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce"],
        encoding="utf-8",
    )
    terms, commits = set(), set()
    for record in history.splitlines():
        commit, *identities = record.split("\0")
        commits.add(commit)
        terms.update(identity for identity in identities if identity)
    remotes = subprocess.check_output(
        ["git", "-C", str(root), "remote", "-v"], encoding="utf-8"
    )
    terms.update(re.findall(r"(?:github|gitlab)\.com[/:]([^/:\s]+)/", remotes, re.I))
    return terms, commits


def check(
    root: Path,
    files: Iterable[tuple[Path, str]] | None = None,
    *,
    terms: Iterable[str] = (),
    commits: Iterable[str] = (),
) -> list[str]:
    """Find identifying text and files that need review in the current working tree."""
    findings = []
    selected = repository_files(root) if files is None else files
    terms = [term.casefold() for term in terms if term]
    commit_prefixes = {commit[:length].lower() for commit in commits for length in range(7, 41)}
    for path, rel in selected:
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            findings.append(f"{rel}: symbolic link or path outside the repository")
            continue
        if path.name.startswith(".env") or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx"}:
            findings.append(f"{rel}: possible credential file")
        if path.suffix.lower() in {".zip", ".gz", ".tar", ".7z"}:
            findings.append(f"{rel}: archive contents require review")
        for term in terms:
            if term in rel.casefold():
                findings.append(f"{rel}: identifying search term in filename")
        if any(value.lower() in commit_prefixes for value in COMMIT_PATTERN.findall(rel)):
            findings.append(f"{rel}: repository commit ID in filename")
        try:
            # Also inspect printable metadata in binary files without deserializing them.
            content = path.read_bytes().decode("utf-8", errors="replace")
        except OSError as error:
            findings.append(f"{rel}: could not read file: {error.strerror}")
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            for public_url in PUBLIC_REFERENCE_URLS:
                line = re.sub(re.escape(public_url) + r"(?![\w./-])", "", line)
            for label, pattern in PATTERNS:
                if pattern.search(line):
                    findings.append(f"{rel}:{line_no}: {label}")
            if any(term in line.casefold() for term in terms):
                findings.append(f"{rel}:{line_no}: identifying search term")
            if any(value.lower() in commit_prefixes for value in COMMIT_PATTERN.findall(line)):
                findings.append(f"{rel}:{line_no}: repository commit ID; redact in anonymous mirror")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Git candidate files for identifying text.")
    parser.add_argument("--root", default=".", help="Repository directory to scan.")
    parser.add_argument(
        "--term", action="append", default=[], help="Additional identifying name or term; repeatable."
    )
    parser.add_argument(
        "--git-identities", action="store_true",
        help="Also check local Git author names, emails, remote owners, and commit IDs.",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    try:
        files = repository_files(root)
        terms, commits = git_identities(root) if args.git_identities else (set(), set())
    except (OSError, subprocess.CalledProcessError) as error:
        parser.error(f"Cannot read Git candidate files: {error}")
    findings = check(root, files, terms=terms | set(args.term), commits=commits)
    print(f"Checked {len(files)} existing tracked and nonignored untracked files.")
    print("Scope: text and printable metadata; review binary contents and Git/hosting metadata separately.")
    if findings:
        print("Potential anonymity issues:")
        for finding in findings:
            print(f"  {finding}")
        raise SystemExit(1)
    print("No identifying patterns found in the checked files.")


if __name__ == "__main__":
    main()
