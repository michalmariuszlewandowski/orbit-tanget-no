#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import re
from collections.abc import Iterable
from pathlib import Path


TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cfg",
    ".cmd",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".rst",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_EXCLUDES = {
    ".git/**",
    ".venv/**",
    ".deps/**",
    ".mypy_cache/**",
    ".pytest_cache/**",
    "__pycache__/**",
    "uv.lock",
    "paper/related_work.bib",
}

# Attribution of an external baseline is required and does not identify the
# authors of this artifact. Keep this exception exact, rather than allowing
# arbitrary code-host links.
PUBLIC_REFERENCE_URLS = {
    "https://github.com/camlab-ethz/ConvolutionalNeuralOperator",
}

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "local user path",
        re.compile(r"(?:[A-Za-z]:)?[/\\]Users[/\\][^/\\\s]+|/home/[^/\s]+", re.IGNORECASE),
    ),
    (
        "email address",
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    ),
    (
        "non-anonymous LaTeX author",
        re.compile(r"\\author\{(?!Anonymous Authors\})[^}]+\}", re.IGNORECASE),
    ),
    (
        "affiliation metadata",
        re.compile(
            r"\b(?:affiliation|affiliations|orcid|acknowledg(?:e|ment|ments)|funded by|grant)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "public code host URL",
        re.compile(
            r"\b(?:https?://)?(?:www\.)?(?:github|gitlab)\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+",
            re.IGNORECASE,
        ),
    ),
]


def _norm(path: Path) -> str:
    return path.as_posix().strip("/")


def _load_submission_excludes(root: Path) -> set[str]:
    path = root / ".submissionignore"
    if not path.exists():
        return set()
    excludes: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        excludes.add(line)
        if line.endswith("/"):
            excludes.add(f"{line}**")
    return excludes


def _is_excluded(rel_path: str, patterns: set[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _iter_text_files(root: Path, excludes: set[str]):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = _norm(path.relative_to(root))
        if rel == "scripts/check_anonymity.py":
            continue
        if _is_excluded(rel, excludes):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES or path.name in {"Makefile"}:
            yield path, rel


def check(root: Path, files: Iterable[tuple[Path, str]] | None = None) -> list[str]:
    """Scan a workspace, or exactly the selected release payload when supplied."""
    excludes = DEFAULT_EXCLUDES | _load_submission_excludes(root)
    findings: list[str] = []
    selected = files if files is not None else _iter_text_files(root, excludes)
    for path, rel in selected:
        if rel == "scripts/check_anonymity.py":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name != "Makefile":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            scanned_line = line
            for public_url in PUBLIC_REFERENCE_URLS:
                scanned_line = re.sub(re.escape(public_url) + r"(?![\w./-])", "", scanned_line)
            for label, pattern in PATTERNS:
                if pattern.search(scanned_line):
                    findings.append(f"{rel}:{line_no}: {label}: {line.strip()}")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan anonymous submission files for identifying metadata."
    )
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Scan the broader workspace instead of the curated release.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if args.all_files:
        findings = check(root)
    else:
        from make_anonymous_submission import release_files

        findings = check(
            root, release_files(root, root / "dist/local_orbit_consistency_anonymous.zip")
        )
    if findings:
        print("Potential anonymity issues:")
        for finding in findings:
            print(f"  {finding}")
        raise SystemExit(1)
    print("Anonymity scan passed.")


if __name__ == "__main__":
    main()
