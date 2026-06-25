#!/usr/bin/env python
from __future__ import annotations

import argparse
import fnmatch
import zipfile
from pathlib import Path

from check_anonymity import check

ALLOWED_FILES = {
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "src/otno/__init__.py",
    "src/otno/config.py",
    "src/otno/reporting.py",
    "src/otno/utils.py",
    "src/otno/data/__init__.py",
    "src/otno/data/datasets.py",
    "src/otno/data/generators.py",
    "src/otno/data/solvers1d.py",
    "src/otno/data/solvers2d.py",
    "src/otno/models/__init__.py",
    "src/otno/models/canonical.py",
    "src/otno/models/fno.py",
    "src/otno/symmetry/__init__.py",
    "src/otno/symmetry/registry.py",
    "src/otno/symmetry/transforms.py",
    "src/otno/training/__init__.py",
    "src/otno/training/losses.py",
    "src/otno/training/metrics.py",
    "src/otno/training/trainer.py",
    "docs/final_paper_artifacts.md",
    "scripts/check_anonymity.py",
    "scripts/evaluate.py",
    "scripts/generate_data.py",
    "scripts/make_anonymous_submission.py",
    "scripts/make_paper_diagnostics.py",
    "scripts/plot_galilean_n64_paper_figures.py",
    "scripts/reproduce_paper_artifacts.py",
    "scripts/run_chunked_matrix.py",
    "scripts/run_matrix.py",
    "scripts/run_ood_severity_matrix.py",
    "scripts/train.py",
    "configs/pilot/2d_navier_stokes_galilean.yaml",
    "configs/ablations/2d_galilean_n64_1pct_lambda_0p1.yaml",
    "configs/ablations/2d_galilean_n64_2pct_headline_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_0p1_ood_severity.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_high_probe.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_robustness.yaml",
    "configs/ablations/2d_galilean_n64_2pct_oracle_canonicalization.yaml",
    "configs/ablations/2d_galilean_n64_5pct_lambda_0p1.yaml",
    "configs/ablations/2d_galilean_n64_label_efficiency_2pct_ood_severity.yaml",
    "configs/ablations/2d_galilean_n64_label_efficiency_compute_matched.yaml",
    "configs/ablations/2d_galilean_n64_paper_ladder.yaml",
}

ALLOWED_GLOBS = {
    "runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.*",
    "runs/figures/2d_galilean_n64_defect_ood_correlation.*",
    "runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_2pct_lambda_extended.*",
    "runs/paper_tables/2d_galilean_n64_2pct_lambda_robustness.*",
    "runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization.*",
    "runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization_summary.*",
    "runs/paper_tables/2d_galilean_n64_claim_summary.csv",
    "runs/paper_tables/2d_galilean_n64_defect_ood_correlation.*",
    "runs/paper_tables/2d_galilean_n64_final_manifest.csv",
    "runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_training_compute.*",
}


def _norm(path: Path) -> str:
    return path.as_posix().strip("/")


def _load_excludes(root: Path) -> set[str]:
    excludes: set[str] = set()
    ignore_file = root / ".submissionignore"
    if not ignore_file.exists():
        return excludes
    for raw in ignore_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        excludes.add(line)
        if line.endswith("/"):
            excludes.add(f"{line}**")
    return excludes


def _is_excluded(rel_path: str, patterns: set[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _is_allowed(rel_path: str) -> bool:
    if rel_path in ALLOWED_FILES:
        return True
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in ALLOWED_GLOBS)


def _iter_package_files(root: Path, out_path: Path):
    excludes = _load_excludes(root)
    out_resolved = out_path.resolve()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == out_resolved:
            continue
        rel = _norm(path.relative_to(root))
        if not _is_allowed(rel):
            continue
        if _is_excluded(rel, excludes):
            continue
        yield path, rel


def build_archive(root: Path, out_path: Path) -> None:
    findings = check(root)
    if findings:
        print("Refusing to package because anonymity scan failed:")
        for finding in findings:
            print(f"  {finding}")
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, rel in sorted(_iter_package_files(root, out_path), key=lambda item: item[1]):
            archive.write(path, rel)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an anonymous submission ZIP.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--out",
        default="dist/local_orbit_consistency_anonymous.zip",
        help="Output ZIP path.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    build_archive(root, out_path)


if __name__ == "__main__":
    main()
