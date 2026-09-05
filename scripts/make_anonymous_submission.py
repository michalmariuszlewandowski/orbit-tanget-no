#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

from check_anonymity import check

ALLOWED_FILES = {
    "README.md",
    "LICENSE",
    "Makefile",
    ".gitignore",
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "docs/reproducibility.md",
    "VALIDATION.md",
    "scripts/reproduce_release.py",
    "configs/release.yaml",
    "scripts/check_anonymity.py",
    "scripts/check_configs.py",
    "scripts/clean.py",
    "scripts/quality_gate.py",
    "scripts/smoke_test.py",
    "scripts/validate_solvers.py",
    "scripts/evaluate.py",
    "scripts/generate_data.py",
    "scripts/make_anonymous_submission.py",
    "scripts/make_eta_epsilon_sensitivity_table.py",
    "scripts/make_finite_objective_comparison.py",
    "scripts/make_deeponet_backbone_table.py",
    "scripts/make_cno2d_backbone_table.py",
    "scripts/make_paper_diagnostics.py",
    "scripts/make_paper_tables.py",
    "scripts/make_solver_closure_tables.py",
    "scripts/plot_galilean_n64_paper_figures.py",
    "scripts/plot_navier_stokes_qualitative.py",
    "scripts/reproduce_paper_artifacts.py",
    "scripts/run_matrix.py",
    "scripts/run_ood_severity_matrix.py",
    "scripts/train.py",
    "scripts/verify_rmd17_cached_dataset.py",
    "scripts/verify_paper_results.py",
    "configs/experiments_core.yaml",
    "configs/baselines/1d_burgers_aug.yaml",
    "configs/baselines/1d_burgers_aug_orbit.yaml",
    "configs/baselines/1d_burgers_baseline.yaml",
    "configs/baselines/1d_burgers_canonical.yaml",
    "configs/pilot/1d_advection_translation.yaml",
    "configs/pilot/2d_navier_stokes_galilean.yaml",
    "configs/pilot/2d_navier_stokes_galilean_deeponet.yaml",
    "configs/pilot/2d_navier_stokes_galilean_cno2d.yaml",
    "configs/pilot/2d_navier_stokes_translation_d4.yaml",
    "configs/ablations/2d_galilean_n64_1pct_lambda_0p1.yaml",
    "configs/ablations/2d_galilean_n64_2pct_headline_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_cno2d_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_0p1_ood_severity.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_high_probe.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_robustness.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_0p1_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_oracle_canonicalization.yaml",
    "configs/ablations/2d_galilean_n64_5pct_lambda_0p1.yaml",
    "configs/ablations/2d_galilean_n64_label_efficiency_2pct_ood_severity.yaml",
    "configs/ablations/2d_galilean_n64_label_efficiency_compute_matched.yaml",
    "configs/ablations/2d_galilean_n64_compute_matched_aug.yaml",
    "configs/ablations/2d_galilean_n64_paper_ladder.yaml",
    "configs/ablations/1d_heat_dirichlet_boundary_mask.yaml",
    "configs/ablations/1d_heat_dirichlet_common_mask_evaluation.yaml",
    "configs/ablations/2d_d4_five_seed_completion.yaml",
    "configs/ablations/2d_d4_focused_lambda_refinement.yaml",
    "configs/ablations/2d_d4_focused_low_label_pilot.yaml",
    "configs/ablations/2d_d4_focused_seed_replication.yaml",
    "configs/ablations/2d_d4_gfno_baseline.yaml",
    "configs/ablations/2d_galilean_n64_2pct_eta_epsilon_sensitivity.yaml",
    "configs/ablations/2d_galilean_n64_2pct_plain_finite_baseline.yaml",
    "configs/ablations/2d_galilean_n64_2pct_tangent_baseline.yaml",
    "configs/ablations/1d_final_four_method_seeds.yaml",
    "configs/pilot/1d_heat_dirichlet_nonperiodic.yaml",
    "configs/pilot/1d_burgers_translation_galilean.yaml",
    "configs/pilot/rmd17_ethanol_force_500.yaml",
    "figure_specs/2d_galilean_n64_id_ood_seed23.yaml",
    "figure_specs/paper_results_v11.yaml",
    # The current raw-data regeneration path does not reproduce the historical
    # cached split used by the reviewer run. This small processed tensor is the
    # immutable evidence input and is included by hash below.
    "data/rmd17/rmd17_ethanol_force_500.pt",
}

FORCE_INCLUDE_FILES = {
    "data/rmd17/rmd17_ethanol_force_500.pt",
}

ALLOWED_GLOBS = {
    # Ship the implementation and its regression tests together. The unrelated
    # external KdV replication tests are omitted with that unpublished branch.
    "src/otno/**/*.py",
    "src/otno/*.py",
    "tests/test_*.py",
    "runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.*",
    "runs/figures/2d_galilean_n64_defect_ood_correlation.*",
    "runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.*",
    "runs/figures/2d_galilean_n64_id_ood_fields.*",
    "runs/figures/2d_galilean_n64_fixed_stress_diagnostics.*",
    "runs/figures/2d_galilean_n64_equivariance_residuals.*",
    "runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.*",
    "runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.*",
    "runs/paper_tables/2d_galilean_n64_2pct_lambda_extended.*",
    "runs/paper_tables/2d_galilean_n64_2pct_lambda_robustness.*",
    "runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization.*",
    "runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization_summary.*",
    "runs/paper_tables/2d_galilean_n64_claim_summary.csv",
    "runs/paper_tables/2d_galilean_n64_defect_ood_correlation.*",
    "runs/paper_tables/2d_galilean_n64_final_manifest.csv",
    "runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.*",
    "runs/paper_tables/2d_galilean_n64_id_ood_fields.json",
    "runs/paper_tables/2d_galilean_n64_training_compute.*",
    "runs/paper_tables/1d_heat_dirichlet_common_mask_evaluation.*",
    "runs/paper_tables/1d_burgers_solver_closure.*",
    "runs/paper_tables/1d_final_four_method.*",
    "runs/paper_tables/2d_d4_rotation_with_gfno_5seed.*",
    "runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.*",
    "runs/paper_tables/2d_galilean_n64_2pct_plain_finite_baseline.*",
    "runs/paper_tables/2d_galilean_n64_2pct_tangent_baseline.*",
    "runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.*",
    "runs/paper_tables/2d_galilean_n64_solver_closure*",
}

OMITTED_FILES = {
    "tests/test_lpsda_comparison_table.py",
    "tests/test_lpsda_replication.py",
}

MANIFEST_NAME = "RELEASE_MANIFEST.json"
DATA_MANIFEST_NAME = "DATA_MANIFEST.json"
EVIDENCE_MANIFEST_NAME = "EVIDENCE_MANIFEST.json"
MANIFEST_NAMES = {MANIFEST_NAME, DATA_MANIFEST_NAME, EVIDENCE_MANIFEST_NAME}
REFERENCE_PATTERN = re.compile(
    r"(?<![\w/])(?:configs|scripts)/[A-Za-z0-9_./-]+\.(?:yaml|yml|py|ps1)"
)


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
    if rel_path in OMITTED_FILES:
        return False
    if rel_path in ALLOWED_FILES:
        return True
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in ALLOWED_GLOBS)


def _iter_package_files(root: Path, out_path: Path):
    excludes = _load_excludes(root)
    out_resolved = out_path.resolve()
    # Traverse only the payload roots, never local datasets, virtual environments,
    # or the duplicated manuscript/rebuttal trees.
    candidates = {root / rel for rel in ALLOWED_FILES}
    for pattern in ALLOWED_GLOBS:
        candidates.update(root.glob(pattern))
    for path in sorted(candidates):
        if not path.is_file():
            continue
        if path.resolve() == out_resolved:
            continue
        rel = _norm(path.relative_to(root))
        if not _is_allowed(rel):
            continue
        if _is_excluded(rel, excludes) and rel not in FORCE_INCLUDE_FILES:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Release input must be a regular repository file: {rel}")
        yield path, rel


def release_files(root: Path, out_path: Path) -> list[tuple[Path, str]]:
    """Select the curated payload and close local script/config dependencies."""
    selected = {rel: path for path, rel in _iter_package_files(root, out_path)}
    missing = sorted(ALLOWED_FILES - selected.keys())
    if missing:
        raise ValueError("Missing required release files: " + ", ".join(missing))
    registry_path = root / "configs/release.yaml"
    if registry_path.is_file():
        import yaml

        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        artifacts = set()
        paper_reference = root / "figure_specs/paper_results_v11.yaml"
        if paper_reference.is_file():
            reference = yaml.safe_load(paper_reference.read_text(encoding="utf-8"))
            artifacts.update(table["csv"] for table in reference["tables"])
        for suite in registry.get("suites", {}).values():
            for step in suite.get("reports", []) + suite.get("figures", []):
                artifacts.update(step.get("outputs", []))
                artifacts.update(step.get("sources", []))
                artifacts.update(
                    path
                    for path in step.get("requires", [])
                    if path.startswith(("runs/paper_tables/", "runs/figures/"))
                )
            for step in suite.get("evaluations", []):
                artifacts.update(
                    step["out_prefix"] + suffix for suffix in (".runs.csv", ".aggregate.csv")
                )
        for rel in artifacts:
            path = root / rel
            if not path.is_file():
                raise ValueError(f"Missing declared release artifact: {rel}")
            if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Invalid release artifact path: {rel}")
            selected[rel] = path
    pending = list(selected)
    while pending:
        rel = pending.pop()
        # Test fixtures and the packager itself contain illustrative or omitted
        # filenames. Executable recipes and documentation must resolve.
        if rel.startswith("tests/") or rel == "scripts/make_anonymous_submission.py":
            continue
        path = selected[rel]
        if path.suffix not in {".md", ".py", ".yaml", ".yml", ".ps1"}:
            continue
        content = path.read_text(encoding="utf-8")
        references = set(REFERENCE_PATTERN.findall(content.replace("\\", "/")))
        if rel.startswith("scripts/") and path.suffix == ".py":
            for node in ast.walk(ast.parse(content)):
                names = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                elif isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                for name in names:
                    local_script = f"scripts/{name}.py"
                    if "." not in name and (root / local_script).is_file():
                        references.add(local_script)
        for reference in references:
            if reference in selected:
                continue
            dependency = root / reference
            if not dependency.is_file():
                raise ValueError(f"{rel} references missing release dependency {reference}")
            if not dependency.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Release dependency escapes repository: {reference}")
            selected[reference] = dependency
            pending.append(reference)
    return sorted(((path, rel) for rel, path in selected.items()), key=lambda item: item[1])


def _zip_write(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    # Stable metadata makes identical source bytes produce an identical archive.
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, content)


def build_archive(root: Path, out_path: Path) -> None:
    files = release_files(root, out_path)
    findings = check(root, files)
    if findings:
        print("Refusing to package because anonymity scan failed:")
        for finding in findings:
            print(f"  {finding}")
        raise SystemExit(1)

    expected_hashes = {}
    registry_path = root / "configs/release.yaml"
    if registry_path.is_file():
        import yaml

        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        expected_hashes = {
            dataset["path"]: dataset["paper_sha256"] for dataset in registry["datasets"].values()
        }
    _write_archive(files, out_path, MANIFEST_NAME, expected_hashes)


def build_data_archive(root: Path, out_path: Path) -> None:
    """Bundle the exact evidence inputs declared in the paper release registry."""
    import yaml

    registry = yaml.safe_load((root / "configs/release.yaml").read_text(encoding="utf-8"))
    datasets = registry["datasets"].values()
    files = []
    expected_hashes = {}
    for dataset in datasets:
        rel = dataset["path"]
        path = root / rel
        if not rel.startswith("data/") or not path.resolve().is_relative_to(
            (root / "data").resolve()
        ):
            raise ValueError(f"Dataset path must stay within data/: {rel}")
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing regular dataset file: {rel}")
        if rel in expected_hashes:
            raise ValueError(f"Repeated dataset path: {rel}")
        expected_hashes[rel] = dataset["paper_sha256"]
        files.append((path, rel))
    if not files:
        raise ValueError("The release registry contains no datasets")
    _write_archive(
        sorted(files, key=lambda item: item[1]), out_path, DATA_MANIFEST_NAME, expected_hashes
    )


def evidence_files(
    root: Path, requested_suites: list[str], *, include_last: bool = False
) -> tuple[list[tuple[Path, str]], list[str]]:
    """Select original records/checkpoints for the declared reproduction plan."""
    from reproduce_release import build_plan, load_registry, selected_suites

    registry = load_registry(root)
    suites = selected_suites(registry, requested_suites)
    selected = set()
    for step in build_plan(registry, suites, root):
        if step.stage not in {"train", "evaluate"}:
            continue
        for rel in step.produces:
            if rel.endswith("/checkpoints/last.pt") and not include_last:
                continue
            selected.add(rel)
        if step.protected_dir and step.stage == "evaluate":
            # Adaptation has additional config, RNG, and checkpoint records beyond
            # adapt_results.json. Include only this explicitly selected run.
            selected.update(
                path.relative_to(root).as_posix()
                for path in (root / step.protected_dir).rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl", ".yaml", ".pt"}
            )
    files = []
    missing = []
    for rel in sorted(selected):
        path = root / rel
        if not path.is_file():
            missing.append(rel)
        elif path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid evidence path: {rel}")
        else:
            files.append((path, rel))
    if missing:
        raise ValueError("Missing required evidence files:\n" + "\n".join(missing))
    if not files:
        raise ValueError("No evidence records selected")
    return files, suites


def build_evidence_archive(
    root: Path, out_path: Path, requested_suites: list[str], *, include_last: bool = False
) -> None:
    files, suites = evidence_files(root, requested_suites, include_last=include_last)
    findings = check(root, files)
    if findings:
        raise ValueError("Identifying metadata in evidence records:\n" + "\n".join(findings))
    print(
        f"Evidence suites: {', '.join(suites)}; {sum(path.stat().st_size for path, _ in files)} input bytes",
        flush=True,
    )
    _write_archive(
        files,
        out_path,
        EVIDENCE_MANIFEST_NAME,
        manifest_metadata={"suites": suites, "includes_last_checkpoints": include_last},
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_archive(
    files: list[tuple[Path, str]],
    out_path: Path,
    manifest_name: str,
    expected_hashes: dict[str, str] | None = None,
    manifest_metadata: dict | None = None,
) -> None:
    if any(path.resolve() == out_path.resolve() for path, _ in files):
        raise ValueError("An archive output cannot replace one of its input files")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    records = []
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path, rel in files:
                content = path.read_bytes()
                digest = hashlib.sha256(content).hexdigest()
                if (
                    expected_hashes is not None
                    and rel in expected_hashes
                    and digest != expected_hashes[rel]
                ):
                    raise ValueError(f"Dataset differs from paper SHA-256: {rel}")
                _zip_write(archive, rel, content)
                records.append({"path": rel, "size": len(content), "sha256": digest})
            manifest = {"schema_version": 1, "files": records, **(manifest_metadata or {})}
            _zip_write(
                archive, manifest_name, (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
            )
        verify_release(temporary)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"wrote {out_path} ({len(records)} payload files, {out_path.stat().st_size} bytes)")
    print(f"sha256 {_file_sha256(out_path)}")


def _verify_contents(manifest: dict, read_bytes) -> int:
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported or malformed release manifest")
    seen = set()
    for record in manifest["files"]:
        name = record["path"]
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or name in seen
            or name in MANIFEST_NAMES
        ):
            raise ValueError(f"Invalid or repeated manifest path: {name}")
        seen.add(name)
        content = read_bytes(name)
        if (
            len(content) != record["size"]
            or hashlib.sha256(content).hexdigest() != record["sha256"]
        ):
            raise ValueError(f"Release checksum mismatch: {name}")
    return len(seen)


def verify_release(path: Path) -> int:
    """Verify all recorded bytes in a ZIP or an extracted release directory."""
    if path.is_dir():
        manifests = [path / name for name in sorted(MANIFEST_NAMES) if (path / name).is_file()]
        if not manifests:
            raise ValueError("No release manifest found")
        return sum(
            _verify_contents(
                json.loads(manifest.read_text(encoding="utf-8")),
                lambda name: (path / name).read_bytes(),
            )
            for manifest in manifests
        )
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Release ZIP contains duplicate filenames")
        manifests = set(names) & MANIFEST_NAMES
        if len(manifests) != 1:
            raise ValueError("A release ZIP must contain exactly one release manifest")
        manifest_name = manifests.pop()
        manifest = json.loads(archive.read(manifest_name))
        count = _verify_contents(manifest, archive.read)
        expected = {record["path"] for record in manifest["files"]} | {manifest_name}
        if set(names) != expected:
            raise ValueError("Release ZIP has files absent from its manifest")
        return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an anonymous submission ZIP.")
    parser.add_argument("--root", default=".", help="Repository root.")
    parser.add_argument(
        "--verify", type=Path, help="Verify an existing ZIP or extracted release directory."
    )
    parser.add_argument(
        "--data-out", type=Path, help="Also build a companion ZIP of exact paper datasets."
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        help="Also bundle original selected run records and best checkpoints (large).",
    )
    parser.add_argument(
        "--evidence-suite",
        action="append",
        help="Repeat to select evidence suites and their dependencies; default: all.",
    )
    parser.add_argument(
        "--evidence-last",
        action="store_true",
        help="Include last checkpoints as well as best checkpoints in the evidence ZIP.",
    )
    parser.add_argument(
        "--out",
        default="dist/local_orbit_consistency_anonymous.zip",
        help="Output ZIP path.",
    )
    args = parser.parse_args()

    if args.verify is not None:
        count = verify_release(args.verify)
        print(f"Release checksums verified: {count} files.")
        return

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    outputs = [out_path] + [
        path if path.is_absolute() else root / path
        for path in (args.data_out, args.evidence_out)
        if path is not None
    ]
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("Code, data, and evidence archives must have different output paths")
    build_archive(root, out_path)
    if args.data_out is not None:
        data_out = args.data_out if args.data_out.is_absolute() else root / args.data_out
        build_data_archive(root, data_out)
    if args.evidence_out is not None:
        evidence_out = (
            args.evidence_out if args.evidence_out.is_absolute() else root / args.evidence_out
        )
        build_evidence_archive(
            root, evidence_out, args.evidence_suite or ["all"], include_last=args.evidence_last
        )


if __name__ == "__main__":
    main()
