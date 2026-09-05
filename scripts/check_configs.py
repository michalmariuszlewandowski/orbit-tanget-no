#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from collections import Counter

from otno.config import apply_dotted_overrides, format_config_values, load_config, validate_config
from otno.models import build_model
from otno.symmetry.registry import build_transform


def _validate_single_config(path: Path) -> list[str]:
    errors: list[str] = []
    cfg = load_config(path)
    if "adaptation" in cfg:
        for key in ["adaptation", "runtime"]:
            if key not in cfg:
                errors.append(f"{path}: missing {key}")
        return errors
    for key in ["dataset", "model", "training", "runtime"]:
        if key not in cfg:
            errors.append(f"{path}: missing {key}")
    if errors:
        return errors
    try:
        validate_config(cfg)
        _ = build_model(cfg)
    except Exception as exc:
        errors.append(f"{path}: model build failed: {exc}")
    if cfg.get("symmetry"):
        try:
            _ = build_transform(cfg.get("symmetry"))
        except Exception as exc:
            errors.append(f"{path}: transform build failed: {exc}")
    return errors


def _validate_matrix(path: Path) -> list[str]:
    errors: list[str] = []
    matrix = load_config(path)
    entries = matrix.get("experiments", [])
    if not entries:
        return [f"{path}: no experiments"]
    run_dirs = []
    for idx, entry in enumerate(entries):
        cfg_path = ROOT / entry["config"]
        if not cfg_path.exists():
            errors.append(f"{path} entry {idx}: missing config {entry['config']}")
            continue
        base = load_config(cfg_path)
        merged = apply_dotted_overrides(base, entry.get("overrides", {}))
        run_dir = merged.get("runtime", {}).get("run_dir")
        if not run_dir:
            errors.append(f"{path} entry {idx}: missing runtime.run_dir after overrides")
        else:
            seeds = entry.get("seeds", [None])
            if not isinstance(seeds, list):
                seeds = [seeds]
            for seed in seeds:
                run_dirs.append(f"{str(run_dir).rstrip('/')}/seed_{int(seed)}" if seed is not None else run_dir)
        try:
            validate_config(merged)
            _ = build_model(merged)
            if merged.get("symmetry"):
                _ = build_transform(merged.get("symmetry"))
        except Exception as exc:
            errors.append(f"{path} entry {idx}: merged config invalid: {exc}")
    duplicates = [run for run, count in Counter(run_dirs).items() if count > 1]
    for run in duplicates:
        errors.append(f"{path}: duplicate run_dir {run}")
    return errors


def _validate_adaptation_matrix(path: Path, *, require_checkpoints: bool = False) -> list[str]:
    errors: list[str] = []
    matrix = load_config(path)
    entries = matrix.get("adaptations", [])
    if not entries:
        return [f"{path}: no adaptations"]
    run_dirs = []
    for idx, entry in enumerate(entries):
        cfg_path = ROOT / entry["config"]
        if not cfg_path.exists():
            errors.append(f"{path} entry {idx}: missing config {entry['config']}")
            continue
        seeds = entry.get("seeds", [None])
        if not isinstance(seeds, list):
            seeds = [seeds]
        allow_missing = bool(entry.get("allow_missing_checkpoint", False))
        for seed in seeds:
            context = dict(entry.get("format", {}))
            if seed is not None:
                context["seed"] = int(seed)
            overrides = format_config_values(dict(entry.get("overrides", {})), context)
            merged = apply_dotted_overrides(load_config(cfg_path), overrides)
            for key in ["adaptation", "runtime"]:
                if key not in merged:
                    errors.append(f"{path} entry {idx}: missing {key} after overrides")
            run_dir = merged.get("runtime", {}).get("run_dir")
            if not run_dir:
                errors.append(f"{path} entry {idx}: missing runtime.run_dir after overrides")
            else:
                run_dirs.append(run_dir)
            checkpoint = merged.get("adaptation", {}).get("checkpoint")
            if not checkpoint:
                errors.append(f"{path} entry {idx}: missing adaptation.checkpoint after overrides")
            elif require_checkpoints and not allow_missing and not (ROOT / checkpoint).exists():
                errors.append(f"{path} entry {idx}: missing adaptation checkpoint {checkpoint}")
            trainable = str(merged.get("adaptation", {}).get("trainable", "projector"))
            if trainable not in {"projector", "last_block", "all"}:
                errors.append(f"{path} entry {idx}: unknown adaptation.trainable={trainable!r}")
    duplicates = [run for run, count in Counter(run_dirs).items() if count > 1]
    for run in duplicates:
        errors.append(f"{path}: duplicate run_dir {run}")
    return errors


def _validate_evaluation_matrix(path: Path, *, require_checkpoints: bool = False) -> list[str]:
    errors: list[str] = []
    matrix = load_config(path)
    entries = matrix.get("evaluations", [])
    if not entries:
        return [f"{path}: no evaluations"]
    out_dirs = []
    for idx, entry in enumerate(entries):
        allow_missing = bool(entry.get("allow_missing_checkpoint", False))
        seeds = entry.get("seeds", [None])
        methods = entry.get("methods", [None])
        if not isinstance(seeds, list):
            seeds = [seeds]
        if not isinstance(methods, list):
            methods = [methods]
        for method in methods:
            for seed in seeds:
                context = dict(entry.get("format", {}))
                if method is not None:
                    context["method"] = method
                if seed is not None:
                    context["seed"] = int(seed)
                expanded = format_config_values(dict(entry), context)
                checkpoint = expanded.get("checkpoint")
                out_dir = expanded.get("out_dir")
                if not checkpoint:
                    errors.append(f"{path} entry {idx}: missing checkpoint")
                elif require_checkpoints and not allow_missing and not (ROOT / checkpoint).exists():
                    errors.append(f"{path} entry {idx}: missing checkpoint {checkpoint}")
                if not out_dir:
                    errors.append(f"{path} entry {idx}: missing out_dir")
                else:
                    out_dirs.append(out_dir)
                try:
                    _ = build_transform({"symmetry": expanded.get("symmetry")})
                except Exception as exc:
                    errors.append(f"{path} entry {idx}: transform build failed: {exc}")
    duplicates = [run for run, count in Counter(out_dirs).items() if count > 1]
    for run in duplicates:
        errors.append(f"{path}: duplicate out_dir {run}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment configs and matrix files.")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--require-checkpoints", action="store_true",
        help="Also require existing evaluation/adaptation checkpoints; omit before training.",
    )
    args = parser.parse_args()
    root = Path(args.root)
    errors: list[str] = []
    for path in sorted((root / "configs").glob("**/*.yaml")):
        cfg = load_config(path)
        if "release_version" in cfg:
            from reproduce_release import load_registry

            try:
                load_registry(root, path.relative_to(root).as_posix())
            except (ValueError, KeyError, FileNotFoundError) as exc:
                errors.append(f"{path}: invalid release registry: {exc}")
        elif "experiments" in cfg:
            errors.extend(_validate_matrix(path))
        elif "adaptations" in cfg:
            errors.extend(_validate_adaptation_matrix(path, require_checkpoints=args.require_checkpoints))
        elif "evaluations" in cfg:
            errors.extend(_validate_evaluation_matrix(path, require_checkpoints=args.require_checkpoints))
        else:
            errors.extend(_validate_single_config(path))
    if errors:
        print("CONFIG CHECK FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("CONFIG CHECK PASSED")


if __name__ == "__main__":
    main()
