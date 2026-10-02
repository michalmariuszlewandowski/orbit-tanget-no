#!/usr/bin/env python
"""Preview or remove final-training snapshots from completed release runs."""

from __future__ import annotations

import argparse
import json
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from reproduce_release import ROOT, build_plan, completed, load_registry, selected_suites


def _local_file(root: Path, relative: str) -> Path:
    """Reject traversal, symlinks and Windows junctions before touching a file."""
    root = root.resolve()
    path = root / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Expected a repository-relative path: {relative}")
    if not path.resolve().is_relative_to(root) or path.resolve() == root:
        raise ValueError(f"Path escapes the repository: {relative}")
    for part in [path, *path.parents]:
        if part == root:
            break
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError(f"Refusing a linked cleanup path: {relative}")
    if not path.is_file():
        raise ValueError(f"Not a regular file: {relative}")
    return path


def _record(root: Path, relative: str) -> dict[str, Any]:
    info = _local_file(root, relative).stat()
    return {"path": relative, "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def candidates(root: Path, steps: list) -> tuple[list[dict], list[str]]:
    required = {path for step in steps for path in step.requires + step.produces}
    selected, skipped = [], []
    for step in steps:
        if step.stage != "train" or not step.protected_dir:
            continue
        relative = f"{step.protected_dir}/checkpoints/last.pt"
        if not (root / relative).exists():
            continue
        try:
            if relative in required:
                raise ValueError("checkpoint is a declared release dependency")
            snapshot = _record(root, relative)
            evidence = [_record(root, path) for path in step.produces]
            if not completed(step, root):
                raise ValueError("training evidence is incomplete")
            # Legacy runs can retain an old partial_metrics.json after completion.
            # Check the saved epoch and final results instead of that filename alone.
            payload = torch.load(root / relative, map_location="cpu", weights_only=False, mmap=True)
            epoch = payload.get("epoch")
            target_epoch = step.expected_config["training"]["epochs"]
            metrics = json.loads((root / step.protected_dir / "test_metrics.json").read_text())
            if epoch != target_epoch or payload.get("results") != metrics:
                raise ValueError("snapshot does not contain the completed run's final results")
            del payload
            # Evidence must be stable throughout inspection, including the checkpoint.
            for record in [snapshot, *evidence]:
                if _record(root, record["path"]) != record:
                    raise ValueError("run changed during inspection")
            selected.append({**snapshot, "epoch": epoch, "evidence": evidence})
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            skipped.append(f"{relative}: {exc}")
    return selected, skipped


def remove_candidates(root: Path, selected: list[dict], manifest: Path) -> int:
    """Validate the entire batch and save an audit record before deleting files."""
    for item in selected:
        if not item["path"].startswith("runs/") or not item["path"].endswith(
            "/checkpoints/last.pt"
        ):
            raise ValueError(f"Not a final-training checkpoint: {item['path']}")
        for record in [item, *item["evidence"]]:
            actual = _record(root, record["path"])
            if any(actual[key] != record[key] for key in actual):
                raise ValueError(f"File changed since inspection: {record['path']}")
    manifest = manifest.resolve()
    if not manifest.is_relative_to(root.resolve()) or manifest.suffix != ".json":
        raise ValueError("Cleanup manifest must be a JSON file inside the repository")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "Remove completed release runs' last.pt; preserve best.pt and all evidence.",
        "candidates": selected,
        "removed": [],
        "removed_bytes": 0,
    }
    # Exclusive creation preserves previous cleanup records.
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    try:
        for item in selected:
            for record in [item, *item["evidence"]]:
                actual = _record(root, record["path"])
                if any(actual[key] != record[key] for key in actual):
                    raise ValueError(f"File changed since inspection: {record['path']}")
            _local_file(root, item["path"]).unlink()
            report["removed"].append(item["path"])
            report["removed_bytes"] += item["bytes"]
    finally:
        manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report["removed_bytes"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute", action="store_true", help="Delete the previewed class of files"
    )
    parser.add_argument("--manifest", type=Path, help="New JSON audit file inside the repository")
    args = parser.parse_args()
    registry = load_registry(ROOT)
    steps = build_plan(registry, selected_suites(registry, ["all"]), ROOT)
    selected, skipped = candidates(ROOT, steps)
    for item in selected:
        print(f"{item['bytes'] / 2**20:.1f} MiB  {item['path']}")
    for message in skipped:
        print(f"KEEP {message}")
    size = sum(item["bytes"] for item in selected)
    print(f"{len(selected)} completed-run last.pt files: {size / 2**30:.3f} GiB")
    if not args.execute:
        print("Preview only. Use --execute to remove these final-training snapshots.")
        return
    if not selected:
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = args.manifest or ROOT / "runs/cleanup" / f"checkpoints_{timestamp}.json"
    removed = remove_candidates(ROOT, selected, manifest)
    print(f"Removed {removed / 2**30:.3f} GiB; audit: {manifest}")


if __name__ == "__main__":
    main()
