#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import shlex
import subprocess

import yaml

from otno.config import apply_dotted_overrides, format_config_values, load_config
from otno.training.adaptation import adapt_from_config
from run_matrix import _value_to_cli


def _jobs(matrix_path: Path) -> list[tuple[str, dict]]:
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    jobs: list[tuple[str, dict]] = []
    for entry in matrix.get("adaptations", []):
        seeds = entry.get("seeds", [None])
        if not isinstance(seeds, list):
            seeds = [seeds]
        for seed in seeds:
            context = dict(entry.get("format", {}))
            if seed is not None:
                context["seed"] = int(seed)
            overrides = format_config_values(dict(entry.get("overrides", {})), context)
            if seed is not None:
                overrides["seed"] = int(seed)
            jobs.append((entry["config"], overrides))
    return jobs


def _command(config: str, overrides: dict) -> list[str]:
    cmd = [sys.executable, "scripts/adapt.py", "--config", config]
    for key, value in overrides.items():
        cmd.extend(["--override", f"{key}={_value_to_cli(value)}"])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a YAML adaptation matrix by invoking scripts/adapt.py."
    )
    parser.add_argument("--matrix", required=True, help="Adaptation matrix YAML file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run jobs without spawning scripts/adapt.py subprocesses.",
    )
    args = parser.parse_args()

    failed = False
    for config, overrides in _jobs(Path(args.matrix)):
        cmd = _command(config, overrides)
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        if args.dry_run:
            continue
        run_dir_text = str(overrides.get("runtime.run_dir", "")).strip()
        overwrite = bool(overrides.get("runtime.overwrite", False))
        if run_dir_text and not overwrite and (ROOT / run_dir_text / "adapt_results.json").exists():
            print(f"skip existing adaptation: {run_dir_text}", flush=True)
            continue
        if args.in_process:
            try:
                adapt_from_config(apply_dotted_overrides(load_config(config), overrides))
            except Exception as exc:
                failed = True
                print(f"adaptation matrix job failed: {exc}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
            continue
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            failed = True
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
