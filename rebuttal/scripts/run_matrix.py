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


def _value_to_cli(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    text = yaml.safe_dump(value, default_flow_style=True, sort_keys=False).strip()
    if text.endswith("\n..."):
        text = text[:-4].strip()
    return text.replace("...", "").strip() or "null"


def _jobs(matrix_path: Path) -> list[tuple[str, dict]]:
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    jobs: list[tuple[str, dict]] = []
    for entry in matrix.get("experiments", []):
        config = entry["config"]
        base_overrides = dict(entry.get("overrides", {}))
        seeds = entry.get("seeds", [None])
        if not isinstance(seeds, list):
            seeds = [seeds]
        for seed in seeds:
            overrides = dict(base_overrides)
            if seed is not None:
                overrides["seed"] = int(seed)
                run_dir = str(overrides.get("runtime.run_dir", "")).rstrip("/")
                if run_dir:
                    overrides["runtime.run_dir"] = f"{run_dir}/seed_{int(seed)}"
            jobs.append((config, overrides))
    return jobs


def _command(config: str, overrides: dict) -> list[str]:
    cmd = [sys.executable, "scripts/train.py", "--config", config]
    for key, value in overrides.items():
        cmd.extend(["--override", f"{key}={_value_to_cli(value)}"])
    return cmd


def _commands(matrix_path: Path) -> list[list[str]]:
    commands: list[list[str]] = []
    for config, overrides in _jobs(matrix_path):
        commands.append(_command(config, overrides))
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a YAML experiment matrix by invoking scripts/train.py."
    )
    parser.add_argument(
        "--matrix",
        "--config",
        dest="matrix",
        required=True,
        help="Experiment matrix YAML file",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help=(
            "Skip jobs whose resolved runtime.run_dir already contains "
            "test_metrics.json; partial protected runs still fail safely."
        ),
    )
    parser.add_argument(
        "--seed",
        dest="seeds",
        action="append",
        type=int,
        help="Run only this seed; repeat the flag to select multiple seeds.",
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        help="Run only this training.method; repeat the flag to select multiple methods.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run jobs without spawning scripts/train.py subprocesses.",
    )
    args = parser.parse_args()
    jobs = _jobs(Path(args.matrix))
    if args.seeds:
        selected_seeds = set(args.seeds)
        jobs = [job for job in jobs if int(job[1].get("seed", -1)) in selected_seeds]
    if args.methods:
        selected_methods = {method.lower() for method in args.methods}
        jobs = [
            job
            for job in jobs
            if str(job[1].get("training.method", "")).lower() in selected_methods
        ]
    if not jobs:
        raise SystemExit("No matrix jobs matched the requested filters")
    train_from_config = None
    load_config = None
    apply_dotted_overrides = None
    if args.in_process and not args.dry_run:
        from otno.config import apply_dotted_overrides, load_config
        from otno.training.trainer import train_from_config

    for config, overrides in jobs:
        if args.skip_completed:
            run_dir_value = overrides.get("runtime.run_dir")
            run_dir = Path(str(run_dir_value)) if run_dir_value else None
            if run_dir is not None and (run_dir / "test_metrics.json").is_file():
                print(f"skipping completed matrix job: {run_dir}", flush=True)
                continue
        cmd = _command(config, overrides)
        print(" ".join(shlex.quote(part) for part in cmd), flush=True)
        if args.dry_run:
            continue
        if args.in_process:
            assert train_from_config is not None
            assert load_config is not None
            assert apply_dotted_overrides is not None
            try:
                train_from_config(apply_dotted_overrides(load_config(config), overrides))
            except Exception as exc:
                print(f"matrix job failed: {exc}", file=sys.stderr, flush=True)
                if not args.continue_on_error:
                    raise
            continue
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0 and not args.continue_on_error:
            raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
