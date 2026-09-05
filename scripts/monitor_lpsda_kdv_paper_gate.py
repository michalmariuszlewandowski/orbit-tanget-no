#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _rollout_time_steps(config: dict[str, Any], fallback: int) -> int:
    nt_effective = int(_get(config, "dataset.nt_effective") or 0)
    time_history = int(_get(config, "model.time_history") or 0)
    time_future = int(_get(config, "model.time_future") or 0)
    if nt_effective <= 0 or time_history <= 0 or time_future <= 0:
        return fallback
    max_start_time = nt_effective - time_history - time_future
    starts = range(time_history, max_start_time + 1, time_future)
    return max(1, len(list(starts)) * time_future)


def _paper_nmse(metrics: dict[str, Any], rollout_steps: int, threshold: float) -> float:
    if "trajectory_nmse_cumulative" in metrics:
        return float(metrics["trajectory_nmse"])
    raw = float(metrics["trajectory_nmse"])
    if raw > threshold and rollout_steps > 0:
        return raw / rollout_steps
    return raw


def _cumulative_nmse(metrics: dict[str, Any]) -> float:
    if "trajectory_nmse_cumulative" in metrics:
        return float(metrics["trajectory_nmse_cumulative"])
    return float(metrics["trajectory_nmse"])


def _write_status(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {message}\n")


def _last_line(path: Path) -> str:
    if not path.exists():
        return ""
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                last = line.strip()
    return last


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate LPSDA KdV seed 23 using paper-normalized NMSE.")
    parser.add_argument("--base", default="runs/external/lpsda_kdv_faithful/baseline")
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--fallback-rollout-time-steps", type=int, default=100)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    seed_dir = base / "seed_23"
    metrics_path = seed_dir / "test_metrics.json"
    config_path = seed_dir / "config.yaml"
    stdout_path = seed_dir / "overnight.stdout.log"
    stderr_path = seed_dir / "overnight.stderr.log"
    status_path = base / "paper_gate_python.status.log"
    remaining_launcher = base / "launch_remaining_seeds.cmd"
    remaining_process_path = base / "remaining_seeds_process.json"

    status_path.write_text(
        f"{datetime.now().isoformat()} python paper gate started "
        f"threshold={args.threshold} poll_seconds={args.poll_seconds}\n",
        encoding="utf-8",
    )

    while True:
        if metrics_path.exists():
            config = _load_yaml(config_path)
            rollout_steps = _rollout_time_steps(config, args.fallback_rollout_time_steps)
            metrics = _load_json(metrics_path)
            paper_nmse = _paper_nmse(metrics, rollout_steps, args.threshold)
            cumulative_nmse = _cumulative_nmse(metrics)
            _write_status(
                status_path,
                "seed_23 completed "
                f"paper_trajectory_nmse={paper_nmse} "
                f"cumulative_trajectory_nmse={cumulative_nmse} "
                f"rollout_time_steps={rollout_steps}",
            )
            if paper_nmse <= args.threshold:
                if remaining_process_path.exists():
                    try:
                        existing = _load_json(remaining_process_path)
                        pid = int(existing.get("pid", 0))
                        subprocess.run(["powershell.exe", "-NoProfile", "-Command", f"Get-Process -Id {pid}"], check=True)
                        _write_status(status_path, "remaining seeds already running; not launching duplicate")
                        return
                    except Exception:
                        pass
                process = subprocess.Popen(["cmd.exe", "/c", str(remaining_launcher)], cwd=str(base))
                remaining_process_path.write_text(
                    json.dumps(
                        {
                            "pid": process.pid,
                            "started": datetime.now().isoformat(),
                            "threshold": args.threshold,
                            "seed_23_paper_trajectory_nmse": paper_nmse,
                            "seed_23_cumulative_trajectory_nmse": cumulative_nmse,
                            "launcher": str(remaining_launcher),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _write_status(status_path, f"remaining seeds launched pid={process.pid}")
                return
            _write_status(status_path, "paper nmse > threshold; not launching remaining seeds")
            return

        stderr_bytes = stderr_path.stat().st_size if stderr_path.exists() else 0
        _write_status(
            status_path,
            f"waiting metrics last_progress='{_last_line(stdout_path)}' stderr_bytes={stderr_bytes}",
        )
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
