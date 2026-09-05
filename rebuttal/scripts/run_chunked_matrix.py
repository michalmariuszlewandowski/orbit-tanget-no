#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_matrix import _command, _jobs
from otno.config import apply_dotted_overrides, load_config


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _next_chunk_log(run_dir: Path) -> Path:
    existing = sorted(run_dir.glob("launch_chunk_*.log"))
    return run_dir / f"launch_chunk_{len(existing) + 1:02}.log"


def _summarize(run_dir: Path) -> str:
    test_path = run_dir / "test_metrics.json"
    if test_path.exists():
        metrics = _read_json(test_path)
        return (
            "complete "
            f"relative_l2={metrics.get('relative_l2')} "
            f"orbit_ood={metrics.get('orbit_ood_relative_l2')} "
            f"defect={metrics.get('equivariance_defect_relative')}"
        )
    partial_path = run_dir / "partial_metrics.json"
    if partial_path.exists():
        metrics = _read_json(partial_path)
        return (
            "partial "
            f"completed_epoch={metrics.get('completed_epoch')} "
            f"target_epochs={metrics.get('target_epochs')} "
            f"best_val={metrics.get('best_val_relative_l2')}"
        )
    return "no metrics yet"


def _belt(done: int, total: int, *, width: int = 24) -> str:
    total = max(1, int(total))
    done = min(max(0, int(done)), total)
    filled = round(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def _effective_epochs(config: str, overrides: dict[str, Any]) -> int:
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    merged = apply_dotted_overrides(load_config(config_path), overrides)
    return int(merged.get("training", {}).get("epochs", 0))


def _completed_epochs(run_dir: Path, total_epochs: int) -> int:
    if (run_dir / "test_metrics.json").exists():
        return total_epochs
    partial_path = run_dir / "partial_metrics.json"
    if partial_path.exists():
        metrics = _read_json(partial_path)
        return int(metrics.get("completed_epoch", 0) or 0)
    train_path = run_dir / "train_metrics.jsonl"
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    return 0


def _run_command(cmd: list[str], log_file, *, stream_output: bool) -> int:
    if not stream_output:
        completed = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return int(completed.returncode)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    while True:
        chunk = process.stdout.read(1)
        if chunk:
            log_file.write(chunk)
            log_file.flush()
            sys.stdout.write(chunk)
            sys.stdout.flush()
            continue
        if process.poll() is not None:
            break
    return int(process.wait())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a matrix in resume-friendly training chunks."
    )
    parser.add_argument("--matrix", required=True, help="Experiment matrix YAML file")
    parser.add_argument("--chunk-epochs", type=int, default=20)
    parser.add_argument("--max-chunks-per-job", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Tee each training subprocess to the terminal as well as the chunk log.",
    )
    args = parser.parse_args()
    if args.chunk_epochs <= 0:
        raise SystemExit("--chunk-epochs must be positive")

    jobs = _jobs(Path(args.matrix))
    for job_index, (config, overrides) in enumerate(jobs, start=1):
        run_dir_text = str(overrides.get("runtime.run_dir", "")).strip()
        if not run_dir_text:
            raise SystemExit(f"matrix job missing runtime.run_dir: {config}")
        run_dir = ROOT / run_dir_text
        total_epochs = _effective_epochs(config, overrides)
        print(
            f"matrix {_belt(job_index - 1, len(jobs))} job {job_index}/{len(jobs)}",
            flush=True,
        )
        print(f"job {run_dir_text}", flush=True)

        if (run_dir / "test_metrics.json").exists():
            print(f"  skip: {_summarize(run_dir)}", flush=True)
            continue

        for _ in range(args.max_chunks_per_job):
            chunk_overrides = dict(overrides)
            chunk_overrides["training.stop_after_epochs"] = args.chunk_epochs
            chunk_overrides.setdefault("runtime.overwrite", False)
            if (run_dir / "checkpoints" / "last.pt").exists():
                chunk_overrides["runtime.resume"] = True

            cmd = _command(config, chunk_overrides)
            log_path = _next_chunk_log(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            completed_epochs = _completed_epochs(run_dir, total_epochs)
            print(f"  launch {log_path.relative_to(ROOT)}", flush=True)
            print(f"  progress {_belt(completed_epochs, total_epochs)} epochs", flush=True)
            print("  " + " ".join(shlex.quote(part) for part in cmd), flush=True)
            if args.dry_run:
                break

            with log_path.open("w", encoding="utf-8") as log_file:
                log_file.write(" ".join(shlex.quote(part) for part in cmd) + "\n")
                log_file.flush()
                returncode = _run_command(
                    cmd,
                    log_file,
                    stream_output=args.stream_output,
                )
            if returncode != 0:
                print(f"  failed rc={returncode}: {log_path}", flush=True)
                if not args.continue_on_error:
                    raise SystemExit(returncode)
                break

            print(f"  {_summarize(run_dir)}", flush=True)
            print(f"  progress {_belt(_completed_epochs(run_dir, total_epochs), total_epochs)} epochs", flush=True)
            if (run_dir / "test_metrics.json").exists():
                break
        else:
            message = f"hit --max-chunks-per-job for {run_dir_text}"
            if args.continue_on_error:
                print(f"  {message}", flush=True)
            else:
                raise SystemExit(message)
        print(f"matrix {_belt(job_index, len(jobs))} jobs", flush=True)


if __name__ == "__main__":
    main()
