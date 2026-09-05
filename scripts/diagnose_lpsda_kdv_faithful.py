#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import yaml


PUBLISHED_FNO_NO_NMSE = 0.1699
PUBLISHED_FNO_NO_STD = 0.0100
DEFAULT_GATE = 0.20

EXPECTED = {
    "dataset.train_samples": 64,
    "dataset.valid_samples": 512,
    "dataset.test_samples": 512,
    "dataset.nx": 256,
    "dataset.nt": 250,
    "dataset.nt_effective": 140,
    "dataset.end_time": 100.0,
    "dataset.length": 128.0,
    "dataset.tol": 1.0e-9,
    "model.modes": 32,
    "model.width": 256,
    "model.num_layers": 5,
    "model.time_history": 20,
    "model.time_future": 20,
    "training.method": "baseline",
    "training.batch_size": 16,
    "training.epochs": 20,
    "training.lr": 1.0e-4,
    "training.lr_decay": 0.4,
    "training.cycles_multiplier": 2,
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) <= max(1.0e-12, abs(expected) * 1.0e-9)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _rollout_time_steps(config: dict[str, Any] | None) -> int:
    if config is None:
        return 100
    nt_effective = int(_get(config, "dataset.nt_effective") or 140)
    time_history = int(_get(config, "model.time_history") or 20)
    time_future = int(_get(config, "model.time_future") or 20)
    max_start_time = nt_effective - time_history - time_future
    starts = range(time_history, max_start_time + 1, time_future)
    return max(1, len(list(starts)) * time_future)


def _paper_metric(record: dict[str, Any], key: str, rollout_time_steps: int) -> float:
    cumulative_key = f"{key}_cumulative"
    if cumulative_key in record:
        return float(record[key])
    return float(record[key]) / rollout_time_steps


def _cumulative_metric(record: dict[str, Any], key: str, rollout_time_steps: int) -> float:
    cumulative_key = f"{key}_cumulative"
    if cumulative_key in record:
        return float(record[cumulative_key])
    return float(record[key])


def _progress(stdout_path: Path) -> dict[str, int | str]:
    if not stdout_path.exists():
        return {"train": 0, "valid": 0, "test": 0, "last": ""}
    lines = stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
    out: dict[str, int | str] = {"train": 0, "valid": 0, "test": 0, "last": lines[-1] if lines else ""}
    for split in ("train", "valid", "test"):
        pattern = re.compile(rf"^generated {split} (\d+)/")
        values = [int(match.group(1)) for line in lines if (match := pattern.match(line))]
        out[split] = max(values) if values else 0
    return out


def _append_config_checks(lines: list[str], config: dict[str, Any] | None) -> list[str]:
    mismatches: list[str] = []
    lines.append("## Protocol Checks")
    if config is None:
        lines.append("- Missing `config.yaml` in the run directory.")
        return ["missing config.yaml"]
    for key, expected in EXPECTED.items():
        actual = _get(config, key)
        ok = _same(expected, actual)
        marker = "OK" if ok else "MISMATCH"
        lines.append(f"- `{key}`: {marker} (expected `{expected}`, observed `{actual}`)")
        if not ok:
            mismatches.append(key)
    return mismatches


def _append_training_curve(
    lines: list[str], rows: list[dict[str, Any]], rollout_time_steps: int
) -> None:
    lines.append("## Validation Curve")
    if not rows:
        lines.append("- `train_metrics.jsonl` is missing or empty; no validation trend is available.")
        return

    best = min(rows, key=lambda row: _paper_metric(row, "unrolled_mse", rollout_time_steps))
    last = rows[-1]
    lines.append(f"- Epoch rows observed: `{len(rows)}`")
    lines.append(f"- Rollout time steps used for paper normalization: `{rollout_time_steps}`")
    lines.append(
        "- Best validation unrolled MSE: "
        f"`{_paper_metric(best, 'unrolled_mse', rollout_time_steps):.6g}` at epoch `{best.get('epoch')}`"
    )
    lines.append(
        "- Best validation trajectory NMSE: "
        f"`{_paper_metric(best, 'trajectory_nmse', rollout_time_steps):.6g}` at epoch `{best.get('epoch')}`"
    )
    lines.append(
        "- Last validation trajectory NMSE: "
        f"`{_paper_metric(last, 'trajectory_nmse', rollout_time_steps):.6g}` at epoch `{last.get('epoch')}`"
    )
    lines.append(
        "- Last validation cumulative trajectory NMSE: "
        f"`{_cumulative_metric(last, 'trajectory_nmse', rollout_time_steps):.6g}` at epoch `{last.get('epoch')}`"
    )
    if len(rows) >= 2:
        prev = rows[-2]
        delta = _paper_metric(last, "trajectory_nmse", rollout_time_steps) - _paper_metric(
            prev, "trajectory_nmse", rollout_time_steps
        )
        lines.append(f"- Last-epoch trajectory-NMSE change: `{delta:+.6g}`")
    recent = rows[-5:]
    formatted = [
        f"{row.get('epoch')}:{_paper_metric(row, 'trajectory_nmse', rollout_time_steps):.4g}"
        for row in recent
    ]
    lines.append(f"- Recent trajectory NMSE values: `{', '.join(formatted)}`")


def diagnose(run_dir: Path, out_path: Path, threshold: float) -> None:
    config = _load_yaml(run_dir / "config.yaml")
    meta = _load_json(run_dir / "meta.json")
    metrics = _load_json(run_dir / "test_metrics.json")
    train_rows = _load_jsonl(run_dir / "train_metrics.jsonl")
    rollout_steps = _rollout_time_steps(config)
    progress = _progress(run_dir / "overnight.stdout.log")
    stderr_path = run_dir / "overnight.stderr.log"
    stderr_bytes = stderr_path.stat().st_size if stderr_path.exists() else 0

    lines: list[str] = [
        "# Faithful LPSDA KdV Seed-23 Diagnosis",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Published FNO(NO) reference: `{PUBLISHED_FNO_NO_NMSE:.4f} +/- {PUBLISHED_FNO_NO_STD:.4f}`",
        f"- Gate threshold used by monitor: `{threshold:.4f}`",
        f"- Generation progress: train `{progress['train']}/64`, valid `{progress['valid']}/512`, test `{progress['test']}/512`",
        f"- Last progress line: `{progress['last']}`",
        f"- Stderr bytes: `{stderr_bytes}`",
        "",
    ]

    mismatches = _append_config_checks(lines, config)
    lines.append("")
    lines.append("## Metrics")
    if metrics is None:
        lines.append("- `test_metrics.json` is not present yet; the run has not reached the comparison gate.")
    else:
        nmse = _paper_metric(metrics, "trajectory_nmse", rollout_steps)
        cumulative_nmse = _cumulative_metric(metrics, "trajectory_nmse", rollout_steps)
        delta = nmse - PUBLISHED_FNO_NO_NMSE
        z_like = delta / PUBLISHED_FNO_NO_STD
        decision = "PASS" if nmse <= threshold else "FAIL"
        lines.append(f"- `trajectory_nmse`: `{nmse:.6g}`")
        lines.append(f"- `trajectory_nmse_cumulative`: `{cumulative_nmse:.6g}`")
        lines.append(f"- Rollout time steps used for paper normalization: `{rollout_steps}`")
        lines.append(f"- Difference vs published FNO(NO): `{delta:+.6g}` ({z_like:+.2f} published standard deviations)")
        lines.append(f"- Gate decision at threshold `{threshold:.4f}`: `{decision}`")
        for key in ("best_val_unrolled_mse", "parameters", "latency_ms_per_sample", "train_wall_seconds"):
            if key in metrics:
                lines.append(f"- `{key}`: `{metrics[key]}`")

    lines.append("")
    _append_training_curve(lines, train_rows, rollout_steps)
    lines.append("")
    lines.append("## Artifact Checks")
    if meta is None:
        lines.append("- Missing `meta.json`; training has probably not started yet.")
    else:
        for key in ("dataset_sha256", "config_hash", "parameters", "device"):
            if key in meta:
                lines.append(f"- `{key}`: `{meta[key]}`")
    for name in ("config.yaml", "meta.json", "best_model.pt", "train_metrics.jsonl", "test_metrics.json"):
        path = run_dir / name
        state = "present" if path.exists() else "missing"
        lines.append(f"- `{name}`: {state}")

    lines.append("")
    lines.append("## Initial Interpretation")
    if metrics is None:
        lines.append("- The run is still pending; no result can be judged yet.")
    elif mismatches:
        lines.append("- The metric should not be interpreted as a faithful reproduction until the protocol mismatches above are resolved.")
    elif _paper_metric(metrics, "trajectory_nmse", rollout_steps) <= threshold:
        lines.append("- The seed is close enough to the published FNO(NO) reference for the planned multi-seed average.")
    else:
        lines.append("- The protocol checks pass but the metric misses the gate; next inspect training curves, rollout/evaluation normalization, and FNO implementation parity.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a faithful LPSDA KdV replication run.")
    parser.add_argument("--run-dir", default="runs/external/lpsda_kdv_faithful/baseline/seed_23")
    parser.add_argument("--out", default="runs/external/lpsda_kdv_faithful/baseline/seed_23/diagnosis.md")
    parser.add_argument("--threshold", type=float, default=DEFAULT_GATE)
    args = parser.parse_args()
    diagnose(Path(args.run_dir), Path(args.out), args.threshold)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
