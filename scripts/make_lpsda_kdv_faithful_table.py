#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


PUBLISHED_ROWS = [
    ("LPSDA Table 3", "Published KdV 40s FNO(AR)", 64, 0.1248, 0.0108),
    (
        "LPSDA Table 3",
        r"Published KdV 40s FNO(AR)+LPSDA, \(g_1g_2g_3g_4\)",
        64,
        0.0606,
        0.0040,
    ),
]


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


def _rollout_time_steps(config: dict[str, Any]) -> int:
    nt_effective = int(_get(config, "dataset.nt_effective") or 140)
    time_history = int(_get(config, "model.time_history") or 20)
    time_future = int(_get(config, "model.time_future") or 20)
    max_start_time = nt_effective - time_history - time_future
    starts = range(time_history, max_start_time + 1, time_future)
    return max(1, len(list(starts)) * time_future)


def _paper_metric(metrics: dict[str, Any], key: str, rollout_time_steps: int) -> float | None:
    if key not in metrics:
        return None
    if f"{key}_cumulative" in metrics:
        return float(metrics[key])
    return float(metrics[key]) / rollout_time_steps


def _cumulative_metric(metrics: dict[str, Any], key: str) -> float | None:
    cumulative_key = f"{key}_cumulative"
    if cumulative_key in metrics:
        return float(metrics[cumulative_key])
    return _safe_float(metrics, key)


def _safe_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    return float(value)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _tex_pm(value: float | None, sd: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    if sd is None:
        return rf"\({value:.{digits}f}\)"
    return rf"\({value:.{digits}f}\pm{sd:.{digits}f}\)"


def _mean(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if not finite:
        return None
    return float(mean(finite))


def _std(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    if len(finite) < 2:
        return 0.0 if len(finite) == 1 else None
    return float(stdev(finite))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize faithful LPSDA KdV baseline replication runs.")
    parser.add_argument("--root", default="runs/external/lpsda_kdv_faithful/baseline")
    parser.add_argument("--out-prefix", default="runs/paper_tables/lpsda_kdv_faithful_baseline")
    parser.add_argument("--seeds", default="23,31,47,59,71")
    args = parser.parse_args()

    root = Path(args.root)
    out_prefix = Path(args.out_prefix)
    seeds = [int(seed) for seed in args.seeds.split(",") if seed.strip()]

    run_rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for seed in seeds:
        metrics_path = root / f"seed_{seed}" / "test_metrics.json"
        if not metrics_path.exists():
            missing.append(seed)
            continue
        metrics = _load_json(metrics_path)
        config = _load_yaml(metrics_path.parent / "config.yaml")
        rollout_steps = int(metrics.get("rollout_time_steps") or _rollout_time_steps(config))
        run_rows.append(
            {
                "source": "ours independent",
                "method": "FNO (AR), local calibration",
                "seed": seed,
                "samples": metrics.get("num_samples", 64),
                "rollout_time_steps": rollout_steps,
                "trajectory_nmse": _paper_metric(metrics, "trajectory_nmse", rollout_steps),
                "trajectory_nmse_within_seed_std": _paper_metric(
                    metrics, "trajectory_nmse_std", rollout_steps
                ),
                "trajectory_nmse_cumulative": _cumulative_metric(metrics, "trajectory_nmse"),
                "latency_ms_per_sample": _safe_float(metrics, "latency_ms_per_sample"),
                "parameters": metrics.get("parameters"),
                "train_wall_seconds": _safe_float(metrics, "train_wall_seconds"),
                "run_dir": str(metrics_path.parent),
            }
        )

    aggregate_rows: list[dict[str, Any]] = []
    for source, method, samples, nmse, nmse_std in PUBLISHED_ROWS:
        aggregate_rows.append(
            {
                "source": source,
                "method": method,
                "samples": samples,
                "seeds": "",
                "trajectory_nmse": nmse,
                "trajectory_nmse_std": nmse_std,
                "latency_ms_per_sample": None,
                "latency_ms_per_sample_std": None,
                "parameters": "",
            }
        )

    if run_rows:
        aggregate_rows.append(
            {
                "source": "ours independent",
                "method": "FNO (AR), local calibration",
                "samples": int(run_rows[0]["samples"]),
                "seeds": len(run_rows),
                "trajectory_nmse": _mean([row["trajectory_nmse"] for row in run_rows]),
                "trajectory_nmse_std": _std([row["trajectory_nmse"] for row in run_rows]),
                "latency_ms_per_sample": _mean([row["latency_ms_per_sample"] for row in run_rows]),
                "latency_ms_per_sample_std": _std([row["latency_ms_per_sample"] for row in run_rows]),
                "parameters": run_rows[0]["parameters"],
            }
        )

    aggregate_fields = [
        "source",
        "method",
        "samples",
        "seeds",
        "trajectory_nmse",
        "trajectory_nmse_std",
        "latency_ms_per_sample",
        "latency_ms_per_sample_std",
        "parameters",
    ]
    run_fields = [
        "source",
        "method",
        "seed",
        "samples",
        "rollout_time_steps",
        "trajectory_nmse",
        "trajectory_nmse_within_seed_std",
        "trajectory_nmse_cumulative",
        "latency_ms_per_sample",
        "parameters",
        "train_wall_seconds",
        "run_dir",
    ]
    _write_csv(out_prefix.with_suffix(".csv"), aggregate_rows, aggregate_fields)
    _write_csv(out_prefix.with_suffix(".runs.csv"), run_rows, run_fields)

    with out_prefix.with_suffix(".tex").open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Independent LPSDA-aligned KdV 40s FNO(AR) baseline calibration at 64 "
            "training samples. Published rows are transcribed from Brandstetter et al., Table 3, "
            "and are not local reruns. Our row aggregates completed seeds from an independent "
            "implementation aligned to the released protocol, using the same local "
            "Radau data-generation protocol, five Fourier blocks, 32 modes, width 256, and the LPSDA "
            "normalized unrolled MSE averaged over rollout time steps. Published uncertainty is the "
            "reported bootstrap $\\pm2$ SD interval; local uncertainty is sample standard deviation "
            "across training seeds.}\n"
            "\\label{tab:lpsda-kdv-faithful-baseline}\n"
            "\\begin{tabular}{llrrrr}\n"
            "\\toprule\n"
            "Source & Method & Seeds & Samples & ID NMSE & Latency ms/sample \\\\\n"
            "\\midrule\n"
        )
        for row in aggregate_rows:
            f.write(
                f"{row['source']} & {row['method']} & "
                f"{row['seeds'] if row['seeds'] != '' else '--'} & "
                f"{row['samples']} & "
                f"{_tex_pm(row['trajectory_nmse'], row['trajectory_nmse_std'])} & "
                f"{_tex_pm(row['latency_ms_per_sample'], row['latency_ms_per_sample_std'], digits=2)} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"completed seeds: {[row['seed'] for row in run_rows]}")
    print(f"missing seeds: {missing}")
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
