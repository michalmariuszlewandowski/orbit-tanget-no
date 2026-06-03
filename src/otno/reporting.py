from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def collect_run_rows(runs_dir: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(runs_dir).glob("**/test_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        row: dict[str, Any] = {"run_dir": str(metrics_path.parent)}
        _flatten("", payload, row)
        config_path = metrics_path.parent / "config.yaml"
        if config_path.exists():
            row["config_path"] = str(config_path)
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            config_flat: dict[str, Any] = {}
            _flatten("config", config, config_flat)
            row.update(config_flat)
            training = config.get("training", {}) if isinstance(config, dict) else {}
            if isinstance(training, dict):
                row.setdefault("data_fraction", float(training.get("data_fraction", 1.0)))
                if "orbit_data_fraction" in training:
                    row.setdefault("orbit_data_fraction", float(training["orbit_data_fraction"]))
                if "orbit_steps_per_epoch" in training:
                    row.setdefault("orbit_steps_per_epoch", int(training["orbit_steps_per_epoch"]))
                if "lambda_orbit" in training:
                    row.setdefault("lambda_orbit", float(training["lambda_orbit"]))
        rows.append(row)
    return rows


def aggregate_runs(df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    group_cols = group_cols or [
        col
        for col in ["method", "model_name", "dataset_kind", "data_fraction"]
        if col in df.columns
    ]
    metric_cols = [
        col
        for col in [
            "relative_l2",
            "orbit_ood_relative_l2",
            "equivariance_defect_relative",
            "latency_ms_per_sample",
            "latency_ms_per_batch",
        ]
        if col in df.columns
    ]
    if not group_cols or not metric_cols:
        return df
    agg = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    return agg
