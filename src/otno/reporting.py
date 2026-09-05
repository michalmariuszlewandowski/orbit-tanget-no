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
                if "steps_per_epoch" in training:
                    row.setdefault("steps_per_epoch", int(training["steps_per_epoch"]))
                if "orbit_data_fraction" in training:
                    row.setdefault("orbit_data_fraction", float(training["orbit_data_fraction"]))
                if "orbit_steps_per_epoch" in training:
                    row.setdefault("orbit_steps_per_epoch", int(training["orbit_steps_per_epoch"]))
                if "lambda_orbit" in training:
                    row.setdefault("lambda_orbit", float(training["lambda_orbit"]))
        rows.append(row)
    return rows


def validate_fresh_run_provenance(df: pd.DataFrame, *, source_root: str | Path) -> None:
    """Validate fresh campaigns from a source archive, including archives without Git."""
    from otno.utils import file_sha256, source_manifest, stable_json_hash

    root = Path(source_root)
    current_source = source_manifest(root)["source_sha256"]
    dataset_hashes: dict[Path, str] = {}
    campaign_dataset = None
    for _, row in df.iterrows():
        run_dir = root / str(row["run_dir"])
        manifest_path = run_dir / "source_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Missing or invalid source manifest: {manifest_path}") from exc
        recorded_source = row.get("source_sha256")
        if (not isinstance(manifest, dict)
                or not isinstance(manifest.get("files"), dict) or not manifest["files"]
                or stable_json_hash(manifest["files"]) != manifest.get("source_sha256")
                or recorded_source != manifest.get("source_sha256")
                or recorded_source != current_source):
            raise ValueError(f"Fresh run source_sha256 does not match the actual release source: {run_dir}")
        if row.get("environment.torch_num_threads") != 1:
            raise ValueError(f"Fresh campaign requires environment.torch_num_threads=1: {run_dir}")
        dataset_path = root / str(row["config.dataset.path"])
        if dataset_path not in dataset_hashes:
            dataset_hashes[dataset_path] = file_sha256(dataset_path)
        actual_dataset = dataset_hashes[dataset_path]
        if row.get("dataset_sha256") != actual_dataset:
            raise ValueError(f"Fresh run dataset_sha256 does not match the actual dataset: {run_dir}")
        if campaign_dataset is not None and actual_dataset != campaign_dataset:
            raise ValueError("Fresh campaign contains inconsistent dataset_sha256 values")
        campaign_dataset = actual_dataset


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
            "oracle_canonical_ood_relative_l2",
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
