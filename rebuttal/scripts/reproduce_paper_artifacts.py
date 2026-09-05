#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import yaml

from otno.reporting import collect_run_rows


METRIC_COLS = [
    "relative_l2",
    "orbit_ood_relative_l2",
    "oracle_canonical_ood_relative_l2",
    "equivariance_defect_relative",
    "epsilon_mean",
    "latency_ms_per_sample",
    "latency_ms_per_batch",
    "parameters",
]

REQUIRED_TRAINING_FILES = [
    "config.yaml",
    "meta.json",
    "rng_state_initial.pt",
    "rng_state_final.pt",
    "train_metrics.jsonl",
    "val_metrics.jsonl",
    "test_metrics.json",
    "checkpoints/best.pt",
    "checkpoints/last.pt",
]

TRAINING_SPECS = {
    "headline": [
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_headline_5seed/"
            "fraction_0.02/baseline_steps_4",
            "method": "baseline",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_headline_5seed/"
            "fraction_0.02/orbit_lambda_0.05_steps_4",
            "method": "orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.05,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_label_efficiency_compute_matched/"
            "fraction_0.02/aug_steps_6",
            "method": "aug",
            "data_fraction": 0.02,
            "steps_per_epoch": 6,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.1_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.10,
        },
    ],
    "label_efficiency": [
        {
            "prefix": "runs/ablations/2d_galilean_n64_label_efficiency_compute_matched/"
            "fraction_0.01/aug_steps_6",
            "method": "aug",
            "data_fraction": 0.01,
            "steps_per_epoch": 6,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_1pct_lambda_0p1/"
            "fraction_0.01/aug_orbit_lambda_0.1_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.01,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.10,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_label_efficiency_compute_matched/"
            "fraction_0.02/aug_steps_6",
            "method": "aug",
            "data_fraction": 0.02,
            "steps_per_epoch": 6,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.1_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.10,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_compute_matched/"
            "fraction_0.05/aug_steps_6",
            "method": "aug",
            "data_fraction": 0.05,
            "steps_per_epoch": 6,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_5pct_lambda_0p1/"
            "fraction_0.05/aug_orbit_lambda_0.1_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.05,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.10,
        },
    ],
    "lambda_robustness": [
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.01_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.01,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_paper_ladder/"
            "fraction_0.02/aug_orbit_lambda_0.05_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.05,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.1_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.10,
        },
    ],
    "lambda_extended": [
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.01_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.01,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_paper_ladder/"
            "fraction_0.02/aug_orbit_lambda_0.05_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.05,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.1_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.10,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_high_probe/"
            "fraction_0.02/aug_orbit_lambda_0.2_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.20,
        },
        {
            "prefix": "runs/ablations/2d_galilean_n64_2pct_lambda_high_probe/"
            "fraction_0.02/aug_orbit_lambda_0.3_steps_4",
            "method": "aug_orbit",
            "data_fraction": 0.02,
            "steps_per_epoch": 4,
            "lambda_orbit": 0.30,
        },
    ],
}

TABLE_OUTPUTS = {
    "headline": {
        "out_prefix": "2d_galilean_n64_2pct_headline_lambda_0p1",
        "group_cols": ["method", "data_fraction", "steps_per_epoch"],
    },
    "label_efficiency": {
        "out_prefix": "2d_galilean_n64_label_efficiency_lambda_0p1",
        "group_cols": ["method", "data_fraction", "steps_per_epoch"],
    },
    "lambda_robustness": {
        "out_prefix": "2d_galilean_n64_2pct_lambda_robustness",
        "group_cols": ["method", "data_fraction", "steps_per_epoch", "lambda_orbit"],
    },
    "lambda_extended": {
        "out_prefix": "2d_galilean_n64_2pct_lambda_extended",
        "group_cols": ["method", "data_fraction", "steps_per_epoch", "lambda_orbit"],
    },
}

SEVERITY_MATRICES = [
    {
        "path": "configs/ablations/2d_galilean_n64_label_efficiency_2pct_ood_severity.yaml",
        "methods": {"aug_steps_6"},
    },
    {
        "path": "configs/ablations/2d_galilean_n64_2pct_lambda_0p1_ood_severity.yaml",
        "methods": {"aug_orbit_lambda_0.1_steps_4"},
    },
]

ORACLE_MATRIX = "configs/ablations/2d_galilean_n64_2pct_oracle_canonicalization.yaml"


def _norm_path(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip("/")


def _repo_relative_path(value: str | Path) -> str:
    path = Path(value)
    if path.is_absolute():
        try:
            return _norm_path(path.resolve(strict=False).relative_to(ROOT.resolve(strict=False)))
        except ValueError:
            return _norm_path(path)
    return _norm_path(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in df.columns
    ]
    return df


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    value_cols = [col for col in METRIC_COLS if col in df.columns]
    if df.empty or not value_cols:
        return pd.DataFrame()
    aggregate = (
        df.groupby(group_cols, dropna=False)[value_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    return _flatten_columns(aggregate)


def _write_table(df: pd.DataFrame, out_prefix: Path, group_cols: list[str]) -> pd.DataFrame:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    aggregate = _aggregate(df, group_cols)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        if aggregate.empty:
            f.write("% No completed runs found.\n")
        else:
            f.write(aggregate.to_latex(index=False, escape=False))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}", flush=True)
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}", flush=True)
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}", flush=True)
    return aggregate


def _select_training_rows(all_runs: pd.DataFrame, table_name: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    all_runs = all_runs.copy()
    all_runs["run_dir_norm"] = all_runs["run_dir"].map(_norm_path)
    for spec_index, spec in enumerate(TRAINING_SPECS[table_name]):
        prefix = _norm_path(str(spec["prefix"]))
        rows = all_runs[all_runs["run_dir_norm"].str.startswith(prefix)].copy()
        if rows.empty:
            raise SystemExit(f"No completed runs matched prefix: {prefix}")
        for key, value in spec.items():
            if key != "prefix":
                rows[key] = value
        rows["paper_table"] = table_name
        rows["spec_index"] = spec_index
        frames.append(rows)
    out = pd.concat(frames, ignore_index=True)
    sort_cols = [col for col in ["spec_index", "seed", "run_dir"] if col in out.columns]
    return out.sort_values(sort_cols).drop(columns=["run_dir_norm", "spec_index"])


def _format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, context) for key, item in value.items()}
    return value


def _symmetry_value(symmetry: dict[str, Any], key: str) -> Any:
    if key in symmetry:
        return symmetry[key]
    for transform in symmetry.get("transforms", []):
        if isinstance(transform, dict) and key in transform:
            return transform[key]
    return None


def _iter_severity_jobs(matrix_path: Path) -> list[dict[str, Any]]:
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    jobs: list[dict[str, Any]] = []
    for entry in matrix.get("evaluations", []):
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
                job = _format_value(dict(entry), context)
                job.pop("methods", None)
                job.pop("seeds", None)
                job.pop("format", None)
                if method is not None:
                    job["method"] = method
                if seed is not None:
                    job["seed"] = int(seed)
                jobs.append(job)
    return jobs


def _collect_severity_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in SEVERITY_MATRICES:
        allowed_methods = spec["methods"]
        for job in _iter_severity_jobs(ROOT / str(spec["path"])):
            if job.get("method") not in allowed_methods:
                continue
            out_dir = ROOT / str(job["out_dir"])
            metrics_path = out_dir / "severity_metrics.json"
            if not metrics_path.exists():
                raise SystemExit(f"Missing severity metrics: {metrics_path}")
            metrics = _read_json(metrics_path)
            symmetry = job.get("symmetry", {})
            rows.append(
                {
                    "checkpoint": _norm_path(job["checkpoint"]),
                    "run_dir": _norm_path(out_dir.relative_to(ROOT)),
                    "method": job.get("method"),
                    "seed": job.get("seed"),
                    "severity": job.get("severity"),
                    "severity_scale": job.get("severity_scale"),
                    "max_shift": _symmetry_value(symmetry, "max_shift"),
                    "max_boost": _symmetry_value(symmetry, "max_boost"),
                    "paper_table": "ood_severity",
                    **metrics,
                }
            )
    df = pd.DataFrame(rows)
    return df.sort_values(["severity_scale", "method", "seed"])


def _collect_oracle_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for job in _iter_severity_jobs(ROOT / ORACLE_MATRIX):
        out_dir = ROOT / str(job["out_dir"])
        metrics_path = out_dir / "severity_metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Missing oracle canonicalization metrics: {metrics_path}")
        metrics = _read_json(metrics_path)
        symmetry = job.get("symmetry", {})
        rows.append(
            {
                "checkpoint": _norm_path(job["checkpoint"]),
                "run_dir": _norm_path(out_dir.relative_to(ROOT)),
                "method": job.get("method"),
                "seed": job.get("seed"),
                "severity": job.get("severity"),
                "severity_scale": job.get("severity_scale"),
                "max_shift": _symmetry_value(symmetry, "max_shift"),
                "max_boost": _symmetry_value(symmetry, "max_boost"),
                "paper_table": "oracle_canonicalization",
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values(["severity_scale", "method", "seed"])


def _manifest_for_training(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in df.to_dict(orient="records"):
        run_dir_text = _norm_path(record["run_dir"])
        if run_dir_text in seen:
            continue
        seen.add(run_dir_text)
        run_dir = ROOT / run_dir_text
        meta_path = run_dir / "meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        missing = [
            relative for relative in REQUIRED_TRAINING_FILES if not (run_dir / relative).exists()
        ]
        rows.append(
            {
                "artifact_type": "training_run",
                "paper_table": record.get("paper_table"),
                "method": record.get("method"),
                "data_fraction": record.get("data_fraction"),
                "lambda_orbit": record.get("lambda_orbit"),
                "seed": record.get("seed"),
                "severity": None,
                "run_dir": run_dir_text,
                "checkpoint": _repo_relative_path(run_dir / "checkpoints/best.pt"),
                "config_hash": meta.get("config_hash", record.get("config_hash")),
                "dataset_sha256": meta.get("dataset_sha256", record.get("dataset_sha256")),
                "parameters": meta.get("parameters", record.get("parameters")),
                "latency_ms_per_sample": record.get("latency_ms_per_sample"),
                "complete": not missing,
                "missing_files": ";".join(missing),
            }
        )
    return rows


def _manifest_for_severity(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        run_dir_text = _norm_path(record["run_dir"])
        run_dir = ROOT / run_dir_text
        checkpoint = ROOT / str(record["checkpoint"])
        missing = []
        if not (run_dir / "severity_metrics.json").exists():
            missing.append("severity_metrics.json")
        if not checkpoint.exists():
            missing.append(str(record["checkpoint"]))
        rows.append(
            {
                "artifact_type": "severity_eval",
                "paper_table": record.get("paper_table", "ood_severity"),
                "method": record.get("method"),
                "data_fraction": 0.02,
                "lambda_orbit": 0.10 if record.get("method") == "aug_orbit_lambda_0.1_steps_4" else None,
                "seed": record.get("seed"),
                "severity": record.get("severity"),
                "run_dir": run_dir_text,
                "checkpoint": record["checkpoint"],
                "config_hash": None,
                "dataset_sha256": None,
                "parameters": None,
                "latency_ms_per_sample": None,
                "complete": not missing,
                "missing_files": ";".join(missing),
            }
        )
    return rows


def _metric_mean(df: pd.DataFrame, metric: str, **keys: Any) -> float:
    rows = df.copy()
    for key, value in keys.items():
        rows = rows[rows[key] == value]
    if len(rows) != 1:
        raise SystemExit(f"Expected one row for {keys}, found {len(rows)}")
    return float(rows.iloc[0][f"{metric}_mean"])


def _add_reduction(
    rows: list[dict[str, Any]],
    *,
    setting: str,
    comparison: str,
    metric: str,
    reference: float,
    candidate: float,
    n: int,
) -> None:
    rows.append(
        {
            "setting": setting,
            "comparison": comparison,
            "metric": metric,
            "reference_mean": reference,
            "candidate_mean": candidate,
            "relative_reduction_pct": 100.0 * (reference - candidate) / reference,
            "seed_count": n,
        }
    )


def _claim_summary(
    *,
    headline: pd.DataFrame,
    label_efficiency: pd.DataFrame,
    severity: pd.DataFrame,
    out_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for metric in ["orbit_ood_relative_l2", "equivariance_defect_relative"]:
        _add_reduction(
            rows,
            setting="2pct_headline",
            comparison="aug_orbit_lambda_0.10_vs_aug_compute_matched",
            metric=metric,
            reference=_metric_mean(headline, metric, method="aug"),
            candidate=_metric_mean(headline, metric, method="aug_orbit"),
            n=5,
        )
        _add_reduction(
            rows,
            setting="2pct_headline",
            comparison="aug_orbit_lambda_0.10_vs_supervised_baseline",
            metric=metric,
            reference=_metric_mean(headline, metric, method="baseline"),
            candidate=_metric_mean(headline, metric, method="aug_orbit"),
            n=5,
        )
    for fraction, n in [(0.01, 3), (0.02, 5), (0.05, 3)]:
        for metric in ["orbit_ood_relative_l2", "equivariance_defect_relative"]:
            _add_reduction(
                rows,
                setting=f"label_fraction_{fraction:g}",
                comparison="aug_orbit_lambda_0.10_vs_aug_compute_matched",
                metric=metric,
                reference=_metric_mean(
                    label_efficiency, metric, method="aug", data_fraction=fraction
                ),
                candidate=_metric_mean(
                    label_efficiency, metric, method="aug_orbit", data_fraction=fraction
                ),
                n=n,
            )
    for severity_name in ["boost_0.10", "boost_0.20", "train_radius", "boost_0.35", "boost_0.50"]:
        for metric in ["orbit_ood_relative_l2", "equivariance_defect_relative"]:
            _add_reduction(
                rows,
                setting=f"severity_{severity_name}",
                comparison="aug_orbit_lambda_0.10_vs_aug_compute_matched",
                metric=metric,
                reference=_metric_mean(
                    severity, metric, method="aug_steps_6", severity=severity_name
                ),
                candidate=_metric_mean(
                    severity,
                    metric,
                    method="aug_orbit_lambda_0.1_steps_4",
                    severity=severity_name,
                ),
                n=5,
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}", flush=True)


def _regenerate_figures(*, table_dir: Path, figures_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "plot_galilean_n64_paper_figures.py"),
        "--label-runs-csv",
        str(table_dir / "2d_galilean_n64_label_efficiency_lambda_0p1.runs.csv"),
        "--severity-runs-csv",
        str(table_dir / "2d_galilean_n64_2pct_ood_severity_lambda_0p1.runs.csv"),
        "--out-dir",
        str(figures_dir),
    ]
    subprocess.run(cmd, check=True)


def _regenerate_qualitative_figure() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "plot_navier_stokes_qualitative.py"),
        "--figure-config",
        str(ROOT / "figure_specs" / "2d_galilean_n64_id_ood_seed23.yaml"),
    ]
    subprocess.run(cmd, check=True)


def _regenerate_diagnostics(*, table_dir: Path, figures_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_paper_diagnostics.py"),
        "--headline-csv",
        str(table_dir / "2d_galilean_n64_2pct_headline_lambda_0p1.runs.csv"),
        "--label-csv",
        str(table_dir / "2d_galilean_n64_label_efficiency_lambda_0p1.runs.csv"),
        "--lambda-csv",
        str(table_dir / "2d_galilean_n64_2pct_lambda_robustness.runs.csv"),
        "--severity-csv",
        str(table_dir / "2d_galilean_n64_2pct_ood_severity_lambda_0p1.runs.csv"),
        "--table-dir",
        str(table_dir),
        "--figures-dir",
        str(figures_dir),
        "--root",
        str(ROOT),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate cached paper-facing N64 Galilean artifacts."
    )
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out-dir", default="runs/paper_tables")
    parser.add_argument("--figures-dir", default="runs/figures")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    all_runs = pd.DataFrame(collect_run_rows(args.runs))
    if all_runs.empty:
        raise SystemExit(f"No test_metrics.json files found under {args.runs}")

    selected_training: list[pd.DataFrame] = []
    aggregates: dict[str, pd.DataFrame] = {}
    for table_name, output in TABLE_OUTPUTS.items():
        rows = _select_training_rows(all_runs, table_name)
        selected_training.append(rows)
        aggregates[table_name] = _write_table(
            rows,
            out_dir / output["out_prefix"],
            list(output["group_cols"]),
        )

    severity_rows = _collect_severity_rows()
    aggregates["severity"] = _write_table(
        severity_rows,
        out_dir / "2d_galilean_n64_2pct_ood_severity_lambda_0p1",
        ["severity", "severity_scale", "max_boost", "method"],
    )
    oracle_rows = _collect_oracle_rows()
    aggregates["oracle_canonicalization"] = _write_table(
        oracle_rows,
        out_dir / "2d_galilean_n64_2pct_oracle_canonicalization",
        ["severity", "severity_scale", "max_boost", "method"],
    )

    manifest_rows = []
    manifest_rows.extend(_manifest_for_training(pd.concat(selected_training, ignore_index=True)))
    manifest_rows.extend(_manifest_for_severity(severity_rows))
    manifest_rows.extend(_manifest_for_severity(oracle_rows))
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["artifact_type", "paper_table", "data_fraction", "method", "severity", "seed"],
        na_position="last",
    )
    manifest_path = out_dir / "2d_galilean_n64_final_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path}", flush=True)
    if not manifest["complete"].all():
        incomplete = manifest[~manifest["complete"]]
        raise SystemExit(
            "Incomplete final artifacts:\n"
            + incomplete[["artifact_type", "run_dir", "missing_files"]].to_string(index=False)
        )

    _claim_summary(
        headline=aggregates["headline"],
        label_efficiency=aggregates["label_efficiency"],
        severity=aggregates["severity"],
        out_path=out_dir / "2d_galilean_n64_claim_summary.csv",
    )
    _regenerate_diagnostics(table_dir=out_dir, figures_dir=Path(args.figures_dir))
    if not args.skip_figures:
        _regenerate_figures(table_dir=out_dir, figures_dir=Path(args.figures_dir))
        _regenerate_qualitative_figure()


if __name__ == "__main__":
    main()
