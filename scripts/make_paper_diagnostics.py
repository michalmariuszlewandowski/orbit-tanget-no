#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "baseline": "FNO",
    "baseline_steps_4": "FNO",
    "orbit": "FNO + orbit",
    "aug": "FNO + aug.",
    "aug_orbit": "FNO + aug. + orbit",
    "aug_steps_6": "FNO + aug.",
    "aug_orbit_lambda_0.1_steps_4": "FNO + aug. + orbit",
}

METHOD_COLORS = {
    "baseline": "#7f7f7f",
    "orbit": "#54a24b",
    "aug": "#4c78a8",
    "aug_orbit": "#f58518",
    "aug_steps_6": "#4c78a8",
    "aug_orbit_lambda_0.1_steps_4": "#f58518",
}

TRAIN_METHODS_WITH_AUG = {"aug", "augmentation", "aug_orbit", "orbit_aug"}
TRAIN_METHODS_WITH_ORBIT = {
    "orbit",
    "orb",
    "aug_orbit",
    "orbit_aug",
}


def _norm_path(value: Any) -> str:
    return str(value).replace("\\", "/")


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input CSV: {path}")
    return pd.read_csv(path)


def _finite_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)


def _pearson(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return math.nan
    return float(x.corr(y, method="pearson"))


def _spearman(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 2:
        return math.nan
    return _pearson(x.rank(method="average"), y.rank(method="average"))


def _residualize(y: pd.Series, covariates: pd.DataFrame) -> pd.Series:
    if y.empty:
        return y
    cov = covariates.copy()
    numeric_cols = [col for col in cov.columns if pd.api.types.is_numeric_dtype(cov[col])]
    categorical_cols = [col for col in cov.columns if col not in numeric_cols]
    for col in numeric_cols:
        values = pd.to_numeric(cov[col], errors="coerce")
        cov[col] = values.fillna(values.median() if values.notna().any() else 0.0)
    cov = pd.get_dummies(cov, columns=categorical_cols, dummy_na=True, drop_first=True)
    x = cov.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x, y.to_numpy(dtype=float), rcond=None)
    return pd.Series(y.to_numpy(dtype=float) - x @ beta, index=y.index)


def _partial_pearson(df: pd.DataFrame, covariate_cols: list[str]) -> float:
    if len(df) <= len(covariate_cols) + 2:
        return math.nan
    cov = df[covariate_cols].copy()
    defect_resid = _residualize(df["equivariance_defect_relative"], cov)
    ood_resid = _residualize(df["orbit_ood_relative_l2"], cov)
    return _pearson(defect_resid, ood_resid)


def _training_observations(paths: list[Path]) -> pd.DataFrame:
    rows: dict[str, dict[str, Any]] = {}
    for path in paths:
        df = _read_csv(path)
        for record in df.to_dict(orient="records"):
            run_dir = _norm_path(record.get("run_dir", ""))
            if not run_dir or run_dir in rows:
                continue
            rows[run_dir] = {
                "source": "training",
                "run_dir": run_dir,
                "method": record.get("method"),
                "method_label": METHOD_LABELS.get(str(record.get("method")), str(record.get("method"))),
                "seed": record.get("seed"),
                "data_fraction": record.get("data_fraction", record.get("config.training.data_fraction")),
                "lambda_orbit": record.get("lambda_orbit", record.get("config.training.lambda_orbit")),
                "max_boost": record.get("config.symmetry.max_boost", record.get("config.dataset.max_boost")),
                "relative_l2": record.get("relative_l2"),
                "orbit_ood_relative_l2": record.get("orbit_ood_relative_l2"),
                "equivariance_defect_relative": record.get("equivariance_defect_relative"),
            }
    return pd.DataFrame(rows.values())


def _severity_observations(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        method = str(record.get("method"))
        rows.append(
            {
                "source": "severity",
                "run_dir": _norm_path(record.get("run_dir", "")),
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "seed": record.get("seed"),
                "data_fraction": 0.02,
                "lambda_orbit": 0.10 if method == "aug_orbit_lambda_0.1_steps_4" else math.nan,
                "max_boost": record.get("max_boost"),
                "severity": record.get("severity"),
                "relative_l2": record.get("relative_l2"),
                "orbit_ood_relative_l2": record.get("orbit_ood_relative_l2"),
                "equivariance_defect_relative": record.get("equivariance_defect_relative"),
            }
        )
    return pd.DataFrame(rows)


def build_correlation_observations(
    *,
    headline_csv: Path,
    label_csv: Path,
    lambda_csv: Path,
    severity_csv: Path,
) -> pd.DataFrame:
    training = _training_observations([headline_csv, label_csv, lambda_csv])
    severity = _severity_observations(severity_csv)
    observations = pd.concat([training, severity], ignore_index=True)
    cols = ["orbit_ood_relative_l2", "equivariance_defect_relative"]
    observations = _finite_rows(observations, cols)
    return observations.sort_values(["source", "method", "data_fraction", "max_boost", "seed"])


def write_correlation_tables(observations: pd.DataFrame, out_prefix: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = {
        "all": observations,
        "training": observations[observations["source"] == "training"],
        "severity": observations[observations["source"] == "severity"],
    }
    for scope, df in scopes.items():
        df = _finite_rows(df, ["equivariance_defect_relative", "orbit_ood_relative_l2"])
        covariates = [col for col in ["source", "method", "data_fraction", "max_boost", "lambda_orbit"] if col in df.columns]
        rows.append(
            {
                "scope": scope,
                "n": len(df),
                "pearson_r": _pearson(df["equivariance_defect_relative"], df["orbit_ood_relative_l2"]),
                "spearman_rho": _spearman(df["equivariance_defect_relative"], df["orbit_ood_relative_l2"]),
                "partial_pearson_r": _partial_pearson(df, covariates),
                "controls": "+".join(covariates),
            }
        )
    table = pd.DataFrame(rows)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    table.to_csv(out_prefix.with_suffix(".csv"), index=False)
    with out_prefix.with_suffix(".tex").open("w", encoding="utf-8") as f:
        f.write(table.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.3f}"))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")
    return table


def plot_correlation(observations: pd.DataFrame, table: pd.DataFrame, out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.7, 3.45), constrained_layout=True)
    for method, rows in observations.groupby("method", sort=True):
        color = METHOD_COLORS.get(str(method), "#333333")
        label = METHOD_LABELS.get(str(method), str(method))
        train_rows = rows[rows["source"] == "training"]
        severity_rows = rows[rows["source"] == "severity"]
        if not train_rows.empty:
            ax.scatter(
                train_rows["equivariance_defect_relative"],
                train_rows["orbit_ood_relative_l2"],
                label=label,
                color=color,
                marker="o",
                s=28,
                alpha=0.7,
                edgecolors="none",
            )
        if not severity_rows.empty:
            ax.scatter(
                severity_rows["equivariance_defect_relative"],
                severity_rows["orbit_ood_relative_l2"],
                label=f"{label}, severity",
                color=color,
                marker="s",
                s=24,
                alpha=0.45,
                edgecolors="none",
            )
    df = observations.sort_values("equivariance_defect_relative")
    x = df["equivariance_defect_relative"].to_numpy(dtype=float)
    y = df["orbit_ood_relative_l2"].to_numpy(dtype=float)
    if len(df) >= 2:
        coef = np.polyfit(x, y, deg=1)
        ax.plot(x, coef[0] * x + coef[1], color="#222222", linewidth=1.3, alpha=0.8)
    all_row = table[table["scope"] == "all"].iloc[0]
    ax.text(
        0.04,
        0.96,
        f"Pearson r={all_row['pearson_r']:.2f}\nSpearman rho={all_row['spearman_rho']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2.0},
    )
    ax.set_xlabel("Equivariance defect")
    ax.set_ylabel("Orbit OOD relative L2")
    ax.set_title("Defect predicts Galilean OOD error")
    ax.grid(True, color="#dddddd", linewidth=0.75, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        dedup.setdefault(label, handle)
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=7.2, loc="lower right")
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
        print(f"wrote {out_prefix.with_suffix(suffix)}")
    plt.close(fig)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _estimate_forward_multiplier(method: str) -> int:
    method = method.lower()
    count = 1
    if method in TRAIN_METHODS_WITH_AUG:
        count += 1
    if method in TRAIN_METHODS_WITH_ORBIT:
        count += 1
    return count


def _compute_row(record: dict[str, Any], root: Path) -> dict[str, Any]:
    run_dir = root / str(record["run_dir"])
    train_rows = _read_jsonl(run_dir / "train_metrics.jsonl")
    method = str(record.get("method", record.get("config.training.method", ""))).lower()
    batch_size = int(float(record.get("config.training.batch_size", record.get("batch_size", 0))))
    supervised_steps = int(sum(int(row.get("train_supervised_steps", 0)) for row in train_rows))
    optimizer_steps = supervised_steps
    forward_multiplier = _estimate_forward_multiplier(method)
    model_forward_passes = supervised_steps * forward_multiplier
    augmentation_batches = supervised_steps if method in TRAIN_METHODS_WITH_AUG else 0
    orbit_batches = supervised_steps if method in TRAIN_METHODS_WITH_ORBIT else 0
    wall_seconds = pd.to_numeric(pd.Series([record.get("train_wall_seconds")]), errors="coerce").iloc[0]
    wall_source = "recorded"
    if not np.isfinite(wall_seconds):
        initial_rng = run_dir / "rng_state_initial.pt"
        final_rng = run_dir / "rng_state_final.pt"
        if initial_rng.exists() and final_rng.exists():
            wall_seconds = max(0.0, final_rng.stat().st_mtime - initial_rng.stat().st_mtime)
            wall_source = "rng_file_mtime"
        else:
            wall_seconds = math.nan
            wall_source = "missing"
    return {
        "run_dir": _norm_path(record.get("run_dir")),
        "method": record.get("method"),
        "method_label": METHOD_LABELS.get(str(record.get("method")), str(record.get("method"))),
        "seed": record.get("seed"),
        "data_fraction": record.get("data_fraction"),
        "steps_per_epoch": record.get("steps_per_epoch"),
        "lambda_orbit": record.get("lambda_orbit"),
        "epochs_logged": len(train_rows),
        "optimizer_steps": optimizer_steps,
        "model_forward_passes_est": model_forward_passes,
        "backward_passes_est": optimizer_steps,
        "augmented_labeled_samples_est": augmentation_batches * batch_size,
        "orbit_consistency_samples_est": orbit_batches * batch_size,
        "eval_seconds": record.get("eval_seconds"),
        "latency_ms_per_sample": record.get("latency_ms_per_sample"),
        "train_wall_minutes_est": wall_seconds / 60.0 if np.isfinite(wall_seconds) else math.nan,
        "train_wall_source": wall_source,
    }


def write_compute_table(headline_csv: Path, out_prefix: Path, root: Path) -> pd.DataFrame:
    df = _read_csv(headline_csv)
    rows = [_compute_row(record, root) for record in df.to_dict(orient="records")]
    runs = pd.DataFrame(rows)
    group_cols = ["method_label", "data_fraction", "steps_per_epoch"]
    value_cols = [
        "optimizer_steps",
        "model_forward_passes_est",
        "backward_passes_est",
        "augmented_labeled_samples_est",
        "orbit_consistency_samples_est",
        "train_wall_minutes_est",
        "eval_seconds",
        "latency_ms_per_sample",
    ]
    aggregate = (
        runs.groupby(group_cols, dropna=False)[value_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregate.columns
    ]
    aggregate["relative_train_time_vs_aug"] = np.nan
    for data_fraction, group in aggregate.groupby("data_fraction", dropna=False):
        aug_rows = group[group["method_label"] == "FNO + aug."]
        if aug_rows.empty:
            continue
        aug_minutes = float(aug_rows["train_wall_minutes_est_mean"].iloc[0])
        if not np.isfinite(aug_minutes) or aug_minutes <= 0.0:
            continue
        mask = aggregate["data_fraction"].eq(data_fraction)
        aggregate.loc[mask, "relative_train_time_vs_aug"] = (
            aggregate.loc[mask, "train_wall_minutes_est_mean"] / aug_minutes
        )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        f.write(aggregate.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.3f}"))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}")
    return aggregate


def write_oracle_summary_table(oracle_runs_csv: Path, out_prefix: Path) -> pd.DataFrame | None:
    if not oracle_runs_csv.exists() or oracle_runs_csv.stat().st_size == 0:
        return None
    df = _read_csv(oracle_runs_csv)
    required = {
        "severity",
        "severity_scale",
        "max_boost",
        "method",
        "orbit_ood_relative_l2",
        "oracle_canonical_ood_relative_l2",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{oracle_runs_csv} missing columns: {', '.join(missing)}")
    df["method_label"] = df["method"].map(lambda value: METHOD_LABELS.get(str(value), str(value)))
    aggregate = (
        df.groupby(["severity", "severity_scale", "max_boost", "method_label"], dropna=False)[
            ["orbit_ood_relative_l2", "oracle_canonical_ood_relative_l2"]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregate.columns
    ]
    aggregate["oracle_gap_pct_mean"] = 100.0 * (
        aggregate["orbit_ood_relative_l2_mean"]
        - aggregate["oracle_canonical_ood_relative_l2_mean"]
    ) / aggregate["oracle_canonical_ood_relative_l2_mean"].clip(lower=1e-12)
    aggregate = aggregate.sort_values(["severity_scale", "method_label"])
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(out_prefix.with_suffix(".csv"), index=False)
    with out_prefix.with_suffix(".tex").open("w", encoding="utf-8") as f:
        f.write(aggregate.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.3f}"))
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper diagnostics from cached N64 results.")
    parser.add_argument("--headline-csv", default="runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.runs.csv")
    parser.add_argument("--label-csv", default="runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.runs.csv")
    parser.add_argument("--lambda-csv", default="runs/paper_tables/2d_galilean_n64_2pct_lambda_robustness.runs.csv")
    parser.add_argument("--severity-csv", default="runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.runs.csv")
    parser.add_argument("--oracle-runs-csv", default="runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization.runs.csv")
    parser.add_argument("--table-dir", default="runs/paper_tables")
    parser.add_argument("--figures-dir", default="runs/figures")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root)
    table_dir = Path(args.table_dir)
    figures_dir = Path(args.figures_dir)
    observations = build_correlation_observations(
        headline_csv=Path(args.headline_csv),
        label_csv=Path(args.label_csv),
        lambda_csv=Path(args.lambda_csv),
        severity_csv=Path(args.severity_csv),
    )
    corr_prefix = table_dir / "2d_galilean_n64_defect_ood_correlation"
    corr_table = write_correlation_tables(observations, corr_prefix)
    plot_correlation(observations, corr_table, figures_dir / "2d_galilean_n64_defect_ood_correlation")
    write_compute_table(
        Path(args.headline_csv),
        table_dir / "2d_galilean_n64_training_compute",
        root,
    )
    write_oracle_summary_table(
        Path(args.oracle_runs_csv),
        table_dir / "2d_galilean_n64_2pct_oracle_canonicalization_summary",
    )


if __name__ == "__main__":
    main()
