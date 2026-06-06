#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import pandas as pd


METHOD_LABELS = {
    "aug": "Augmentation, 6 steps",
    "aug_orbit": "Augmentation + orbit (lambda=0.10), 4 steps",
    "aug_steps_6": "Augmentation, 6 steps",
    "aug_orbit_steps_4": "Augmentation + orbit (lambda=0.05), 4 steps",
    "aug_orbit_lambda_0.1_steps_4": "Augmentation + orbit (lambda=0.10), 4 steps",
    "semi_aug_orbit": "Semi-supervised orbit",
}

METHOD_STYLES = {
    "aug": {"color": "#4c78a8", "marker": "o", "offset": 0.94},
    "aug_orbit": {"color": "#f58518", "marker": "s", "offset": 1.06},
    "aug_steps_6": {"color": "#4c78a8", "marker": "o", "offset": 0.94},
    "aug_orbit_steps_4": {"color": "#f58518", "marker": "s", "offset": 1.06},
    "aug_orbit_lambda_0.1_steps_4": {"color": "#f58518", "marker": "s", "offset": 1.06},
    "semi_aug_orbit": {"color": "#54a24b", "marker": "^", "offset": 1.00},
}

METRIC_LABELS = {
    "relative_l2": "Relative L2",
    "orbit_ood_relative_l2": "Orbit OOD relative L2",
    "equivariance_defect_relative": "Equivariance defect",
}

SEVERITY_ORDER = ["boost_0.10", "boost_0.20", "train_radius", "boost_0.35", "boost_0.50"]


def _std_aggregate(df: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    grouped = df.groupby(keys, as_index=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in grouped.columns
    ]
    return grouped


def _format_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#dddddd", linewidth=0.75, alpha=0.85)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_metric_curve(
    ax: plt.Axes,
    df: pd.DataFrame,
    agg: pd.DataFrame,
    *,
    x_col: str,
    methods: list[str],
    metric: str,
) -> None:
    for method in methods:
        raw_rows = df[df["method"] == method].sort_values(x_col)
        agg_rows = agg[agg["method"] == method].sort_values(x_col)
        if agg_rows.empty:
            continue
        style = METHOD_STYLES[method]
        x = agg_rows[x_col].astype(float)
        y = agg_rows[f"{metric}_mean"].astype(float)
        err = agg_rows[f"{metric}_std"].fillna(0.0).astype(float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            color=style["color"],
            marker=style["marker"],
            markersize=5.2,
            linewidth=2.0,
            capsize=3.2,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        point_x = raw_rows[x_col].astype(float) * float(style["offset"])
        ax.scatter(
            point_x,
            raw_rows[metric].astype(float),
            color=style["color"],
            s=17,
            alpha=0.34,
            linewidths=0,
            zorder=2,
        )
    ax.set_ylabel(METRIC_LABELS[metric])
    _format_axes(ax)


def plot_label_efficiency(runs_csv: Path, out_prefix: Path) -> None:
    df = pd.read_csv(runs_csv)
    metrics = ["orbit_ood_relative_l2", "equivariance_defect_relative"]
    required = {"method", "data_fraction", "seed", *metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{runs_csv} missing columns: {', '.join(missing)}")

    methods = ["aug", "aug_orbit"]
    df = df[df["method"].isin(methods)].copy()
    agg = _std_aggregate(df, ["data_fraction", "method"], metrics)
    fractions = sorted(float(value) for value in df["data_fraction"].unique())
    tick_labels = [f"{100.0 * value:g}%" for value in fractions]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        _plot_metric_curve(
            ax,
            df,
            agg,
            x_col="data_fraction",
            methods=methods,
            metric=metric,
        )
        ax.set_xscale("log")
        ax.set_xticks(fractions)
        ax.set_xticklabels(tick_labels)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Labeled training fraction")
    axes[0].set_title("Symmetry-induced OOD error")
    axes[1].set_title("Equivariance defect")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("N64 Galilean label efficiency", y=1.04, fontsize=12)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_ood_severity(runs_csv: Path, out_prefix: Path) -> None:
    df = pd.read_csv(runs_csv)
    metrics = ["orbit_ood_relative_l2", "equivariance_defect_relative"]
    required = {"method", "max_boost", "seed", *metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{runs_csv} missing columns: {', '.join(missing)}")

    orbit_method = (
        "aug_orbit_lambda_0.1_steps_4"
        if "aug_orbit_lambda_0.1_steps_4" in set(df["method"])
        else "aug_orbit_steps_4"
    )
    methods = ["aug_steps_6", orbit_method]
    df = df[df["method"].isin(methods)].copy()
    df["severity_order"] = df["severity"].map({name: i for i, name in enumerate(SEVERITY_ORDER)})
    df = df.sort_values(["severity_order", "method", "seed"])
    agg = _std_aggregate(df, ["max_boost", "method"], metrics)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.05), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        _plot_metric_curve(
            ax,
            df,
            agg,
            x_col="max_boost",
            methods=methods,
            metric=metric,
        )
        ax.set_xlabel("Max Galilean boost")
        ax.axvline(0.25, color="#8c8c8c", linestyle="--", linewidth=1.0, alpha=0.8)
        ymin, ymax = ax.get_ylim()
        ax.text(
            0.258,
            ymin + 0.62 * (ymax - ymin),
            "train radius",
            rotation=90,
            va="top",
            ha="left",
            color="#5f5f5f",
            fontsize=8,
        )
    axes[0].set_title("Symmetry-induced OOD error")
    axes[1].set_title("Equivariance defect")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("N64 Galilean OOD severity, 2% labels", y=1.04, fontsize=12)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def plot_semisupervised_ood_severity(
    supervised_runs_csv: Path, semi_runs_csv: Path, out_prefix: Path
) -> None:
    supervised = pd.read_csv(supervised_runs_csv)
    semi = pd.read_csv(semi_runs_csv)
    df = pd.concat([supervised, semi], ignore_index=True)
    metrics = ["orbit_ood_relative_l2", "equivariance_defect_relative"]
    required = {"method", "max_boost", "severity", "seed", *metrics}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"semi-supervised severity inputs missing columns: {', '.join(missing)}")

    orbit_method = (
        "aug_orbit_lambda_0.1_steps_4"
        if "aug_orbit_lambda_0.1_steps_4" in set(df["method"])
        else "aug_orbit_steps_4"
    )
    methods = ["aug_steps_6", orbit_method, "semi_aug_orbit"]
    df = df[df["method"].isin(methods)].copy()
    df["severity_order"] = df["severity"].map({name: i for i, name in enumerate(SEVERITY_ORDER)})
    df = df.sort_values(["severity_order", "method", "seed"])
    agg = _std_aggregate(df, ["max_boost", "method"], metrics)

    fig, axes = plt.subplots(1, 2, figsize=(8.25, 3.05), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        _plot_metric_curve(
            ax,
            df,
            agg,
            x_col="max_boost",
            methods=methods,
            metric=metric,
        )
        ax.set_xlabel("Max Galilean boost")
        ax.axvline(0.25, color="#8c8c8c", linestyle="--", linewidth=1.0, alpha=0.8)
        ymin, ymax = ax.get_ylim()
        ax.text(
            0.258,
            ymin + 0.62 * (ymax - ymin),
            "train radius",
            rotation=90,
            va="top",
            ha="left",
            color="#5f5f5f",
            fontsize=8,
        )
    axes[0].set_title("Symmetry-induced OOD error")
    axes[1].set_title("Equivariance defect")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("N64 Galilean semi-supervised severity, 2% labels", y=1.04, fontsize=12)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paper-facing N64 Galilean figures.")
    parser.add_argument(
        "--label-runs-csv",
        default="runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.runs.csv",
    )
    parser.add_argument(
        "--severity-runs-csv",
        default="runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.runs.csv",
    )
    parser.add_argument(
        "--semi-severity-runs-csv",
        default="runs/paper_tables/2d_galilean_n64_2pct_semi_supervised_ood_severity.runs.csv",
    )
    parser.add_argument("--out-dir", default="runs/figures")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.dpi": 150,
        }
    )

    out_dir = Path(args.out_dir)
    label_prefix = out_dir / "2d_galilean_n64_label_efficiency_lambda_0p1"
    severity_prefix = out_dir / "2d_galilean_n64_2pct_ood_severity_lambda_0p1"
    semi_severity_prefix = out_dir / "2d_galilean_n64_2pct_semi_supervised_ood_severity_lambda_0p1"
    plot_label_efficiency(Path(args.label_runs_csv), label_prefix)
    plot_ood_severity(Path(args.severity_runs_csv), severity_prefix)
    print(f"wrote {label_prefix.with_suffix('.png')}")
    print(f"wrote {label_prefix.with_suffix('.pdf')}")
    print(f"wrote {severity_prefix.with_suffix('.png')}")
    print(f"wrote {severity_prefix.with_suffix('.pdf')}")
    semi_runs_csv = Path(args.semi_severity_runs_csv)
    if semi_runs_csv.exists() and semi_runs_csv.stat().st_size > 0:
        plot_semisupervised_ood_severity(
            Path(args.severity_runs_csv), semi_runs_csv, semi_severity_prefix
        )
        print(f"wrote {semi_severity_prefix.with_suffix('.png')}")
        print(f"wrote {semi_severity_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
