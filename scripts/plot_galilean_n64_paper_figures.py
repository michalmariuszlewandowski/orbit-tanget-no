#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import NullLocator
import pandas as pd


METHOD_LABELS = {
    "aug": "Augmentation",
    "aug_orbit": "Augmentation + orbit",
    "aug_steps_6": "Aug, 6 updates/epoch",
    "aug_orbit_steps_4": "Augmentation + orbit (lambda=0.05), 4 steps",
    "aug_orbit_lambda_0.1_steps_4": r"Aug + orbit, $\lambda=0.10$, 4 updates/epoch",
}

METHOD_STYLES = {
    "aug": {"color": "#4c78a8", "marker": "o", "linestyle": "-", "offset": 0.94},
    "aug_orbit": {"color": "#f58518", "marker": "s", "linestyle": "--", "offset": 1.06},
    "aug_steps_6": {"color": "#4c78a8", "marker": "o", "linestyle": "-", "offset": 0.94},
    "aug_orbit_steps_4": {"color": "#f58518", "marker": "s", "linestyle": "--", "offset": 1.06},
    "aug_orbit_lambda_0.1_steps_4": {
        "color": "#f58518",
        "marker": "s",
        "linestyle": "--",
        "offset": 1.06,
    },
}

METRIC_LABELS = {
    "relative_l2": r"Relative $L_2$",
    "orbit_ood_relative_l2": r"Orbit OOD relative $L_2$",
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


def _annotate_train_radius(ax: plt.Axes, *, label_training_range: bool = False) -> None:
    transform = blended_transform_factory(ax.transData, ax.transAxes)
    xlim = ax.get_xlim()
    ax.axvspan(0.0, 0.25, color="#ead28c", alpha=0.55, zorder=0)
    ax.axvline(0.25, color="#4f4f4f", linestyle=":", linewidth=1.6, alpha=0.98)
    ax.set_xlim(xlim)
    if label_training_range:
        ax.text(
            0.105,
            0.93,
            "training range",
            transform=transform,
            color="#514936",
            fontsize=9.5,
            ha="left",
            va="top",
        )
    ax.text(
        0.252,
        0.98,
        r"$r_{\mathrm{train}}=0.25$",
        transform=transform,
        color="#333333",
        fontsize=12,
        fontweight="semibold",
        ha="left",
        va="top",
    )


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
            linestyle=style["linestyle"],
            markersize=5.8,
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
            marker=style["marker"],
            s=22,
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

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.35), constrained_layout=True)
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
    fig.suptitle("N64 Galilean label efficiency", y=1.04, fontsize=14)
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

    fig = plt.figure(figsize=(8.8, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.14, 0.16, 1.0])
    title_ax = fig.add_subplot(grid[0, :])
    legend_ax = fig.add_subplot(grid[1, :])
    axes = [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    for aux_ax in [title_ax, legend_ax]:
        aux_ax.axis("off")
    title_ax.text(
        0.5,
        0.55,
        "N64 Galilean OOD severity, 2% labels",
        ha="center",
        va="center",
        fontsize=14,
    )
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        _plot_metric_curve(
            ax,
            df,
            agg,
            x_col="max_boost",
            methods=methods,
            metric=metric,
        )
        ax.set_xlabel("Max Galilean boost")
        _annotate_train_radius(ax, label_training_range=idx == 0)
    axes[0].set_title("Symmetry-induced OOD error")
    axes[1].set_title("Equivariance defect")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center",
        ncol=2,
        columnspacing=1.4,
        handlelength=2.8,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot N64 Galilean experiment figures.")
    parser.add_argument(
        "--label-runs-csv",
        default="runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.runs.csv",
    )
    parser.add_argument(
        "--severity-runs-csv",
        default="runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.runs.csv",
    )
    parser.add_argument("--out-dir", default="runs/figures")
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 12.5,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.5,
            "figure.dpi": 150,
        }
    )

    out_dir = Path(args.out_dir)
    label_prefix = out_dir / "2d_galilean_n64_label_efficiency_lambda_0p1"
    severity_prefix = out_dir / "2d_galilean_n64_2pct_ood_severity_lambda_0p1"
    plot_label_efficiency(Path(args.label_runs_csv), label_prefix)
    plot_ood_severity(Path(args.severity_runs_csv), severity_prefix)
    print(f"wrote {label_prefix.with_suffix('.png')}")
    print(f"wrote {label_prefix.with_suffix('.pdf')}")
    print(f"wrote {severity_prefix.with_suffix('.png')}")
    print(f"wrote {severity_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
