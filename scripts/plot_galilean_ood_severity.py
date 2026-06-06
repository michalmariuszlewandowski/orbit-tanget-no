#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "orbit_ood_relative_l2": "Orbit OOD relative L2",
    "equivariance_defect_relative": "Equivariance defect",
}

METHOD_LABELS = {
    "aug": "Augmentation",
    "aug_orbit": "Augmentation + orbit",
}

METHOD_STYLES = {
    "aug": {"color": "#4c78a8", "marker": "o"},
    "aug_orbit": {"color": "#f58518", "marker": "s"},
}


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = list(METRICS)
    grouped = (
        df.groupby(["max_boost", "method"], as_index=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(part for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in grouped.columns
    ]
    return grouped.sort_values(["max_boost", "method"])


def _plot_metric(ax: plt.Axes, agg: pd.DataFrame, metric: str) -> None:
    for method in ["aug", "aug_orbit"]:
        rows = agg[agg["method"] == method].sort_values("max_boost")
        if rows.empty:
            continue
        style = METHOD_STYLES[method]
        x = rows["max_boost"].astype(float)
        y = rows[f"{metric}_mean"].astype(float)
        err = rows[f"{metric}_std"].fillna(0.0).astype(float)
        ax.errorbar(
            x,
            y,
            yerr=err,
            label=METHOD_LABELS[method],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.5,
            capsize=3,
        )
    ax.set_xlabel("Max Galilean boost")
    ax.set_ylabel(METRICS[metric])
    ax.axvline(0.25, color="#8c8c8c", linestyle="--", linewidth=1.0, alpha=0.8)
    y0, y1 = ax.get_ylim()
    ax.text(
        0.258,
        y0 + 0.58 * (y1 - y0),
        "train radius",
        rotation=90,
        va="top",
        ha="left",
        color="#5f5f5f",
        fontsize=8,
    )
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot N64 Galilean OOD severity curves from per-run CSV metrics."
    )
    parser.add_argument(
        "--runs-csv",
        default="runs/paper_tables/2d_galilean_n64_ood_severity.runs.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="runs/figures/2d_galilean_n64_ood_severity",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.runs_csv)
    required = {"max_boost", "method", *METRICS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(missing)}")

    agg = _aggregate(df)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.15), constrained_layout=True)
    _plot_metric(axes[0], agg, "orbit_ood_relative_l2")
    _plot_metric(axes[1], agg, "equivariance_defect_relative")
    axes[0].set_title("Symmetry-induced OOD error")
    axes[1].set_title("Equivariance defect")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("N64 Galilean severity sweep, 5% labels", y=1.04, fontsize=12)

    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_prefix.with_suffix('.png')}")
    print(f"wrote {out_prefix.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
