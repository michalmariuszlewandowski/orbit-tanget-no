#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.container import ErrorbarContainer
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.transforms import blended_transform_factory

METHOD_LABELS = {
    "aug": "Aug.",
    "aug_orbit": "Aug. + LOCO",
    "aug_steps_6": "Aug.",
    "aug_orbit_steps_4": "Aug. + LOCO",
    "aug_orbit_lambda_0.1_steps_4": "Aug. + LOCO",
}

METHOD_STYLES = {
    "aug": {"color": "#4c78a8", "marker": "o", "linestyle": "-"},
    "aug_orbit": {"color": "#f58518", "marker": "s", "linestyle": "--"},
    "aug_steps_6": {"color": "#4c78a8", "marker": "o", "linestyle": "-"},
    "aug_orbit_steps_4": {"color": "#f58518", "marker": "s", "linestyle": "--"},
    "aug_orbit_lambda_0.1_steps_4": {
        "color": "#f58518",
        "marker": "s",
        "linestyle": "--",
    },
}

METRIC_LABELS = {
    "relative_l2": r"Relative $L_2$",
    "orbit_ood_relative_l2": "Orbit OOD\nrelative $L_2$",
    "equivariance_defect_relative": "Equivariance\ndefect",
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
    ax.grid(True, color="#dddddd", linewidth=0.65, alpha=0.5)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _create_figure(title: str | None = None) -> tuple[plt.Figure, plt.Axes, list[plt.Axes]]:
    fig = plt.figure(figsize=(9.8, 3.85), constrained_layout=True)
    if title is None:
        grid = fig.add_gridspec(2, 2, height_ratios=[0.12, 1.0])
        legend_ax = fig.add_subplot(grid[0, :])
        legend_ax.axis("off")
        axes = [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]
        return fig, legend_ax, axes
    grid = fig.add_gridspec(3, 2, height_ratios=[0.08, 0.16, 1.0])
    title_ax = fig.add_subplot(grid[0, :])
    legend_ax = fig.add_subplot(grid[1, :])
    axes = [fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1])]
    for aux_ax in [title_ax, legend_ax]:
        aux_ax.axis("off")
    title_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=13)
    return fig, legend_ax, axes


def _add_method_legend(legend_ax: plt.Axes, source_ax: plt.Axes, *, compact: bool = False) -> None:
    handles, labels = source_ax.get_legend_handles_labels()
    legend_options = {}
    if compact:
        handles = [
            handle.lines[0] if isinstance(handle, ErrorbarContainer) else handle
            for handle in handles
        ]
        legend_options = {
            "fontsize": plt.rcParams["axes.labelsize"],
            "markerscale": 0.75,
            "borderpad": 0.0,
        }
    legend_ax.legend(
        handles,
        labels,
        frameon=False,
        loc="center",
        ncol=2,
        columnspacing=1.3,
        handlelength=1.5,
        handletextpad=0.45,
        **legend_options,
    )


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
            fontsize=11,
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
    """Plot mean +/- sample SD and faint per-seed results at their actual x values."""
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
            markersize=9.5,
            linewidth=1.8,
            elinewidth=1.2,
            capsize=3.5,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        ax.scatter(
            raw_rows[x_col].astype(float),
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


def plot_label_efficiency(
    runs_csv: Path, out_prefix: Path, *, formats: tuple[str, ...] = ("png", "pdf")
) -> None:
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

    fig, legend_ax, axes = _create_figure("N64 Galilean label efficiency")
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
    _add_method_legend(legend_ax, axes[0])
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        fig.savefig(out_prefix.with_suffix(f".{output_format}"), bbox_inches="tight")
    plt.close(fig)


def plot_ood_severity(
    runs_csv: Path, out_prefix: Path, *, formats: tuple[str, ...] = ("png", "pdf")
) -> None:
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

    fig, legend_ax, axes = _create_figure()
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        _plot_metric_curve(
            ax,
            df,
            agg,
            x_col="max_boost",
            methods=methods,
            metric=metric,
        )
        _annotate_train_radius(ax, label_training_range=idx == 0)
    axes[0].set_title("(a) Prediction error \u2193", fontweight="bold")
    axes[1].set_title("(b) Equivariance defect \u2193", fontweight="bold")
    axes[0].set_ylabel(r"Relative $L_2$ error")
    axes[1].set_ylabel("Relative defect")
    fig.supxlabel("Max Galilean boost", fontsize=plt.rcParams["axes.labelsize"])
    _add_method_legend(legend_ax, axes[0], compact=True)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for output_format in formats:
        fig.savefig(out_prefix.with_suffix(f".{output_format}"), bbox_inches="tight")
    plt.close(fig)

    orbit_weight = "0.1" if orbit_method == "aug_orbit_lambda_0.1_steps_4" else "0.05"
    caption = (
        "Galilean transformation severity for two-dimensional Navier-Stokes prediction "
        "at N=64 with 2% labeled training data, comparing augmentation (Aug.) with "
        f"augmentation plus local orbit consistency (Aug. + LOCO; lambda={orbit_weight}). "
        "Evaluation uses symmetry-transformed test inputs and targets with Galilean "
        "boosts sampled up to the maximum shown on the shared x-axis. "
        "(a) Relative L2 prediction error against the transformed target. "
        "(b) Relative equivariance defect between the prediction on the transformed "
        "input and the transformed prediction on the original input, normalized by "
        "the latter's L2 norm. Lower values are better in both panels. "
        "Curves and error bars show the mean and sample standard deviation across "
        "training seeds; faint markers show individual seeds. "
        "Shading marks the training transformation range, with its maximum boost "
        "r_train=0.25 indicated by the dotted line.\n"
    )
    out_prefix.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")


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
    parser.add_argument("--figure", choices=["all", "label", "severity"], default="all")
    parser.add_argument("--suffix", default="", help="Append to output names, e.g. --suffix=-v2.")
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"]
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 13.5,
            "axes.labelsize": 16.5,
            "xtick.labelsize": 15.5,
            "ytick.labelsize": 15.5,
            "legend.fontsize": 18,
            "legend.markerscale": 1.15,
            "figure.dpi": 150,
        }
    )

    out_dir = Path(args.out_dir)
    formats = tuple(args.formats)
    label_prefix = out_dir / f"2d_galilean_n64_label_efficiency_lambda_0p1{args.suffix}"
    severity_prefix = out_dir / f"2d_galilean_n64_2pct_ood_severity_lambda_0p1{args.suffix}"
    if args.figure in {"all", "label"}:
        plot_label_efficiency(Path(args.label_runs_csv), label_prefix, formats=formats)
        for output_format in formats:
            print(f"wrote {label_prefix.with_suffix(f'.{output_format}')}")
    if args.figure in {"all", "severity"}:
        plot_ood_severity(Path(args.severity_runs_csv), severity_prefix, formats=formats)
        for output_format in formats:
            print(f"wrote {severity_prefix.with_suffix(f'.{output_format}')}")
        print(f"wrote {severity_prefix.with_suffix('.caption.txt')}")


if __name__ == "__main__":
    main()
