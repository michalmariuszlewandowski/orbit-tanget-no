"""Render fixed-stress selection context and equivariance residuals."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from otno.plotting.field_analysis import gradient_binned_mae, gradient_magnitude
from otno.plotting.field_panels import field_array

COLORS = {"aug": "#0072b2", "loco": "#d55e00"}
MARKERS = {"aug": "o", "loco": "s"}
LINESTYLES = {"aug": "-", "loco": "--"}
LABELS = {"aug": "Augmentation", "loco": "Aug. + LOCO"}


def _format_diagnostic_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#dedede", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_error_shift(axis: plt.Axes, errors: dict[str, torch.Tensor]) -> None:
    """Show each example's movement from ID error to fixed-stress error."""
    # (a) Per-example movement from ID to the same fixed Galilean stress.
    all_values: list[np.ndarray] = []
    for method in ["aug", "loco"]:
        x = errors[f"{method}_id"].numpy()
        y = errors[f"{method}_ood"].numpy()
        all_values.extend([x, y])
        axis.scatter(
            x,
            y,
            s=19,
            alpha=0.45,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolors="none",
            label=LABELS[method],
        )
        axis.scatter(
            [x.mean()],
            [y.mean()],
            s=115,
            marker="*",
            color=COLORS[method],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
    lower = max(0.0, min(float(values.min()) for values in all_values) * 0.94)
    upper = max(float(values.max()) for values in all_values) * 1.03
    axis.plot([lower, upper], [lower, upper], color="#555555", linestyle=":", linewidth=1.2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"ID relative $L^2$")
    axis.set_ylabel(r"Fixed-stress relative $L^2$")
    axis.set_title("(a) ID-to-stress error shift")
    # Keep the legend readable independently of the point-cloud transparency.
    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=MARKERS[method],
            markersize=6.5,
            markerfacecolor=COLORS[method],
            markeredgecolor="none",
            label=LABELS[method],
        )
        for method in ["aug", "loco"]
    ]
    axis.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower right",
        handlelength=1.0,
        handletextpad=0.5,
    )
    _format_diagnostic_axis(axis)


def _plot_selection_context(
    axis: plt.Axes, errors: dict[str, torch.Tensor], index: int
) -> dict[str, float]:
    """Place the selected example in the full paired-gain distribution."""
    # (b) Full 256-case distribution used by the median-effect selection rule.
    improvement = (errors["aug_ood"] - errors["loco_ood"]).numpy()
    sorted_improvement = np.sort(improvement)
    cumulative = np.arange(1, improvement.size + 1, dtype=float) / improvement.size
    median = float(np.quantile(improvement, 0.5))
    selected = float(improvement[index])
    selected_rank = float(np.count_nonzero(improvement <= selected) / improvement.size)
    x_lower = min(0.0, float(improvement.min()) * 1.06)
    x_upper = max(0.0, float(improvement.max()) * 1.06)
    ecdf_x = np.concatenate(([x_lower], sorted_improvement, [x_upper]))
    ecdf_y = np.concatenate(([0.0], cumulative, [1.0]))
    axis.axvspan(0.0, x_upper, color="#e5f2e3", alpha=0.75)
    axis.axvline(0.0, color="#555555", linewidth=1.2)
    axis.step(ecdf_x, ecdf_y, where="post", color="#6f4e7c", linewidth=2.2)
    axis.axvline(median, color="#6f4e7c", linestyle="--", linewidth=1.2, alpha=0.85)
    axis.scatter(
        [selected],
        [selected_rank],
        marker="D",
        s=58,
        color="#111111",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    axis.annotate(
        f"qualitative case\n({100.0 * selected_rank:.1f}th percentile)",
        xy=(selected, selected_rank),
        xytext=(12, -30),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10.5,
        arrowprops={"arrowstyle": "-", "color": "#222222", "linewidth": 0.9},
    )
    axis.set_xlim(x_lower, x_upper)
    axis.set_ylim(0.0, 1.02)
    axis.set_xlabel(r"Paired OOD gain, $L^2_{\rm aug}-L^2_{\rm A+L}$")
    axis.set_ylabel("Empirical CDF")
    axis.set_title("(b) Selection context (256 cases)")
    axis.text(
        0.03,
        0.96,
        f"{100.0 * np.mean(improvement > 0):.1f}% favor Aug. + LOCO",
        transform=axis.transAxes,
        ha="left",
        va="top",
    )
    _format_diagnostic_axis(axis)
    return {
        "fraction_positive": float(np.mean(improvement > 0)),
        "mean": float(np.mean(improvement)),
        "q10": float(np.quantile(improvement, 0.10)),
        "q50": median,
        "q90": float(np.quantile(improvement, 0.90)),
        "selected_value": selected,
        "selected_empirical_cdf": selected_rank,
    }


def _plot_gradient_errors(
    axis: plt.Axes, fields: dict[str, torch.Tensor]
) -> dict[str, dict[str, list[float]]]:
    """Compare MAE across target-gradient percentiles with example-level CIs."""
    # (c) A target-gradient-conditioned view of the fixed-stress spatial errors.
    gradient_curves = {
        "aug": gradient_binned_mae(fields["target_ood"], fields["aug_ood"]),
        "loco": gradient_binned_mae(fields["target_ood"], fields["loco_ood"]),
    }
    for method in ["aug", "loco"]:
        curve = gradient_curves[method]
        x = np.asarray(curve["percentile_midpoints"])
        mean = np.asarray(curve["mean_absolute_error"])
        ci95 = np.asarray(curve["normal_approximation_ci95_half_width"])
        axis.plot(
            x,
            mean,
            color=COLORS[method],
            linewidth=2.2,
            linestyle=LINESTYLES[method],
            marker=MARKERS[method],
            markersize=4.5,
            label=LABELS[method],
        )
        axis.fill_between(
            x,
            mean - ci95,
            mean + ci95,
            color=COLORS[method],
            alpha=0.16,
            linewidth=0,
        )
    axis.set_xlabel("Target-gradient percentile")
    axis.set_ylabel("Mean pixel absolute error")
    axis.set_title("(c) Fixed-stress error structure")
    axis.set_xticks([5, 25, 50, 75, 95])
    axis.legend(frameon=False, loc="upper left")
    _format_diagnostic_axis(axis)
    return gradient_curves


def plot_fixed_stress_diagnostics(
    fields: dict[str, torch.Tensor],
    errors: dict[str, torch.Tensor],
    *,
    index: int,
    boost: tuple[float, float],
    out_prefix: Path,
) -> dict[str, Any]:
    """Plot full-test context for the illustrative field selection and OOD artifacts."""
    plt.rcParams.update(
        {
            "font.size": 13.0,
            "axes.titlesize": 14.0,
            "axes.labelsize": 12.5,
            "xtick.labelsize": 11.0,
            "ytick.labelsize": 11.0,
            "legend.fontsize": 10.5,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.65), constrained_layout=True)

    _plot_error_shift(axes[0], errors)
    paired_gain = _plot_selection_context(axes[1], errors, index)
    gradient_curves = _plot_gradient_errors(axes[2], fields)

    fig.suptitle(
        rf"Seed 23, fixed Galilean stress $\delta=({boost[0]:+.2f},{boost[1]:+.2f})$",
        fontsize=15.0,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    return {
        "paired_ood_gain": paired_gain,
        "fixed_stress_relative_l2_means": {
            name: float(values.mean()) for name, values in errors.items()
        },
        "fixed_stress_gradient_binned_mae": gradient_curves,
    }


def plot_equivariance_residuals(
    fields: dict[str, torch.Tensor],
    defects: dict[str, torch.Tensor],
    *,
    index: int,
    boost: tuple[float, float],
    out_prefix: Path,
) -> dict[str, Any]:
    """Plot spatial equivariance residuals for the selected example."""
    aug_residual = np.abs(
        field_array(fields["aug_ood"][index]) - field_array(fields["aug_equiv"][index])
    )
    loco_residual = np.abs(
        field_array(fields["loco_ood"][index]) - field_array(fields["loco_equiv"][index])
    )
    reduction = aug_residual - loco_residual
    residual_limit = max(float(aug_residual.max()), float(loco_residual.max()))
    reduction_limit = float(np.abs(reduction).max())
    fraction_improved = float(np.mean(loco_residual < aug_residual))

    gradient = gradient_magnitude(fields["target_ood"][index]).numpy()
    gradient_threshold = float(np.quantile(gradient, 0.75))
    height, width = gradient.shape
    y = (np.arange(height) + 0.5) / height
    x = (np.arange(width) + 0.5) / width

    plt.rcParams.update(
        {
            "font.size": 15.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.7), constrained_layout=True)
    residual_image = None
    for axis, residual, method, defect in zip(
        axes[:2],
        [aug_residual, loco_residual],
        ["Augmentation", "Aug. + LOCO"],
        [defects["aug"][index], defects["loco"][index]],
    ):
        residual_image = axis.imshow(
            residual,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="magma",
            vmin=0.0,
            vmax=residual_limit,
            interpolation="nearest",
        )
        axis.contour(
            x,
            y,
            gradient,
            levels=[gradient_threshold],
            colors="white",
            linewidths=0.65,
            alpha=0.75,
        )
        axis.set_title(method + "\n" + rf"$D_{{\rm eq}}={float(defect):.3f}$")
    reduction_image = axes[2].imshow(
        reduction,
        origin="lower",
        extent=(0.0, 1.0, 0.0, 1.0),
        cmap="RdBu_r",
        vmin=-reduction_limit,
        vmax=reduction_limit,
        interpolation="nearest",
    )
    axes[2].contour(
        x,
        y,
        gradient,
        levels=[gradient_threshold],
        colors="#222222",
        linewidths=0.65,
        alpha=0.72,
    )
    axes[2].set_title(
        f"Residual gain (red = A+L better)\n{100.0 * fraction_improved:.1f}% pixels improve"
    )
    for column, axis in enumerate(axes):
        axis.set_aspect("equal")
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.set_yticks([0.0, 0.5, 1.0] if column == 0 else [])
        axis.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    assert residual_image is not None
    fig.colorbar(
        residual_image,
        ax=axes[:2].tolist(),
        location="bottom",
        shrink=0.72,
        pad=0.05,
        label=r"$|G(Ta)-T G(a)|$",
    )
    fig.colorbar(
        reduction_image,
        ax=[axes[2]],
        location="bottom",
        shrink=0.82,
        pad=0.05,
        label="residual gain",
    )
    fig.suptitle(
        rf"Fixed-stress equivariance residuals: test index {index}, "
        rf"$\delta=({boost[0]:+.2f},{boost[1]:+.2f})$",
        fontsize=15.5,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    return {
        "selected_relative_defect": {
            "augmentation": float(defects["aug"][index]),
            "augmentation_plus_loco": float(defects["loco"][index]),
        },
        "full_test_fixed_stress_relative_defect_mean": {
            "augmentation": float(defects["aug"].mean()),
            "augmentation_plus_loco": float(defects["loco"].mean()),
        },
        "selected_fraction_loco_lower_absolute_residual": fraction_improved,
        "target_gradient_contour_quantile": 0.75,
        "target_gradient_contour_threshold": gradient_threshold,
        "plot_color_limits": {
            "absolute_equivariance_residual_max": residual_limit,
            "residual_reduction_symmetric_abs_max": reduction_limit,
        },
    }
