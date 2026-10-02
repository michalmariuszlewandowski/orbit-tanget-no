"""Render scalar-field predictions and their absolute errors."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def field_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()[..., 0]


def plot_prediction_fields(
    fields: dict[str, torch.Tensor],
    errors: dict[str, torch.Tensor],
    *,
    index: int,
    boost: tuple[float, float],
    out_prefix: Path,
) -> dict[str, float]:
    """Save paired ID/stress heatmaps with shared color scales as PNG and PDF."""
    rows = [
        (
            "ID",
            fields["target_id"][index],
            fields["aug_id"][index],
            fields["loco_id"][index],
            errors["aug_id"][index],
            errors["loco_id"][index],
        ),
        (
            rf"Fixed stress" "\n" rf"$\delta=({boost[0]:+.2f},{boost[1]:+.2f})$",
            fields["target_ood"][index],
            fields["aug_ood"][index],
            fields["loco_ood"][index],
            errors["aug_ood"][index],
            errors["loco_ood"][index],
        ),
    ]
    selected_fields = [item for row in rows for item in row[1:4]]
    field_limit = max(float(field_array(item).max()) for item in selected_fields)
    field_limit = max(field_limit, max(float(-field_array(item).min()) for item in selected_fields))
    absolute_errors = []
    reductions = []
    for _, target, aug, loco, _, _ in rows:
        aug_error = np.abs(field_array(aug) - field_array(target))
        loco_error = np.abs(field_array(loco) - field_array(target))
        absolute_errors.extend([aug_error, loco_error])
        reductions.append(aug_error - loco_error)
    error_limit = max(float(item.max()) for item in absolute_errors)
    reduction_limit = max(float(np.abs(item).max()) for item in reductions)

    plt.rcParams.update(
        {
            "font.size": 18.0,
            "axes.titlesize": 18.0,
            "axes.labelsize": 17.0,
            "xtick.labelsize": 15.0,
            "ytick.labelsize": 15.0,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 6, figsize=(13.6, 5.4), constrained_layout=True)
    titles = [
        r"Target $\omega(T)$",
        "Augmentation",
        "Aug. + LOCO",
        r"$|e_{\rm aug}|$",
        r"$|e_{\rm A+L}|$",
        "Pixelwise gain\n(red = A+L better)",
    ]
    for column, title in enumerate(titles):
        axes[0, column].set_title(title)

    field_image = None
    error_image = None
    reduction_image = None
    for row_index, (row_label, target, aug, loco, aug_rel, loco_rel) in enumerate(rows):
        target_array = field_array(target)
        aug_array = field_array(aug)
        loco_array = field_array(loco)
        aug_error = np.abs(aug_array - target_array)
        loco_error = np.abs(loco_array - target_array)
        reduction = aug_error - loco_error
        for column, field in enumerate([target_array, aug_array, loco_array]):
            field_image = axes[row_index, column].imshow(
                field,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="RdBu_r",
                vmin=-field_limit,
                vmax=field_limit,
                interpolation="nearest",
            )
        for column, error in zip([3, 4], [aug_error, loco_error]):
            error_image = axes[row_index, column].imshow(
                error,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                cmap="magma",
                vmin=0.0,
                vmax=error_limit,
                interpolation="nearest",
            )
        reduction_image = axes[row_index, 5].imshow(
            reduction,
            origin="lower",
            extent=(0.0, 1.0, 0.0, 1.0),
            cmap="RdBu_r",
            vmin=-reduction_limit,
            vmax=reduction_limit,
            interpolation="nearest",
        )
        axes[row_index, 0].set_ylabel(
            row_label,
            rotation=0,
            ha="right",
            va="center",
            labelpad=20,
            fontweight="semibold",
        )
        axes[row_index, 1].set_xlabel(rf"$L^2_{{\rm rel}}={float(aug_rel):.3f}$")
        axes[row_index, 2].set_xlabel(rf"$L^2_{{\rm rel}}={float(loco_rel):.3f}$")
        aug_error_tensor = torch.as_tensor(aug_error)
        loco_error_tensor = torch.as_tensor(loco_error)
        fraction_improved = float((loco_error_tensor < aug_error_tensor).float().mean())
        axes[row_index, 5].set_xlabel(f"{100.0 * fraction_improved:.1f}% pixels improve")

    for row_index in range(2):
        for column in range(6):
            axis = axes[row_index, column]
            axis.set_aspect("equal")
            axis.set_xticks([0.0, 0.5, 1.0] if row_index == 1 else [])
            axis.set_yticks([0.0, 0.5, 1.0] if column == 0 else [])
    assert field_image is not None and error_image is not None and reduction_image is not None
    fig.colorbar(
        field_image,
        ax=axes[:, :3].ravel().tolist(),
        location="bottom",
        shrink=0.68,
        pad=0.04,
        label="vorticity",
    )
    fig.colorbar(
        error_image,
        ax=axes[:, 3:5].ravel().tolist(),
        location="bottom",
        shrink=0.72,
        pad=0.04,
        label="absolute error",
    )
    fig.colorbar(
        reduction_image,
        ax=axes[:, 5].ravel().tolist(),
        location="bottom",
        shrink=0.88,
        pad=0.04,
        label="absolute-error gain",
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    return {
        "vorticity_symmetric_abs_max": field_limit,
        "absolute_error_max": error_limit,
        "pointwise_reduction_symmetric_abs_max": reduction_limit,
    }
