#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.symmetry.transforms import NavierStokes2DGalilean, TransformSample
from otno.training.losses import relative_defect_per_sample, relative_l2_per_sample
from otno.utils import file_sha256, get_device


DEFAULT_AUG_CHECKPOINT = (
    "runs/ablations/2d_galilean_n64_label_efficiency_compute_matched/"
    "fraction_0.02/aug_steps_6/seed_23/checkpoints/best.pt"
)
DEFAULT_LOCO_CHECKPOINT = (
    "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
    "fraction_0.02/aug_orbit_lambda_0.1_steps_4/seed_23/checkpoints/best.pt"
)
DEFAULT_FIGURE_CONFIG = "figure_specs/2d_galilean_n64_id_ood_seed23.yaml"


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", checkpoint.get("base_config"))
    if config is None:
        raise KeyError(f"Checkpoint has no config: {checkpoint_path}")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    checkpoint_info = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_val_relative_l2": float(checkpoint.get("best_val", float("nan"))),
        "config_hash": checkpoint.get("meta", {}).get("config_hash"),
    }
    return model, config, checkpoint_info


def _validate_checkpoint_configs(
    aug_config: dict[str, Any], loco_config: dict[str, Any]
) -> dict[str, Any]:
    aug_training = aug_config.get("training", {})
    loco_training = loco_config.get("training", {})
    if str(aug_training.get("method", "")).lower() not in {"aug", "augmentation"}:
        raise ValueError("The augmentation checkpoint is not an augmentation-only run")
    if str(loco_training.get("method", "")).lower() not in {"aug_orbit", "orbit_aug"}:
        raise ValueError("The LOCO checkpoint is not an augmentation-plus-orbit run")
    for key in ["seed", "dataset", "model", "symmetry"]:
        if aug_config.get(key) != loco_config.get(key):
            raise ValueError(f"Checkpoint configs disagree on {key!r}")
    aug_fraction = float(aug_training.get("data_fraction", 1.0))
    loco_fraction = float(loco_training.get("data_fraction", 1.0))
    if not np.isclose(aug_fraction, loco_fraction):
        raise ValueError("Checkpoint configs use different labeled fractions")
    return {
        "seed": int(aug_config.get("seed", -1)),
        "data_fraction": aug_fraction,
        "model_name": str(aug_config.get("model", {}).get("name", "")),
        "augmentation_method": str(aug_training.get("method")),
        "augmentation_epochs": int(aug_training.get("epochs", -1)),
        "augmentation_steps_per_epoch": int(aug_training.get("steps_per_epoch", -1)),
        "augmentation_lambda_aug": float(aug_training.get("lambda_aug", float("nan"))),
        "loco_method": str(loco_training.get("method")),
        "loco_epochs": int(loco_training.get("epochs", -1)),
        "loco_steps_per_epoch": int(loco_training.get("steps_per_epoch", -1)),
        "loco_lambda_aug": float(loco_training.get("lambda_aug", float("nan"))),
        "loco_lambda_orbit": float(loco_training.get("lambda_orbit", float("nan"))),
        "loco_normalize_by_epsilon": bool(
            loco_training.get("normalize_by_epsilon", False)
        ),
        "loco_orbit_eta": float(loco_training.get("orbit_eta", float("nan"))),
    }


def _fixed_sample(
    batch_size: int,
    boost: tuple[float, float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TransformSample:
    boost_tensor = torch.tensor(boost, device=device, dtype=dtype).view(1, 2)
    boost_tensor = boost_tensor.expand(batch_size, 2)
    epsilon = torch.linalg.norm(boost_tensor, dim=-1).clamp_min(1e-6)
    return TransformSample(
        params={"boost": boost_tensor},
        epsilon=epsilon,
        name="navier_stokes2d_galilean",
    )


@torch.no_grad()
def _predict(
    aug_model: torch.nn.Module,
    loco_model: torch.nn.Module,
    dataset,
    transform: NavierStokes2DGalilean,
    *,
    boost: tuple[float, float],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    fields: dict[str, list[torch.Tensor]] = {
        "target_id": [],
        "target_ood": [],
        "aug_id": [],
        "aug_ood": [],
        "aug_equiv": [],
        "loco_id": [],
        "loco_ood": [],
        "loco_equiv": [],
    }
    for start in range(0, len(dataset), batch_size):
        stop = min(len(dataset), start + batch_size)
        a = dataset.a[start:stop].to(device)
        target_id = dataset.u[start:stop].to(device)
        sample = _fixed_sample(
            a.shape[0],
            boost,
            device=device,
            dtype=a.dtype,
        )
        a_ood = transform.apply_input(a, sample)
        target_ood = transform.apply_output(target_id, sample)
        fields["target_id"].append(target_id.cpu())
        fields["target_ood"].append(target_ood.cpu())
        aug_id = aug_model(a)
        aug_ood = aug_model(a_ood)
        loco_id = loco_model(a)
        loco_ood = loco_model(a_ood)
        fields["aug_id"].append(aug_id.cpu())
        fields["aug_ood"].append(aug_ood.cpu())
        fields["aug_equiv"].append(transform.apply_output(aug_id, sample).cpu())
        fields["loco_id"].append(loco_id.cpu())
        fields["loco_ood"].append(loco_ood.cpu())
        fields["loco_equiv"].append(transform.apply_output(loco_id, sample).cpu())
    return {name: torch.cat(parts, dim=0) for name, parts in fields.items()}


def _relative_errors(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "aug_id": relative_l2_per_sample(fields["aug_id"], fields["target_id"]),
        "loco_id": relative_l2_per_sample(fields["loco_id"], fields["target_id"]),
        "aug_ood": relative_l2_per_sample(fields["aug_ood"], fields["target_ood"]),
        "loco_ood": relative_l2_per_sample(fields["loco_ood"], fields["target_ood"]),
    }


def _relative_defects(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "aug": relative_defect_per_sample(fields["aug_ood"], fields["aug_equiv"]),
        "loco": relative_defect_per_sample(fields["loco_ood"], fields["loco_equiv"]),
    }


def _gradient_magnitude(target: torch.Tensor) -> torch.Tensor:
    """Periodic centered-difference magnitude for ``[..., height, width, 1]`` fields."""
    target_2d = target[..., 0]
    grad_x = 0.5 * (
        torch.roll(target_2d, -1, dims=-1) - torch.roll(target_2d, 1, dims=-1)
    )
    grad_y = 0.5 * (
        torch.roll(target_2d, -1, dims=-2) - torch.roll(target_2d, 1, dims=-2)
    )
    return torch.sqrt(grad_x.square() + grad_y.square())


def _gradient_binned_mae(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    bins: int = 10,
) -> dict[str, list[float]]:
    """Aggregate per-example MAE in equal-mass target-gradient bins."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    if target.shape != prediction.shape:
        raise ValueError("target and prediction shapes must match")
    gradient = _gradient_magnitude(target).reshape(target.shape[0], -1)
    absolute_error = (prediction - target).abs()[..., 0].reshape(target.shape[0], -1)
    quantiles = torch.linspace(0.0, 1.0, bins + 1, dtype=gradient.dtype)
    edges = torch.quantile(gradient.reshape(-1), quantiles)
    means: list[float] = []
    ci95: list[float] = []
    for bin_index in range(bins):
        lower = edges[bin_index]
        upper = edges[bin_index + 1]
        if bin_index + 1 == bins:
            mask = (gradient >= lower) & (gradient <= upper)
        else:
            mask = (gradient >= lower) & (gradient < upper)
        counts = mask.sum(dim=1)
        valid = counts > 0
        per_example = (absolute_error * mask).sum(dim=1)[valid] / counts[valid]
        means.append(float(per_example.mean()))
        if per_example.numel() > 1:
            sem = per_example.std(unbiased=True) / np.sqrt(per_example.numel())
            ci95.append(float(1.96 * sem))
        else:
            ci95.append(0.0)
    return {
        "percentile_midpoints": [
            float(value) for value in np.linspace(50.0 / bins, 100.0 - 50.0 / bins, bins)
        ],
        "gradient_edges": [float(value) for value in edges],
        "mean_absolute_error": means,
        "normal_approximation_ci95_half_width": ci95,
    }


def _representative_index(errors: dict[str, torch.Tensor]) -> tuple[int, float]:
    improvement = errors["aug_ood"] - errors["loco_ood"]
    median = float(torch.quantile(improvement, 0.5))
    index = int(torch.argmin((improvement - median).abs()))
    return index, median


def _gradient_error_summary(
    target: torch.Tensor, prediction: torch.Tensor
) -> dict[str, float]:
    error = (prediction - target).abs()[..., 0]
    gradient = _gradient_magnitude(target)
    threshold = torch.quantile(gradient.reshape(-1), 0.75)
    high = gradient >= threshold
    low = ~high
    high_mae = error[high].mean()
    low_mae = error[low].mean()
    return {
        "high_gradient_mae": float(high_mae),
        "low_gradient_mae": float(low_mae),
        "high_to_low_ratio": float(high_mae / low_mae.clamp_min(1e-12)),
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _metadata(
    *,
    fields: dict[str, torch.Tensor],
    errors: dict[str, torch.Tensor],
    index: int,
    median_improvement: float,
    boost: tuple[float, float],
    dataset_path: Path,
    aug_checkpoint: Path,
    loco_checkpoint: Path,
    aug_checkpoint_info: dict[str, Any],
    loco_checkpoint_info: dict[str, Any],
    run_spec: dict[str, Any],
    dataset_metadata: dict[str, Any],
    base_frame: tuple[float, float],
    plot_limits: dict[str, float],
    batch_size: int,
    device: torch.device,
    figure_config_path: Path,
) -> dict[str, Any]:
    selected = {name: float(values[index]) for name, values in errors.items()}
    selected["ood_error_reduction"] = selected["aug_ood"] - selected["loco_ood"]
    selected["id_error_reduction"] = selected["aug_id"] - selected["loco_id"]
    means = {name: float(values.mean()) for name, values in errors.items()}
    means["ood_error_reduction"] = means["aug_ood"] - means["loco_ood"]
    means["id_error_reduction"] = means["aug_id"] - means["loco_id"]
    means["aug_ood_to_id_ratio"] = means["aug_ood"] / means["aug_id"]
    means["loco_ood_to_id_ratio"] = means["loco_ood"] / means["loco_id"]
    improvement = errors["aug_ood"] - errors["loco_ood"]
    selected_error_maps = {
        "aug_id": (fields["aug_id"][index] - fields["target_id"][index]).abs(),
        "loco_id": (fields["loco_id"][index] - fields["target_id"][index]).abs(),
        "aug_ood": (fields["aug_ood"][index] - fields["target_ood"][index]).abs(),
        "loco_ood": (fields["loco_ood"][index] - fields["target_ood"][index]).abs(),
    }
    pixel_summaries = {}
    for regime in ["id", "ood"]:
        aug_error = selected_error_maps[f"aug_{regime}"]
        loco_error = selected_error_maps[f"loco_{regime}"]
        pixel_summaries[regime] = {
            "fraction_loco_lower_absolute_error": float((loco_error < aug_error).float().mean()),
            "aug_mean_absolute_error": float(aug_error.mean()),
            "loco_mean_absolute_error": float(loco_error.mean()),
            "aug_p95_absolute_error": float(torch.quantile(aug_error.reshape(-1), 0.95)),
            "loco_p95_absolute_error": float(torch.quantile(loco_error.reshape(-1), 0.95)),
        }
    return {
        "selection_rule": (
            f"Training seed {run_spec['seed']}; outcome-conditioned selection of the held-out "
            "test example whose fixed-boost OOD relative-L2 improvement is closest to the "
            "median over all test examples. The example is illustrative, not inferential."
        ),
        "selection_is_outcome_conditioned": True,
        "selected_test_index_zero_based": index,
        "num_test_examples": int(fields["target_id"].shape[0]),
        "fixed_relative_boost": [float(boost[0]), float(boost[1])],
        "fixed_relative_boost_norm": float(np.linalg.norm(boost)),
        "base_ambient_velocity": [float(base_frame[0]), float(base_frame[1])],
        "transformed_ambient_velocity": [
            float(base_frame[0] + boost[0]),
            float(base_frame[1] + boost[1]),
        ],
        "median_ood_relative_l2_absolute_reduction": median_improvement,
        "ood_relative_l2_reduction_quantiles": {
            "q10": float(torch.quantile(improvement, 0.10)),
            "q50": float(torch.quantile(improvement, 0.50)),
            "q90": float(torch.quantile(improvement, 0.90)),
        },
        "selected_relative_l2": selected,
        "full_test_fixed_boost_mean_relative_l2": means,
        "selected_pixel_error_summary": pixel_summaries,
        "selected_gradient_error_summary": {
            "aug_id": _gradient_error_summary(
                fields["target_id"][index], fields["aug_id"][index]
            ),
            "loco_id": _gradient_error_summary(
                fields["target_id"][index], fields["loco_id"][index]
            ),
            "aug_ood": _gradient_error_summary(
                fields["target_ood"][index], fields["aug_ood"][index]
            ),
            "loco_ood": _gradient_error_summary(
                fields["target_ood"][index], fields["loco_ood"][index]
            ),
        },
        "dataset_split": "test",
        "dataset_tensor_shapes": {
            "input": list(fields["target_id"].shape[:-1]) + [3],
            "target": list(fields["target_id"].shape),
        },
        "dataset_metadata": dataset_metadata,
        "dataset": _repo_relative(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "augmentation_checkpoint": _repo_relative(aug_checkpoint),
        "augmentation_checkpoint_sha256": file_sha256(aug_checkpoint),
        "augmentation_checkpoint_info": aug_checkpoint_info,
        "loco_checkpoint": _repo_relative(loco_checkpoint),
        "loco_checkpoint_sha256": file_sha256(loco_checkpoint),
        "loco_checkpoint_info": loco_checkpoint_info,
        "validated_run_spec": run_spec,
        "generation_device": str(device),
        "generation_batch_size": int(batch_size),
        "figure_config": _repo_relative(figure_config_path),
        "figure_config_sha256": file_sha256(figure_config_path),
        "plot_script": _repo_relative(Path(__file__)),
        "plot_script_sha256": file_sha256(Path(__file__)),
        "plot_color_limits": plot_limits,
        "target_construction": (
            "The OOD target is the deterministic periodic Galilean evaluation output "
            "action applied to the cached held-out target; solver closure is validated "
            "separately."
        ),
    }


def _array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()[..., 0]


def _plot(
    fields: dict[str, torch.Tensor],
    errors: dict[str, torch.Tensor],
    *,
    index: int,
    boost: tuple[float, float],
    out_prefix: Path,
) -> dict[str, float]:
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
    field_limit = max(float(_array(item).max()) for item in selected_fields)
    field_limit = max(field_limit, max(float(-_array(item).min()) for item in selected_fields))
    absolute_errors = []
    reductions = []
    for _, target, aug, loco, _, _ in rows:
        aug_error = np.abs(_array(aug) - _array(target))
        loco_error = np.abs(_array(loco) - _array(target))
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
        target_array = _array(target)
        aug_array = _array(aug)
        loco_array = _array(loco)
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


def _format_diagnostic_axis(axis: plt.Axes) -> None:
    axis.grid(True, color="#dedede", linewidth=0.7, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _plot_fixed_stress_diagnostics(
    fields: dict[str, torch.Tensor],
    errors: dict[str, torch.Tensor],
    *,
    index: int,
    boost: tuple[float, float],
    out_prefix: Path,
) -> dict[str, Any]:
    """Plot full-test context for the illustrative field selection and OOD artifacts."""
    colors = {"aug": "#4c78a8", "loco": "#f58518"}
    labels = {"aug": "Augmentation", "loco": "Aug. + LOCO"}
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

    # (a) Per-example movement from ID to the same fixed Galilean stress.
    all_values: list[np.ndarray] = []
    for method in ["aug", "loco"]:
        x = errors[f"{method}_id"].numpy()
        y = errors[f"{method}_ood"].numpy()
        all_values.extend([x, y])
        axes[0].scatter(
            x,
            y,
            s=19,
            alpha=0.34,
            color=colors[method],
            edgecolors="none",
            label=labels[method],
        )
        axes[0].scatter(
            [x.mean()],
            [y.mean()],
            s=115,
            marker="*",
            color=colors[method],
            edgecolors="white",
            linewidths=0.8,
            zorder=4,
        )
    lower = max(0.0, min(float(values.min()) for values in all_values) * 0.94)
    upper = max(float(values.max()) for values in all_values) * 1.03
    axes[0].plot([lower, upper], [lower, upper], color="#555555", linestyle=":", linewidth=1.2)
    axes[0].set_xlim(lower, upper)
    axes[0].set_ylim(lower, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"ID relative $L^2$")
    axes[0].set_ylabel(r"Fixed-stress relative $L^2$")
    axes[0].set_title("(a) ID-to-stress error shift")
    axes[0].legend(frameon=False, loc="lower right")
    _format_diagnostic_axis(axes[0])

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
    axes[1].axvspan(0.0, x_upper, color="#e5f2e3", alpha=0.75)
    axes[1].axvline(0.0, color="#555555", linewidth=1.2)
    axes[1].step(ecdf_x, ecdf_y, where="post", color="#6f4e7c", linewidth=2.2)
    axes[1].axvline(median, color="#6f4e7c", linestyle="--", linewidth=1.2, alpha=0.85)
    axes[1].scatter(
        [selected],
        [selected_rank],
        marker="D",
        s=58,
        color="#111111",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    axes[1].annotate(
        f"qualitative case\n({100.0 * selected_rank:.1f}th percentile)",
        xy=(selected, selected_rank),
        xytext=(12, -30),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10.5,
        arrowprops={"arrowstyle": "-", "color": "#222222", "linewidth": 0.9},
    )
    axes[1].set_xlim(x_lower, x_upper)
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel(r"Paired OOD gain, $L^2_{\rm aug}-L^2_{\rm A+L}$")
    axes[1].set_ylabel("Empirical CDF")
    axes[1].set_title("(b) Selection context (256 cases)")
    axes[1].text(
        0.03,
        0.96,
        f"{100.0 * np.mean(improvement > 0):.1f}% favor Aug. + LOCO",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
    )
    _format_diagnostic_axis(axes[1])

    # (c) A target-gradient-conditioned view of the fixed-stress spatial errors.
    gradient_curves = {
        "aug": _gradient_binned_mae(fields["target_ood"], fields["aug_ood"]),
        "loco": _gradient_binned_mae(fields["target_ood"], fields["loco_ood"]),
    }
    for method in ["aug", "loco"]:
        curve = gradient_curves[method]
        x = np.asarray(curve["percentile_midpoints"])
        mean = np.asarray(curve["mean_absolute_error"])
        ci95 = np.asarray(curve["normal_approximation_ci95_half_width"])
        axes[2].plot(
            x,
            mean,
            color=colors[method],
            linewidth=2.2,
            marker="o",
            markersize=4.5,
            label=labels[method],
        )
        axes[2].fill_between(
            x,
            mean - ci95,
            mean + ci95,
            color=colors[method],
            alpha=0.16,
            linewidth=0,
        )
    axes[2].set_xlabel("Target-gradient percentile")
    axes[2].set_ylabel("Mean pixel absolute error")
    axes[2].set_title("(c) Fixed-stress error structure")
    axes[2].set_xticks([5, 25, 50, 75, 95])
    axes[2].legend(frameon=False, loc="upper left")
    _format_diagnostic_axis(axes[2])

    fig.suptitle(
        rf"Seed 23, fixed Galilean stress $\delta=({boost[0]:+.2f},{boost[1]:+.2f})$",
        fontsize=15.0,
    )
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in [".png", ".pdf"]:
        fig.savefig(out_prefix.with_suffix(suffix), bbox_inches="tight")
    plt.close(fig)
    return {
        "paired_ood_gain": {
            "fraction_positive": float(np.mean(improvement > 0)),
            "mean": float(np.mean(improvement)),
            "q10": float(np.quantile(improvement, 0.10)),
            "q50": median,
            "q90": float(np.quantile(improvement, 0.90)),
            "selected_value": selected,
            "selected_empirical_cdf": selected_rank,
        },
        "fixed_stress_relative_l2_means": {
            name: float(values.mean()) for name, values in errors.items()
        },
        "fixed_stress_gradient_binned_mae": gradient_curves,
    }


def _plot_equivariance_residuals(
    fields: dict[str, torch.Tensor],
    defects: dict[str, torch.Tensor],
    *,
    index: int,
    boost: tuple[float, float],
    out_prefix: Path,
) -> dict[str, Any]:
    """Plot spatial equivariance residuals for the selected example."""
    aug_residual = np.abs(_array(fields["aug_ood"][index]) - _array(fields["aug_equiv"][index]))
    loco_residual = np.abs(
        _array(fields["loco_ood"][index]) - _array(fields["loco_equiv"][index])
    )
    reduction = aug_residual - loco_residual
    residual_limit = max(float(aug_residual.max()), float(loco_residual.max()))
    reduction_limit = float(np.abs(reduction).max())
    fraction_improved = float(np.mean(loco_residual < aug_residual))

    gradient = _gradient_magnitude(fields["target_ood"][index]).numpy()
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
        "Residual gain (red = A+L better)\n"
        f"{100.0 * fraction_improved:.1f}% pixels improve"
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


def _figure_output_records(prefixes: list[Path]) -> list[dict[str, Any]]:
    records = []
    for prefix in prefixes:
        for suffix in [".png", ".pdf"]:
            path = prefix.with_suffix(suffix)
            record: dict[str, Any] = {
                "path": _repo_relative(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            if suffix == ".png":
                record["pixel_shape"] = list(plt.imread(path).shape)
            records.append(record)
    return records


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_frozen_figure_inputs(
    figure_config: dict[str, Any],
    *,
    aug_checkpoint: Path,
    loco_checkpoint: Path,
    dataset_path: Path,
    run_spec: dict[str, Any],
) -> None:
    expected_hashes = {
        "augmentation_checkpoint_sha256": file_sha256(aug_checkpoint),
        "loco_checkpoint_sha256": file_sha256(loco_checkpoint),
        "dataset_sha256": file_sha256(dataset_path),
    }
    for key, actual in expected_hashes.items():
        expected = figure_config.get(key)
        if not expected:
            raise ValueError(f"Figure config is missing frozen input hash {key!r}")
        if actual != expected:
            raise ValueError(f"Figure config {key}={expected!r}, but the file hash is {actual!r}")

    expected_run_spec = figure_config.get("expected_run_spec")
    if not isinstance(expected_run_spec, dict) or not expected_run_spec:
        raise ValueError("Figure config must define a non-empty expected_run_spec mapping")
    failures = []
    for key, expected in expected_run_spec.items():
        actual = run_spec.get(key)
        if not _value_matches(actual, expected):
            failures.append(f"{key}: expected {expected!r}, found {actual!r}")
    if failures:
        raise ValueError("Figure checkpoint protocol mismatch: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot representative ID/OOD Navier--Stokes predictions and error heatmaps."
    )
    parser.add_argument("--figure-config", default=DEFAULT_FIGURE_CONFIG)
    parser.add_argument("--aug-checkpoint")
    parser.add_argument("--loco-checkpoint")
    parser.add_argument("--boost-x", type=float)
    parser.add_argument("--boost-y", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--out-prefix",
    )
    parser.add_argument(
        "--metadata-out",
    )
    parser.add_argument("--diagnostics-out-prefix")
    parser.add_argument("--equivariance-out-prefix")
    args = parser.parse_args()

    figure_config_path = ROOT / args.figure_config
    with figure_config_path.open("r", encoding="utf-8") as handle:
        figure_config = yaml.safe_load(handle) or {}
    configured_boost = figure_config.get("fixed_relative_boost", [0.50, 0.00])
    aug_checkpoint_arg = args.aug_checkpoint or figure_config.get(
        "augmentation_checkpoint", DEFAULT_AUG_CHECKPOINT
    )
    loco_checkpoint_arg = args.loco_checkpoint or figure_config.get(
        "loco_checkpoint", DEFAULT_LOCO_CHECKPOINT
    )
    boost = (
        float(args.boost_x if args.boost_x is not None else configured_boost[0]),
        float(args.boost_y if args.boost_y is not None else configured_boost[1]),
    )
    batch_size = int(args.batch_size or figure_config.get("batch_size", 16))
    device_name = args.device or figure_config.get("device", "cpu")
    out_prefix_arg = args.out_prefix or figure_config.get(
        "out_prefix", "runs/figures/2d_galilean_n64_id_ood_fields"
    )
    metadata_out_arg = args.metadata_out or figure_config.get(
        "metadata_out", "runs/paper_tables/2d_galilean_n64_id_ood_fields.json"
    )
    diagnostics_out_arg = args.diagnostics_out_prefix or figure_config.get(
        "diagnostics_out_prefix",
        "runs/figures/2d_galilean_n64_fixed_stress_diagnostics",
    )
    equivariance_out_arg = args.equivariance_out_prefix or figure_config.get(
        "equivariance_out_prefix",
        "runs/figures/2d_galilean_n64_equivariance_residuals",
    )

    device = get_device(device_name)
    aug_checkpoint = ROOT / aug_checkpoint_arg
    loco_checkpoint = ROOT / loco_checkpoint_arg
    aug_model, aug_config, aug_checkpoint_info = _load_model(aug_checkpoint, device)
    loco_model, loco_config, loco_checkpoint_info = _load_model(loco_checkpoint, device)
    run_spec = _validate_checkpoint_configs(aug_config, loco_config)
    aug_dataset_path = ROOT / aug_config["dataset"]["path"]
    loco_dataset_path = ROOT / loco_config["dataset"]["path"]
    if aug_dataset_path.resolve() != loco_dataset_path.resolve():
        raise ValueError("The two checkpoints refer to different datasets")
    _validate_frozen_figure_inputs(
        figure_config,
        aug_checkpoint=aug_checkpoint,
        loco_checkpoint=loco_checkpoint,
        dataset_path=aug_dataset_path,
        run_spec=run_spec,
    )
    dataset, dataset_metadata = load_tensor_dataset(aug_dataset_path, "test")
    symmetry = aug_config["symmetry"]
    transform = NavierStokes2DGalilean(
        max_boost=max(abs(boost[0]), abs(boost[1])),
        final_time=float(symmetry.get("final_time", 0.5)),
        length=float(symmetry.get("length", 1.0)),
        boost_x_channel=int(symmetry.get("boost_x_channel", 1)),
        boost_y_channel=int(symmetry.get("boost_y_channel", 2)),
    )
    fields = _predict(
        aug_model,
        loco_model,
        dataset,
        transform,
        boost=boost,
        batch_size=batch_size,
        device=device,
    )
    errors = _relative_errors(fields)
    defects = _relative_defects(fields)
    index, median_improvement = _representative_index(errors)
    out_prefix = ROOT / out_prefix_arg
    plot_limits = _plot(fields, errors, index=index, boost=boost, out_prefix=out_prefix)
    diagnostics_out_prefix = ROOT / diagnostics_out_arg
    diagnostics = _plot_fixed_stress_diagnostics(
        fields,
        errors,
        index=index,
        boost=boost,
        out_prefix=diagnostics_out_prefix,
    )
    equivariance_out_prefix = ROOT / equivariance_out_arg
    equivariance_diagnostics = _plot_equivariance_residuals(
        fields,
        defects,
        index=index,
        boost=boost,
        out_prefix=equivariance_out_prefix,
    )
    base_frame = (
        float(dataset.a[index, 0, 0, int(symmetry.get("boost_x_channel", 1))]),
        float(dataset.a[index, 0, 0, int(symmetry.get("boost_y_channel", 2))]),
    )

    metadata = _metadata(
        fields=fields,
        errors=errors,
        index=index,
        median_improvement=median_improvement,
        boost=boost,
        dataset_path=aug_dataset_path,
        aug_checkpoint=aug_checkpoint,
        loco_checkpoint=loco_checkpoint,
        aug_checkpoint_info=aug_checkpoint_info,
        loco_checkpoint_info=loco_checkpoint_info,
        run_spec=run_spec,
        dataset_metadata=dataset_metadata,
        base_frame=base_frame,
        plot_limits=plot_limits,
        batch_size=batch_size,
        device=device,
        figure_config_path=figure_config_path,
    )
    metadata["fixed_stress_diagnostics"] = diagnostics
    metadata["equivariance_diagnostics"] = equivariance_diagnostics
    metadata["software_versions"] = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "yaml": yaml.__version__,
    }
    metadata["generated_figure_outputs"] = _figure_output_records(
        [out_prefix, diagnostics_out_prefix, equivariance_out_prefix]
    )
    metadata_path = ROOT / metadata_out_arg
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"selected held-out test index {index}")
    print(f"wrote {out_prefix.with_suffix('.png')}")
    print(f"wrote {out_prefix.with_suffix('.pdf')}")
    print(f"wrote {diagnostics_out_prefix.with_suffix('.png')}")
    print(f"wrote {diagnostics_out_prefix.with_suffix('.pdf')}")
    print(f"wrote {equivariance_out_prefix.with_suffix('.png')}")
    print(f"wrote {equivariance_out_prefix.with_suffix('.pdf')}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
