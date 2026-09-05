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
from otno.training.losses import relative_l2_per_sample
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
        "augmentation_method": str(aug_training.get("method")),
        "augmentation_steps_per_epoch": int(aug_training.get("steps_per_epoch", -1)),
        "loco_method": str(loco_training.get("method")),
        "loco_steps_per_epoch": int(loco_training.get("steps_per_epoch", -1)),
        "loco_lambda_orbit": float(loco_training.get("lambda_orbit", float("nan"))),
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
        "loco_id": [],
        "loco_ood": [],
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
        fields["aug_id"].append(aug_model(a).cpu())
        fields["aug_ood"].append(aug_model(a_ood).cpu())
        fields["loco_id"].append(loco_model(a).cpu())
        fields["loco_ood"].append(loco_model(a_ood).cpu())
    return {name: torch.cat(parts, dim=0) for name, parts in fields.items()}


def _relative_errors(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "aug_id": relative_l2_per_sample(fields["aug_id"], fields["target_id"]),
        "loco_id": relative_l2_per_sample(fields["loco_id"], fields["target_id"]),
        "aug_ood": relative_l2_per_sample(fields["aug_ood"], fields["target_ood"]),
        "loco_ood": relative_l2_per_sample(fields["loco_ood"], fields["target_ood"]),
    }


def _representative_index(errors: dict[str, torch.Tensor]) -> tuple[int, float]:
    improvement = errors["aug_ood"] - errors["loco_ood"]
    median = float(torch.quantile(improvement, 0.5))
    index = int(torch.argmin((improvement - median).abs()))
    return index, median


def _gradient_error_summary(
    target: torch.Tensor, prediction: torch.Tensor
) -> dict[str, float]:
    target_2d = target[..., 0]
    error = (prediction - target).abs()[..., 0]
    grad_x = 0.5 * (torch.roll(target_2d, -1, dims=-1) - torch.roll(target_2d, 1, dims=-1))
    grad_y = 0.5 * (torch.roll(target_2d, -1, dims=-2) - torch.roll(target_2d, 1, dims=-2))
    gradient = torch.sqrt(grad_x.square() + grad_y.square())
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
            f"Training seed {run_spec['seed']}; held-out test example whose fixed-boost OOD "
            "relative-L2 improvement is closest to the median over all test examples."
        ),
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
            rf"OOD: $\delta=({boost[0]:+.2f},{boost[1]:+.2f})$",
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
            "font.size": 14.0,
            "axes.titlesize": 15.0,
            "axes.labelsize": 14.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "figure.dpi": 180,
        }
    )
    fig, axes = plt.subplots(2, 6, figsize=(12.4, 4.65), constrained_layout=True)
    titles = [
        r"Target $\omega(T)$",
        "Aug.",
        "Aug. + LOCO",
        r"$|e_{\rm aug}|$",
        r"$|e_{\rm LOCO}|$",
        r"$|e_{\rm aug}|-|e_{\rm LOCO}|$",
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
        axes[row_index, 0].set_ylabel(f"{row_label}\n$y$")
        axes[row_index, 3].text(
            0.03,
            0.96,
            rf"rel. $L_2={float(aug_rel):.3f}$",
            transform=axes[row_index, 3].transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.5},
        )
        axes[row_index, 4].text(
            0.03,
            0.96,
            rf"rel. $L_2={float(loco_rel):.3f}$",
            transform=axes[row_index, 4].transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 1.5},
        )

    for row_index in range(2):
        for column in range(6):
            axis = axes[row_index, column]
            axis.set_aspect("equal")
            axis.set_xticks([0.0, 0.5, 1.0] if row_index == 1 else [])
            axis.set_yticks([0.0, 0.5, 1.0] if column == 0 else [])
            if row_index == 1:
                axis.set_xlabel("$x$")
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
        label="positive favors LOCO",
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
    index, median_improvement = _representative_index(errors)
    out_prefix = ROOT / out_prefix_arg
    plot_limits = _plot(fields, errors, index=index, boost=boost, out_prefix=out_prefix)
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
    metadata_path = ROOT / metadata_out_arg
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"selected held-out test index {index}")
    print(f"wrote {out_prefix.with_suffix('.png')}")
    print(f"wrote {out_prefix.with_suffix('.pdf')}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
