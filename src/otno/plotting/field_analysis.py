"""Predictions and numerical diagnostics for fixed Galilean stress tests.

Scalar fields have shape ``[batch, height, width, 1]``. Predictions are
collected on the CPU so diagnostic calculations do not depend on plot layout.
"""

from __future__ import annotations

import numpy as np
import torch

from otno.data.datasets import TensorDictDataset
from otno.symmetry.transforms import NavierStokes2DGalilean, TransformSample
from otno.training.losses import relative_defect_per_sample, relative_l2_per_sample


def fixed_galilean_sample(
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
def predict_fields(
    aug_model: torch.nn.Module,
    loco_model: torch.nn.Module,
    dataset: TensorDictDataset,
    transform: NavierStokes2DGalilean,
    *,
    boost: tuple[float, float],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Evaluate ID predictions, stressed inputs, and transformed predictions."""
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
        sample = fixed_galilean_sample(
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


def relative_errors(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "aug_id": relative_l2_per_sample(fields["aug_id"], fields["target_id"]),
        "loco_id": relative_l2_per_sample(fields["loco_id"], fields["target_id"]),
        "aug_ood": relative_l2_per_sample(fields["aug_ood"], fields["target_ood"]),
        "loco_ood": relative_l2_per_sample(fields["loco_ood"], fields["target_ood"]),
    }


def relative_defects(fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "aug": relative_defect_per_sample(fields["aug_ood"], fields["aug_equiv"]),
        "loco": relative_defect_per_sample(fields["loco_ood"], fields["loco_equiv"]),
    }


def gradient_magnitude(target: torch.Tensor) -> torch.Tensor:
    """Periodic centered-difference magnitude for ``[..., height, width, 1]`` fields."""
    target_2d = target[..., 0]
    grad_x = 0.5 * (torch.roll(target_2d, -1, dims=-1) - torch.roll(target_2d, 1, dims=-1))
    grad_y = 0.5 * (torch.roll(target_2d, -1, dims=-2) - torch.roll(target_2d, 1, dims=-2))
    return torch.sqrt(grad_x.square() + grad_y.square())


def gradient_binned_mae(
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
    gradient = gradient_magnitude(target).reshape(target.shape[0], -1)
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


def representative_index(errors: dict[str, torch.Tensor]) -> tuple[int, float]:
    """Select the held-out case closest to the median paired OOD improvement."""
    improvement = errors["aug_ood"] - errors["loco_ood"]
    median = float(torch.quantile(improvement, 0.5))
    index = int(torch.argmin((improvement - median).abs()))
    return index, median


def gradient_error_summary(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    error = (prediction - target).abs()[..., 0]
    gradient = gradient_magnitude(target)
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
