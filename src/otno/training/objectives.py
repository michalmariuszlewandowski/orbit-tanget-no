"""Resolve training methods and assemble their supervised and symmetry losses.

Each batch computes losses in the same order: supervised, augmentation, finite
orbit consistency, then tangent propagation. Keeping that order explicit also
keeps the symmetry sampler's random stream reproducible across resumed runs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from otno.definitions import METHOD_ALIASES, TANGENT_METHODS, canonical_name
from otno.symmetry.transforms import BaseTransform
from otno.training.losses import (
    augmented_supervised_loss,
    orbit_consistency_loss,
    relative_l2_loss,
    tangent_propagation_loss,
    trajectory_nmse_loss,
)

_AUGMENTED_METHODS = frozenset(
    {
        "aug",
        "aug_orbit",
        "aug_orbit_shuffle",
        "aug_orbit_no_output",
        "aug_tangent",
        "semi_aug_orbit",
    }
)
_ORBIT_METHODS = frozenset(
    {
        "orbit",
        "aug_orbit",
        "aug_orbit_shuffle",
        "aug_orbit_no_output",
        "semi_aug_orbit",
    }
)


@dataclass(frozen=True)
class TrainingObjective:
    """Loss choices for one experiment; a missing weight disables that term.

    A zero weight still computes the term, preserving the method's sampling and
    forward-pass order. ``method`` retains the supplied alias for run metadata.
    """

    method: str
    supervised_loss: Callable[..., torch.Tensor]
    augmentation_weight: float | None
    orbit_weight: float | None
    tangent_weight: float | None
    normalize_by_epsilon: bool
    eta: float
    orbit_control: str


def build_training_objective(training: dict[str, Any]) -> TrainingObjective:
    """Resolve loss names, method aliases, and finite-orbit control variants."""
    method = str(training.get("method", "baseline")).lower()
    canonical_method = canonical_name(method, METHOD_ALIASES, "training.method")
    loss_name = str(training.get("loss", "relative_l2")).lower()
    if loss_name in {"relative_l2", "rel_l2"}:
        supervised_loss = relative_l2_loss
    elif loss_name in {"trajectory_nmse", "lpsda_nmse", "nmse"}:
        supervised_loss = trajectory_nmse_loss
    else:
        raise ValueError(f"Unknown training.loss={loss_name!r}")

    orbit_control = str(training.get("orbit_control", "physical")).lower()
    if canonical_method == "aug_orbit_shuffle" and orbit_control == "physical":
        orbit_control = "shuffle_output"
    if canonical_method == "aug_orbit_no_output" and orbit_control == "physical":
        orbit_control = "no_output_transform"

    lambda_orbit = float(training.get("lambda_orbit", 1.0))
    return TrainingObjective(
        method=method,
        supervised_loss=supervised_loss,
        augmentation_weight=(
            float(training.get("lambda_aug", 1.0))
            if canonical_method in _AUGMENTED_METHODS
            else None
        ),
        orbit_weight=lambda_orbit if canonical_method in _ORBIT_METHODS else None,
        tangent_weight=(
            float(training.get("lambda_tangent", lambda_orbit))
            if canonical_method in TANGENT_METHODS
            else None
        ),
        normalize_by_epsilon=bool(training.get("normalize_by_epsilon", True)),
        eta=float(training.get("orbit_eta", 1e-6)),
        orbit_control=orbit_control,
    )


def compute_batch_loss(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    objective: TrainingObjective,
    *,
    transform: BaseTransform | None,
    orbit_inputs: Callable[[], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute a joint objective without taking an optimizer step.

    For semi-supervised training, ``orbit_inputs`` supplies an independent batch
    of inputs only. It is called after augmentation, matching the original
    minibatch and RNG order; its labels never enter the objective.
    """
    prediction = model(inputs)
    loss = objective.supervised_loss(prediction, targets)
    logs = {"supervised_loss": float(loss.detach().cpu())}

    if objective.augmentation_weight is not None:
        if transform is None:
            raise ValueError("Augmentation method requires a symmetry transform")
        augmentation = augmented_supervised_loss(
            model,
            inputs,
            targets,
            transform,
            loss_fn=objective.supervised_loss,
        )
        loss = loss + objective.augmentation_weight * augmentation
        logs["aug_loss"] = float(augmentation.detach().cpu())

    if objective.orbit_weight is not None:
        if transform is None:
            raise ValueError("Orbit method requires a symmetry transform")
        orbit_batch, orbit_prediction = inputs, prediction
        if orbit_inputs is not None:
            orbit_batch = orbit_inputs()
            orbit_prediction = model(orbit_batch)
        orbit_loss, orbit_stats = orbit_consistency_loss(
            model,
            orbit_batch,
            transform,
            base_pred=orbit_prediction,
            normalize_by_epsilon=objective.normalize_by_epsilon,
            eta=objective.eta,
            target_mode=objective.orbit_control,
        )
        loss = loss + objective.orbit_weight * orbit_loss
        logs.update(orbit_stats)

    if objective.tangent_weight is not None:
        if transform is None:
            raise ValueError("Tangent method requires a symmetry transform")
        tangent_loss, tangent_stats = tangent_propagation_loss(
            model,
            inputs,
            transform,
            normalize_by_epsilon=objective.normalize_by_epsilon,
            eta=objective.eta,
        )
        loss = loss + objective.tangent_weight * tangent_loss
        logs.update(tangent_stats)

    return loss, logs
