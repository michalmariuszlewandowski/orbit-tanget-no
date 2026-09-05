from __future__ import annotations

import torch

from otno.symmetry.transforms import BaseTransform, TransformSample


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def _broadcast_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    return torch.broadcast_to(mask.to(device=x.device, dtype=x.dtype), x.shape)


def relative_l2_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return per-example relative L2 errors."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")
    diff = pred - target
    ref = target
    if mask is not None:
        m = _broadcast_mask(mask, diff)
        diff = diff * m
        ref = ref * m
    diff_flat = _flatten_per_sample(diff)
    denom = _flatten_per_sample(ref).norm(dim=-1).clamp_min(eps)
    return diff_flat.norm(dim=-1) / denom


def relative_l2_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return relative_l2_per_sample(pred, target, mask=mask, eps=eps).mean()


def trajectory_nmse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """LPSDA-style average over time of spatial normalized MSE."""
    if pred.shape != target.shape:
        raise ValueError(f"pred and target shapes differ: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.ndim != 3:
        return relative_l2_per_sample(pred, target, mask=mask, eps=eps).pow(2)
    diff = pred - target
    ref = target
    if mask is not None:
        m = _broadcast_mask(mask, diff)
        diff = diff * m
        ref = ref * m
    diff_sq = diff.pow(2).sum(dim=1)
    ref_sq = ref.pow(2).sum(dim=1).clamp_min(eps)
    return (diff_sq / ref_sq).mean(dim=-1)


def trajectory_nmse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return trajectory_nmse_per_sample(pred, target, mask=mask, eps=eps).mean()


def relative_defect_per_sample(
    pred_t: torch.Tensor,
    pred_equiv: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if pred_t.shape != pred_equiv.shape:
        raise ValueError(
            f"pred_t and pred_equiv shapes differ: {tuple(pred_t.shape)} vs {tuple(pred_equiv.shape)}"
        )
    diff = pred_t - pred_equiv
    ref = pred_equiv
    if mask is not None:
        m = _broadcast_mask(mask, diff)
        diff = diff * m
        ref = ref * m
    return _flatten_per_sample(diff).norm(dim=-1) / _flatten_per_sample(ref).norm(dim=-1).clamp_min(eps)


def mean_squared_per_sample(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Return a per-example mean square, using the valid mask as denominator when supplied."""
    if mask is not None:
        m = _broadcast_mask(mask, x)
        masked = x * m
        denom = m.reshape(m.shape[0], -1).sum(dim=-1).clamp_min(1.0)
        return masked.reshape(masked.shape[0], -1).pow(2).sum(dim=-1) / denom
    flat = _flatten_per_sample(x)
    return flat.pow(2).mean(dim=-1)


def mean_absolute_per_sample(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is not None:
        m = _broadcast_mask(mask, x)
        masked = x.abs() * m
        denom = m.reshape(m.shape[0], -1).sum(dim=-1).clamp_min(1.0)
        return masked.reshape(masked.shape[0], -1).sum(dim=-1) / denom
    return _flatten_per_sample(x.abs()).mean(dim=-1)


def orbit_consistency_loss(
    model: torch.nn.Module,
    a: torch.Tensor,
    transform: BaseTransform,
    *,
    sample: TransformSample | None = None,
    base_pred: torch.Tensor | None = None,
    normalize_by_epsilon: bool = True,
    eta: float = 1e-6,
    target_mode: str = "physical",
) -> tuple[torch.Tensor, dict[str, float]]:
    if sample is None:
        sample = transform.sample(a.shape[0], a.device, a.dtype)
    if base_pred is None:
        base_pred = model(a)
    a_t = transform.apply_input(a, sample)
    pred_from_transformed_input = model(a_t)
    if target_mode in {"no_output", "no_output_transform", "input_only", "identity_output"}:
        transformed_pred = base_pred
    else:
        transformed_pred = transform.apply_output(base_pred, sample)
    mask = transform.output_mask(pred_from_transformed_input, sample)
    if target_mode in {"shuffle", "shuffled", "shuffle_output", "shuffled_output"}:
        transformed_pred = torch.roll(transformed_pred, shifts=1, dims=0)
        if mask is not None:
            mask = torch.roll(mask, shifts=1, dims=0)
    elif target_mode in {"physical", "correct", "no_output", "no_output_transform", "input_only", "identity_output"}:
        pass
    else:
        raise ValueError(f"Unknown orbit consistency target_mode={target_mode!r}")
    per_sample = mean_squared_per_sample(pred_from_transformed_input - transformed_pred, mask=mask)
    if normalize_by_epsilon:
        per_sample = per_sample / (sample.epsilon.pow(2) + eta)
    loss = per_sample.mean()
    defect = relative_defect_per_sample(pred_from_transformed_input, transformed_pred, mask=mask)
    stats = {
        "orbit_loss": float(loss.detach().cpu()),
        "orbit_defect_relative": float(defect.detach().mean().cpu()),
        "epsilon_mean": float(sample.epsilon.detach().mean().cpu()),
        "epsilon_max": float(sample.epsilon.detach().max().cpu()),
    }
    return loss, stats


def tangent_propagation_loss(
    model: torch.nn.Module,
    a: torch.Tensor,
    transform: BaseTransform,
    *,
    sample: TransformSample | None = None,
    normalize_by_epsilon: bool = True,
    eta: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize the infinitesimal equivariance defect with an input JVP."""
    if sample is None:
        sample = transform.sample_tangent(a.shape[0], a.device, a.dtype)
    input_tangent = transform.input_tangent(a, sample)
    base_pred, pred_tangent = torch.autograd.functional.jvp(
        lambda x: model(x),
        (a,),
        (input_tangent,),
        create_graph=True,
        strict=False,
    )
    target_tangent = transform.output_tangent(base_pred, sample)
    mask = transform.output_mask(base_pred, sample)
    per_sample = mean_squared_per_sample(pred_tangent - target_tangent, mask=mask)
    if normalize_by_epsilon:
        per_sample = per_sample / (sample.epsilon.pow(2) + eta)
    loss = per_sample.mean()
    defect = relative_defect_per_sample(pred_tangent, target_tangent, mask=mask)
    stats = {
        "tangent_loss": float(loss.detach().cpu()),
        "tangent_defect_relative": float(defect.detach().mean().cpu()),
        "tangent_epsilon_mean": float(sample.epsilon.detach().mean().cpu()),
        "tangent_epsilon_max": float(sample.epsilon.detach().max().cpu()),
    }
    return loss, stats


def augmented_supervised_loss(
    model: torch.nn.Module,
    a: torch.Tensor,
    u: torch.Tensor,
    transform: BaseTransform,
    *,
    sample: TransformSample | None = None,
    loss_fn=relative_l2_loss,
) -> torch.Tensor:
    if sample is None:
        sample = transform.sample(a.shape[0], a.device, a.dtype)
    a_t = transform.apply_input(a, sample)
    u_t = transform.apply_output(u, sample)
    mask = transform.output_mask(u_t, sample)
    return loss_fn(model(a_t), u_t, mask=mask)
