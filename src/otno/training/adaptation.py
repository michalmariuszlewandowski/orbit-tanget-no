from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from otno.config import save_config
from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.symmetry.transforms import BaseTransform, TransformSample
from otno.training.losses import orbit_consistency_loss
from otno.training.metrics import evaluate_model, measure_inference_latency
from otno.utils import (
    append_jsonl,
    check_run_directory,
    dump_json,
    get_device,
    make_torch_generator,
    prepare_run_directory,
    seed_worker,
    set_seed,
)


def _last_spectral_block_index(model: torch.nn.Module) -> int | None:
    spectral = getattr(model, "spectral", None)
    if spectral is None or len(spectral) == 0:
        return None
    return len(spectral) - 1


def _select_trainable_parameters(model: torch.nn.Module, mode: str) -> list[torch.nn.Parameter]:
    mode = mode.lower()
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
    elif mode == "projector":
        for name, p in model.named_parameters():
            if name.startswith("fc1") or name.startswith("fc2"):
                p.requires_grad_(True)
    elif mode == "last_block":
        last = _last_spectral_block_index(model)
        for name, p in model.named_parameters():
            in_projector = name.startswith("fc1") or name.startswith("fc2")
            in_last_block = last is not None and (
                f"spectral.{last}" in name or f"pointwise.{last}" in name
            )
            if in_projector or in_last_block:
                p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown adaptation.trainable mode: {mode}")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"No trainable parameters selected for mode: {mode}")
    return params


def _parameter_l2_to_initial(
    params: list[torch.nn.Parameter], initial: list[torch.Tensor]
) -> torch.Tensor:
    total = torch.zeros((), device=params[0].device)
    denom = 0
    for p, p0 in zip(params, initial):
        delta = p - p0
        total = total + delta.abs().pow(2).sum()
        denom += p.numel()
    return total / max(1, denom)


def _prediction_preservation_loss(
    pred: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    diff = pred - reference
    diff_flat = diff.reshape(diff.shape[0], -1)
    ref_flat = reference.reshape(reference.shape[0], -1)
    numer = diff_flat.pow(2).sum(dim=-1)
    denom = ref_flat.pow(2).sum(dim=-1).clamp_min(eps)
    return (numer / denom).mean()


def _apply_target_input_transform(
    a: torch.Tensor,
    transform: BaseTransform | None,
) -> tuple[torch.Tensor, TransformSample | None]:
    if transform is None:
        return a, None
    sample = transform.sample(a.shape[0], a.device, a.dtype)
    return transform.apply_input(a, sample), sample


def _random_orbit_consistency_loss(
    model: torch.nn.Module,
    a: torch.Tensor,
    transform: BaseTransform,
    *,
    normalize_by_epsilon: bool = True,
    eta: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample = transform.sample(a.shape[0], a.device, a.dtype)
    base_pred = model(a)
    a_t = transform.apply_input(a, sample)
    pred_from_transformed_input = model(a_t)
    transformed_pred = transform.apply_output(base_pred, sample)
    if transformed_pred.shape[0] > 1:
        transformed_pred = torch.roll(transformed_pred, shifts=1, dims=0)
    diff = pred_from_transformed_input - transformed_pred.detach()
    per_sample = diff.reshape(diff.shape[0], -1).pow(2).mean(dim=-1)
    if normalize_by_epsilon:
        per_sample = per_sample / (sample.epsilon.pow(2) + eta)
    loss = per_sample.mean()
    return (
        loss,
        {
            "orbit_loss": float(loss.detach().cpu()),
            "orbit_defect_absolute": float(
                diff.reshape(diff.shape[0], -1).norm(dim=-1).mean().detach().cpu()
            ),
            "epsilon_mean": float(sample.epsilon.detach().mean().cpu()),
            "epsilon_max": float(sample.epsilon.detach().max().cpu()),
        },
    )


def _build_adaptation_transform(
    *,
    adaptation_cfg: dict[str, Any],
    key: str,
    default: BaseTransform | None,
) -> BaseTransform | None:
    if key not in adaptation_cfg:
        return default
    cfg = adaptation_cfg.get(key)
    if cfg is None or cfg is False:
        return None
    return build_transform({"symmetry": cfg})


@dataclass
class _AdaptationState:
    """Trainable parameters, frozen anchors, and the configured joint objective."""

    params: list[torch.nn.Parameter]
    initial: list[torch.Tensor]
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[..., tuple[torch.Tensor, dict[str, float]]] | None
    loss_mode: str
    beta: float
    gamma: float
    normalize_by_epsilon: bool
    eta: float


def _build_adaptation_state(
    model: torch.nn.Module, adaptation_cfg: dict[str, Any]
) -> _AdaptationState:
    params = _select_trainable_parameters(model, str(adaptation_cfg.get("trainable", "projector")))
    initial = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(adaptation_cfg.get("lr", 1e-4)),
        weight_decay=float(adaptation_cfg.get("weight_decay", 0.0)),
    )
    beta = float(adaptation_cfg.get("beta_l2_initial", 1e-4))
    gamma = float(adaptation_cfg.get("gamma_prediction_preservation", 0.0))
    normalize_by_epsilon = bool(adaptation_cfg.get("normalize_by_epsilon", True))
    loss_mode = str(adaptation_cfg.get("loss_mode", "orbit")).lower()
    if loss_mode in {"orbit", "orbit_preserve", "orbit_preservation"}:
        loss_fn = orbit_consistency_loss
    elif loss_mode in {"random_orbit", "random_orbit_preserve", "random_orbit_preservation"}:
        loss_fn = _random_orbit_consistency_loss
    elif loss_mode in {"preserve", "preservation", "none"}:
        loss_fn = None
    else:
        raise ValueError(f"Unknown adaptation.loss_mode: {loss_mode}")
    return _AdaptationState(
        params=params,
        initial=initial,
        optimizer=optimizer,
        loss_fn=loss_fn,
        loss_mode=loss_mode,
        beta=beta,
        gamma=gamma,
        normalize_by_epsilon=normalize_by_epsilon,
        eta=float(adaptation_cfg.get("orbit_eta", 1e-6)),
    )


def _adaptation_batch_loss(
    model: torch.nn.Module,
    source_model: torch.nn.Module,
    a: torch.Tensor,
    transform: BaseTransform,
    state: _AdaptationState,
) -> tuple[torch.Tensor, dict[str, float], dict[str, float]]:
    """Orbit + frozen-source prediction preservation + parameter anchoring.

    This calculation consumes inputs only. Labels are reserved for evaluation.
    The random-orbit control alone detaches its shuffled reference prediction.
    """
    if state.loss_fn is not None:
        orb, stats = state.loss_fn(
            model,
            a,
            transform,
            normalize_by_epsilon=state.normalize_by_epsilon,
            eta=state.eta,
        )
    else:
        orb = torch.zeros((), device=a.device)
        stats = {
            "orbit_loss": 0.0,
            "orbit_defect_relative": 0.0,
            "epsilon_mean": 0.0,
            "epsilon_max": 0.0,
        }
    preservation = torch.zeros((), device=a.device)
    if state.gamma > 0:
        pred = model(a)
        with torch.no_grad():
            source_pred = source_model(a)
        preservation = _prediction_preservation_loss(pred, source_pred)
    reg = _parameter_l2_to_initial(state.params, state.initial)
    loss = orb + state.gamma * preservation + state.beta * reg
    values = {
        "loss": float(loss.detach().cpu()),
        "orbit_loss": float(orb.detach().cpu()),
        "preservation_loss": float(preservation.detach().cpu()),
        "reg": float(reg.detach().cpu()),
    }
    return loss, values, stats


def _adaptation_epochs(
    model: torch.nn.Module,
    source_model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    transform: BaseTransform,
    target_input_transform: BaseTransform | None,
    state: _AdaptationState,
    adaptation_cfg: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Optimize fixed epochs and yield records; the caller owns output files."""
    epochs = int(adaptation_cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        model.train()
        totals = dict.fromkeys(("loss", "orbit_loss", "preservation_loss", "reg"), 0.0)
        batches = 0
        stats: dict[str, float] = {}
        for batch in tqdm(loader, desc=f"adapt {epoch}/{epochs}", leave=False):
            a_raw = batch["a"].to(device)
            a, _ = _apply_target_input_transform(a_raw, target_input_transform)
            state.optimizer.zero_grad(set_to_none=True)
            loss, values, stats = _adaptation_batch_loss(model, source_model, a, transform, state)
            loss.backward()
            grad_clip = adaptation_cfg.get("grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(state.params, float(grad_clip))
            state.optimizer.step()
            for key, value in values.items():
                totals[key] += value
            batches += 1
        # Epoch objectives average batches; transform diagnostics describe the last batch.
        yield {
            "epoch": epoch,
            "loss_mode": state.loss_mode,
            "adapt_loss": totals["loss"] / max(1, batches),
            "adapt_orbit_loss": totals["orbit_loss"] / max(1, batches),
            "adapt_prediction_preservation_loss": totals["preservation_loss"] / max(1, batches),
            "adapt_l2_initial": totals["reg"] / max(1, batches),
            **stats,
        }


def _evaluate_adaptation(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    transform: BaseTransform | None,
    adaptation_cfg: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Use identical held-out transformations before and after optimization."""
    return evaluate_model(
        model,
        loader,
        device=device,
        transform=transform,
        n_orbit_samples=int(adaptation_cfg.get("eval_orbit_samples", 4)),
        seed=int(adaptation_cfg.get("eval_seed", seed + 500_000)),
    )


def _summarize_adaptation(
    before: dict[str, Any],
    after: dict[str, Any],
    latency: dict[str, float],
    *,
    ckpt_path: Path,
    adaptation_cfg: dict[str, Any],
    state: _AdaptationState,
) -> dict[str, Any]:
    """Calculate the published deltas without writing artifacts."""
    return {
        "checkpoint": str(ckpt_path),
        "trainable": adaptation_cfg.get("trainable", "projector"),
        "loss_mode": state.loss_mode,
        "adapt_on_target_inputs": bool(adaptation_cfg.get("adapt_on_target_inputs", False)),
        "gamma_prediction_preservation": state.gamma,
        "beta_l2_initial": state.beta,
        "before": before,
        "after": after,
        "id_relative_l2_degradation": after.get("relative_l2", 0.0)
        - before.get("relative_l2", 0.0),
        "id_relative_l2_degradation_pct": 100.0
        * (after.get("relative_l2", 0.0) / max(before.get("relative_l2", 0.0), 1e-12) - 1.0),
        "target_orbit_ood_delta_pct": 100.0
        * (
            after.get("orbit_ood_relative_l2", 0.0)
            / max(before.get("orbit_ood_relative_l2", 0.0), 1e-12)
            - 1.0
        ),
        "target_defect_delta_pct": 100.0
        * (
            after.get("equivariance_defect_relative", 0.0)
            / max(before.get("equivariance_defect_relative", 0.0), 1e-12)
            - 1.0
        ),
        **{f"after_{k}": v for k, v in after.items()},
        **{f"before_{k}": v for k, v in before.items()},
        **latency,
    }


def _write_adaptation_outputs(
    run_dir: Path,
    model: torch.nn.Module,
    base_cfg: dict[str, Any],
    config: dict[str, Any],
    results: dict[str, Any],
) -> None:
    """Keep the JSON report and adapted checkpoint schema together."""
    dump_json(results, run_dir / "adapt_results.json")
    torch.save(
        {
            "model": model.state_dict(),
            "base_config": base_cfg,
            "adapt_config": config,
            "results": results,
        },
        run_dir / "adapted.pt",
    )


def _load_adaptation_models(
    checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module, torch.nn.Module]:
    """Load trainable and frozen-source copies of the same checkpoint weights."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_cfg = ckpt["config"]
    model = build_model(base_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    source_model = build_model(base_cfg).to(device)
    source_model.load_state_dict(ckpt["model"])
    source_model.eval()
    for p in source_model.parameters():
        p.requires_grad_(False)

    return base_cfg, model, source_model


def _build_adaptation_loaders(
    config: dict[str, Any], base_cfg: dict[str, Any], adaptation_cfg: dict[str, Any], seed: int
) -> tuple[DataLoader, DataLoader]:
    """Share samples across shuffled optimization and ordered evaluation loaders."""
    dataset_path = Path(config.get("dataset", {}).get("path", base_cfg["dataset"]["path"]))
    split = config.get("dataset", {}).get("split", "test")
    dataset, _ = load_tensor_dataset(dataset_path, split)
    batch_size = int(
        adaptation_cfg.get("batch_size", base_cfg.get("training", {}).get("batch_size", 32))
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(adaptation_cfg.get("num_workers", 0)),
        generator=make_torch_generator(seed + 400_000),
        worker_init_fn=seed_worker,
    )
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return loader, eval_loader


def adapt_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Evaluate, optimize unlabeled inputs, re-evaluate, and save adapted artifacts."""
    runtime_cfg = config.get("runtime", {})
    if runtime_cfg.get("resume", False):
        raise ValueError("Adaptation does not support runtime.resume; start a new adaptation run.")
    seed = int(config.get("seed", 0))
    deterministic = bool(config.get("runtime", {}).get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    device = get_device(config.get("runtime", {}).get("device", "auto"))
    output_files = ("config.yaml", "adapt_metrics.jsonl", "adapt_results.json", "adapted.pt")
    run_dir = check_run_directory(
        runtime_cfg.get("run_dir", "runs/adapt/default"),
        output_files=output_files,
        overwrite=bool(runtime_cfg.get("overwrite", True)),
    )

    adaptation_cfg = config.get("adaptation", {})
    ckpt_path = Path(adaptation_cfg["checkpoint"])
    base_cfg, model, source_model = _load_adaptation_models(ckpt_path, device)

    base_transform = build_transform(base_cfg.get("symmetry"))
    if base_transform is None:
        raise ValueError("Adaptation requires a symmetry transform in the checkpoint config")
    transform = _build_adaptation_transform(
        adaptation_cfg=adaptation_cfg,
        key="orbit_symmetry",
        default=base_transform,
    )
    if transform is None:
        raise ValueError("Adaptation requires an orbit transform")
    eval_transform = _build_adaptation_transform(
        adaptation_cfg=adaptation_cfg,
        key="eval_symmetry",
        default=transform,
    )
    target_input_transform = _build_adaptation_transform(
        adaptation_cfg=adaptation_cfg,
        key="target_input_symmetry",
        default=eval_transform
        if bool(adaptation_cfg.get("adapt_on_target_inputs", False))
        else None,
    )

    loader, eval_loader = _build_adaptation_loaders(config, base_cfg, adaptation_cfg, seed)

    before = _evaluate_adaptation(
        model,
        eval_loader,
        device=device,
        transform=eval_transform,
        adaptation_cfg=adaptation_cfg,
        seed=seed,
    )

    state = _build_adaptation_state(model, adaptation_cfg)
    prepare_run_directory(
        run_dir,
        output_files=output_files,
        overwrite=bool(runtime_cfg.get("overwrite", True)),
    )
    save_config(config, run_dir / "config.yaml")

    for record in _adaptation_epochs(
        model,
        source_model,
        loader,
        device=device,
        transform=transform,
        target_input_transform=target_input_transform,
        state=state,
        adaptation_cfg=adaptation_cfg,
    ):
        append_jsonl(record, run_dir / "adapt_metrics.jsonl")

    after = _evaluate_adaptation(
        model,
        eval_loader,
        device=device,
        transform=eval_transform,
        adaptation_cfg=adaptation_cfg,
        seed=seed,
    )
    first_batch = next(iter(eval_loader))["a"].to(device)
    latency = measure_inference_latency(
        model,
        first_batch,
        repeats=int(adaptation_cfg.get("latency_repeats", 20)),
        warmup=int(adaptation_cfg.get("latency_warmup", 5)),
    )
    results = _summarize_adaptation(
        before, after, latency, ckpt_path=ckpt_path, adaptation_cfg=adaptation_cfg, state=state
    )
    _write_adaptation_outputs(run_dir, model, base_cfg, config, results)
    return results
