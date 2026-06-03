from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from otno.config import save_config
from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.training.losses import orbit_consistency_loss
from otno.training.metrics import evaluate_model, measure_inference_latency
from otno.utils import (
    append_jsonl,
    dump_json,
    ensure_dir,
    get_device,
    make_torch_generator,
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
            in_last_block = last is not None and (f"spectral.{last}" in name or f"pointwise.{last}" in name)
            if in_projector or in_last_block:
                p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown adaptation.trainable mode: {mode}")
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"No trainable parameters selected for mode: {mode}")
    return params


def _parameter_l2_to_initial(params: list[torch.nn.Parameter], initial: list[torch.Tensor]) -> torch.Tensor:
    total = torch.zeros((), device=params[0].device)
    denom = 0
    for p, p0 in zip(params, initial):
        delta = p - p0
        total = total + delta.abs().pow(2).sum()
        denom += p.numel()
    return total / max(1, denom)


def adapt_from_config(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    deterministic = bool(config.get("runtime", {}).get("deterministic", False))
    set_seed(seed, deterministic=deterministic)
    device = get_device(config.get("runtime", {}).get("device", "auto"))
    run_dir = ensure_dir(config.get("runtime", {}).get("run_dir", "runs/adapt/default"))
    save_config(config, run_dir / "config.yaml")

    adaptation_cfg = config.get("adaptation", {})
    ckpt_path = Path(adaptation_cfg["checkpoint"])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    base_cfg = ckpt["config"]
    model = build_model(base_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    transform = build_transform(base_cfg.get("symmetry"))
    if transform is None:
        raise ValueError("Adaptation requires a symmetry transform in the checkpoint config")

    dataset_path = Path(config.get("dataset", {}).get("path", base_cfg["dataset"]["path"]))
    split = config.get("dataset", {}).get("split", "test")
    dataset, _ = load_tensor_dataset(dataset_path, split)
    batch_size = int(adaptation_cfg.get("batch_size", base_cfg.get("training", {}).get("batch_size", 32)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(adaptation_cfg.get("num_workers", 0)),
        generator=make_torch_generator(seed + 400_000),
        worker_init_fn=seed_worker,
    )
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    before = evaluate_model(
        model,
        eval_loader,
        device=device,
        transform=transform,
        n_orbit_samples=int(adaptation_cfg.get("eval_orbit_samples", 4)),
        seed=int(adaptation_cfg.get("eval_seed", seed + 500_000)),
    )

    params = _select_trainable_parameters(model, str(adaptation_cfg.get("trainable", "projector")))
    initial = [p.detach().clone() for p in params]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(adaptation_cfg.get("lr", 1e-4)),
        weight_decay=float(adaptation_cfg.get("weight_decay", 0.0)),
    )
    epochs = int(adaptation_cfg.get("epochs", 20))
    beta = float(adaptation_cfg.get("beta_l2_initial", 1e-4))
    normalize_by_epsilon = bool(adaptation_cfg.get("normalize_by_epsilon", True))

    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {"loss": 0.0, "orbit_loss": 0.0, "reg": 0.0}
        batches = 0
        stats: dict[str, float] = {}
        for batch in tqdm(loader, desc=f"adapt {epoch}/{epochs}", leave=False):
            a = batch["a"].to(device)
            optimizer.zero_grad(set_to_none=True)
            orb, stats = orbit_consistency_loss(
                model,
                a,
                transform,
                normalize_by_epsilon=normalize_by_epsilon,
                eta=float(adaptation_cfg.get("orbit_eta", 1e-6)),
            )
            reg = _parameter_l2_to_initial(params, initial)
            loss = orb + beta * reg
            loss.backward()
            grad_clip = adaptation_cfg.get("grad_clip", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
            optimizer.step()
            totals["loss"] += float(loss.detach().cpu())
            totals["orbit_loss"] += float(orb.detach().cpu())
            totals["reg"] += float(reg.detach().cpu())
            batches += 1
        record = {
            "epoch": epoch,
            "adapt_loss": totals["loss"] / max(1, batches),
            "adapt_orbit_loss": totals["orbit_loss"] / max(1, batches),
            "adapt_l2_initial": totals["reg"] / max(1, batches),
            **stats,
        }
        append_jsonl(record, run_dir / "adapt_metrics.jsonl")

    after = evaluate_model(
        model,
        eval_loader,
        device=device,
        transform=transform,
        n_orbit_samples=int(adaptation_cfg.get("eval_orbit_samples", 4)),
        seed=int(adaptation_cfg.get("eval_seed", seed + 500_000)),
    )
    first_batch = next(iter(eval_loader))["a"].to(device)
    latency = measure_inference_latency(
        model,
        first_batch,
        repeats=int(adaptation_cfg.get("latency_repeats", 20)),
        warmup=int(adaptation_cfg.get("latency_warmup", 5)),
    )
    results = {
        "checkpoint": str(ckpt_path),
        "trainable": adaptation_cfg.get("trainable", "projector"),
        "before": before,
        "after": after,
        **{f"after_{k}": v for k, v in after.items()},
        **{f"before_{k}": v for k, v in before.items()},
        **latency,
    }
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
    return results
