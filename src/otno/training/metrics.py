from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.data import DataLoader

from otno.symmetry.transforms import BaseTransform
from otno.training.losses import (
    mean_absolute_per_sample,
    relative_defect_per_sample,
    relative_l2_per_sample,
    trajectory_nmse_per_sample,
)


def _rng_context(seed: int | None, device: torch.device):
    if seed is None:
        return nullcontext()
    devices = [device] if device.type == "cuda" else []
    return torch.random.fork_rng(devices=devices, enabled=True)


def _add_stats(name: str, values: torch.Tensor, sums: dict[str, float], sumsqs: dict[str, float], counts: dict[str, int]) -> None:
    flat = values.detach().reshape(-1).float().cpu()
    sums[name] += float(flat.sum())
    sumsqs[name] += float((flat * flat).sum())
    counts[name] += int(flat.numel())


def _finalize(sums: dict[str, float], sumsqs: dict[str, float], counts: dict[str, int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, count in counts.items():
        if count <= 0:
            continue
        mean = sums[key] / count
        var = max(0.0, sumsqs[key] / count - mean * mean)
        std = var**0.5
        out[key] = mean
        out[f"{key}_std"] = std
        out[f"{key}_stderr"] = std / (count**0.5)
    return out


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    transform: BaseTransform | None = None,
    n_orbit_samples: int = 1,
    max_batches: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate with sample-weighted means, standard deviations, and standard errors."""
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    sumsqs: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    start = time.perf_counter()
    num_samples = 0
    num_batches = 0
    with _rng_context(seed, device):
        if seed is not None:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            a = batch["a"].to(device)
            u = batch["u"].to(device)
            pred = model(a)
            _add_stats("relative_l2", relative_l2_per_sample(pred, u), sums, sumsqs, counts)
            if u.ndim == 3 and u.shape[-1] > 3:
                _add_stats("trajectory_nmse", trajectory_nmse_per_sample(pred, u), sums, sumsqs, counts)
            _add_stats("mae", mean_absolute_per_sample(pred - u), sums, sumsqs, counts)
            if u.ndim == 3 and u.shape[-1] == 3:
                _add_stats("force_mae", mean_absolute_per_sample(pred - u), sums, sumsqs, counts)
            num_samples += a.shape[0]
            num_batches += 1
            if transform is not None and n_orbit_samples > 0:
                for _ in range(n_orbit_samples):
                    sample = transform.sample(a.shape[0], a.device, a.dtype)
                    a_t = transform.apply_input(a, sample)
                    u_t = transform.apply_output(u, sample)
                    pred_t = model(a_t)
                    pred_equiv = transform.apply_output(pred, sample)
                    mask = transform.output_mask(u_t, sample)
                    _add_stats("orbit_ood_relative_l2", relative_l2_per_sample(pred_t, u_t, mask=mask), sums, sumsqs, counts)
                    if u_t.ndim == 3 and u_t.shape[-1] > 3:
                        _add_stats(
                            "orbit_ood_trajectory_nmse",
                            trajectory_nmse_per_sample(pred_t, u_t, mask=mask),
                            sums,
                            sumsqs,
                            counts,
                        )
                    _add_stats("orbit_ood_mae", mean_absolute_per_sample(pred_t - u_t, mask=mask), sums, sumsqs, counts)
                    _add_stats("oracle_canonical_ood_relative_l2", relative_l2_per_sample(pred_equiv, u_t, mask=mask), sums, sumsqs, counts)
                    _add_stats("equivariance_defect_relative", relative_defect_per_sample(pred_t, pred_equiv, mask=mask), sums, sumsqs, counts)
                    if u_t.ndim == 3 and u_t.shape[-1] == 3:
                        _add_stats("orbit_ood_force_mae", mean_absolute_per_sample(pred_t - u_t, mask=mask), sums, sumsqs, counts)
                    _add_stats("epsilon", sample.epsilon, sums, sumsqs, counts)
    elapsed = time.perf_counter() - start
    metrics = _finalize(sums, sumsqs, counts)
    if "epsilon" in metrics:
        metrics["epsilon_mean"] = metrics.pop("epsilon")
        metrics["epsilon_mean_std"] = metrics.pop("epsilon_std")
        metrics["epsilon_mean_stderr"] = metrics.pop("epsilon_stderr")
    metrics.update({"num_samples": num_samples, "num_batches": num_batches, "eval_seconds": elapsed})
    return metrics


@torch.no_grad()
def measure_inference_latency(
    model: torch.nn.Module,
    example: torch.Tensor,
    *,
    repeats: int = 50,
    warmup: int = 10,
) -> dict[str, float]:
    model.eval()
    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))
    for _ in range(warmup):
        _ = model(example)
    if example.is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(example)
    if example.is_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        "latency_repeats": repeats,
        "latency_warmup": warmup,
        "latency_ms_per_batch": 1000.0 * elapsed / repeats,
        "latency_ms_per_sample": 1000.0 * elapsed / (repeats * example.shape[0]),
    }
