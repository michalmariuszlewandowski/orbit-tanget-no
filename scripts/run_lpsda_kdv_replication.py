#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import torch
from scipy.fftpack import diff as psdiff
from scipy.integrate import solve_ivp
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from otno.config import load_config, parse_overrides, recursive_update, save_config
from otno.utils import dump_json, ensure_dir, file_sha256, get_device, set_seed, stable_json_hash


@dataclass(frozen=True)
class KdVSpec:
    nx: int = 256
    nt: int = 250
    nt_effective: int = 140
    end_time: float = 100.0
    length: float = 128.0
    tol: float = 1e-9
    waves: int = 10
    lmin: int = 1
    lmax: int = 3


def _initial_condition(
    rng: np.random.Generator, x: np.ndarray, spec: KdVSpec, length: float
) -> np.ndarray:
    amplitudes = rng.random((1, spec.waves)) - 0.5
    phases = 2.0 * np.pi * rng.random((1, spec.waves))
    frequencies = rng.integers(spec.lmin, spec.lmax, (1, spec.waves))
    values = amplitudes * np.sin(2.0 * np.pi * frequencies * x[:, None] / length + phases)
    return np.sum(values, axis=-1)


def _kdv_rhs(_: float, u: np.ndarray, length: float) -> np.ndarray:
    ux = psdiff(u, period=length)
    uxxx = psdiff(u, order=3, period=length)
    return -u * ux - uxxx


def _generate_split(num_samples: int, spec: KdVSpec, seed: int, split: str) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    trajectories = np.empty((num_samples, spec.nt_effective, spec.nx), dtype=np.float32)
    dx_values = np.empty((num_samples,), dtype=np.float32)
    dt_values = np.empty((num_samples,), dtype=np.float32)
    length_values = np.empty((num_samples,), dtype=np.float32)
    end_time_values = np.empty((num_samples,), dtype=np.float32)
    for idx in range(num_samples):
        t_end = spec.end_time * (1.1 - 0.2 * rng.random())
        length = spec.length * (1.1 - 0.2 * rng.random())
        x = np.linspace(0.0, (1.0 - 1.0 / spec.nx) * length, spec.nx)
        t = np.linspace(0.0, t_end, spec.nt)
        u0 = _initial_condition(rng, x, spec, length)
        solved = solve_ivp(
            fun=_kdv_rhs,
            t_span=(t[0], t[-1]),
            y0=u0,
            method="Radau",
            t_eval=t,
            args=(length,),
            atol=spec.tol,
            rtol=spec.tol,
        )
        if not solved.success:
            raise RuntimeError(f"LPSDA KdV Radau solve failed for {split}[{idx}]: {solved.message}")
        trajectories[idx] = solved.y.T[-spec.nt_effective :].astype(np.float32)
        dx_values[idx] = length / spec.nx
        dt_values[idx] = t_end / (spec.nt - 1)
        length_values[idx] = length
        end_time_values[idx] = t_end
        print(f"generated {split} {idx + 1}/{num_samples}", flush=True)
    return {
        "u": torch.from_numpy(trajectories),
        "dx": torch.from_numpy(dx_values),
        "dt": torch.from_numpy(dt_values),
        "length": torch.from_numpy(length_values),
        "end_time": torch.from_numpy(end_time_values),
    }


def generate_dataset(config: dict[str, Any], path: Path) -> None:
    dataset_cfg = config["dataset"]
    spec = KdVSpec(
        nx=int(dataset_cfg.get("nx", 256)),
        nt=int(dataset_cfg.get("nt", 250)),
        nt_effective=int(dataset_cfg.get("nt_effective", 140)),
        end_time=float(dataset_cfg.get("end_time", 100.0)),
        length=float(dataset_cfg.get("length", 128.0)),
        tol=float(dataset_cfg.get("tol", 1e-9)),
    )
    base_seed = int(config.get("seed", 0))
    split_counts = {
        "train": int(dataset_cfg.get("train_samples", 64)),
        "valid": int(dataset_cfg.get("valid_samples", 512)),
        "test": int(dataset_cfg.get("test_samples", 512)),
    }
    payload: dict[str, Any] = {
        "metadata": {
            "kind": "lpsda_kdv_faithful",
            "source": "independent implementation matching brandstetter-johannes/LPSDA",
            "config_hash": stable_json_hash(config)[:12],
            "spec": spec.__dict__,
            "split_counts": split_counts,
        },
        "splits": {},
    }
    offsets = {"train": 0, "valid": 100_000, "test": 200_000}
    for split, count in split_counts.items():
        payload["splits"][split] = _generate_split(count, spec, base_seed + offsets[split], split)
    ensure_dir(path.parent)
    torch.save(payload, path)


def fourier_shift(u: torch.Tensor, eps: torch.Tensor | float, *, dim: int = -1) -> torch.Tensor:
    n = u.shape[dim]
    u_hat = torch.fft.rfft(u, dim=dim, norm="ortho")
    omega = torch.arange(n // 2 + 1, device=u.device, dtype=u.dtype)
    if n % 2 == 0:
        omega[-1] = 0
    eps_t = torch.as_tensor(eps, device=u.device, dtype=u.dtype)
    phase = torch.exp(-2.0j * math.pi * eps_t * omega)
    while phase.ndim < u_hat.ndim:
        phase = phase.unsqueeze(0)
    return torch.fft.irfft(phase * u_hat, n=n, dim=dim, norm="ortho")


def _fourier_shift_per_channel(
    u: torch.Tensor,
    shift: torch.Tensor,
    length: torch.Tensor,
) -> torch.Tensor:
    """Return ``u_b,c(x - shift_b,c)`` for ``u`` with shape ``[B, X, C]``."""
    if u.ndim != 3:
        raise ValueError(f"Expected [batch, x, channels], got {tuple(u.shape)}")
    batch, nx, channels = u.shape
    shift_t = torch.as_tensor(shift, device=u.device, dtype=u.dtype)
    length_t = torch.as_tensor(length, device=u.device, dtype=u.dtype)
    if shift_t.shape != (batch, channels):
        raise ValueError(f"Expected shift [batch, channels], got {tuple(shift_t.shape)}")
    if length_t.shape != (batch,):
        raise ValueError(f"Expected length [batch], got {tuple(length_t.shape)}")
    u_hat = torch.fft.rfft(u.permute(0, 2, 1), dim=-1, norm="ortho")
    modes = torch.arange(nx // 2 + 1, device=u.device, dtype=u.dtype)
    phase = torch.exp(
        -2.0j
        * math.pi
        * (shift_t / length_t[:, None]).unsqueeze(-1)
        * modes.view(1, 1, -1)
    )
    shifted = torch.fft.irfft(phase * u_hat, n=nx, dim=-1, norm="ortho")
    return shifted.permute(0, 2, 1).contiguous()


def _sample_kdv_boost(
    batch_size: int,
    *,
    max_boost: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    boost = (2.0 * torch.rand(batch_size, device=device, dtype=dtype) - 1.0) * float(max_boost)
    epsilon = boost.abs().clamp_min(1e-6)
    return boost, epsilon


def _apply_kdv_galilean_window(
    u: torch.Tensor,
    *,
    boost: torch.Tensor,
    dx: torch.Tensor,
    dt: torch.Tensor,
    offset_steps: int,
) -> torch.Tensor:
    """Apply the KdV Galilean action to one input/output time window.

    The faithful LPSDA data has sample-dependent domain length and timestep. We
    therefore use ``length = nx * dx`` and physical times ``dt * step`` instead
    of the fixed pilot transform.
    """
    if u.ndim != 3:
        raise ValueError(f"Expected [batch, x, channels], got {tuple(u.shape)}")
    batch, nx, channels = u.shape
    boost = boost.to(device=u.device, dtype=u.dtype)
    dx = dx.to(device=u.device, dtype=u.dtype)
    dt = dt.to(device=u.device, dtype=u.dtype)
    if boost.shape != (batch,):
        raise ValueError(f"Expected boost [batch], got {tuple(boost.shape)}")
    steps = torch.arange(offset_steps, offset_steps + channels, device=u.device, dtype=u.dtype)
    times = dt[:, None] * steps[None, :]
    shift = boost[:, None] * times
    length = dx * nx
    return _fourier_shift_per_channel(u, shift, length) + boost.view(-1, 1, 1)


class LPSDAKdvDataset(Dataset):
    def __init__(self, path: Path, split: str, *, augmentation: dict[str, Any] | None = None):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.data = payload["splits"][split]
        self.split = split
        self.augmentation = augmentation if split == "train" else None

    def __len__(self) -> int:
        return int(self.data["u"].shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.data["u"][idx].clone()
        dx = self.data["dx"][idx].clone()
        dt = self.data["dt"][idx].clone()
        if self.augmentation is not None:
            u, dx, dt = self._augment(u, dx, dt)
        return u.float(), dx.float(), dt.float()

    def _augment(
        self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.augmentation is not None
        max_x_shift = float(self.augmentation.get("max_x_shift", 0.0))
        max_velocity = float(self.augmentation.get("max_velocity", 0.0))
        max_scale = float(self.augmentation.get("max_scale", 0.0))
        if max_x_shift > 0.0:
            eps = max_x_shift * (torch.rand(()) - 0.5)
            u = fourier_shift(u, eps=eps, dim=-1)
        if max_velocity > 0.0:
            eps = 2.0 * max_velocity * (torch.rand(()) - 0.5)
            times = dt * torch.arange(u.shape[0], dtype=u.dtype)
            length = dx * u.shape[-1]
            shifts = -(eps * times) / length
            u = fourier_shift(u, eps=shifts[:, None], dim=-1) - eps
        if max_scale > 0.0:
            eps = max_scale * (torch.rand(()) - 0.5)
            u = torch.exp(2.0 * eps) * u
            dx = torch.exp(-eps) * dx
            dt = torch.exp(-3.0 * eps) * dt
        return u, dx, dt


class SpectralConv1dLPSDA(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.modes = int(modes)
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x_ft = torch.fft.rfft(x)
        out_ft = torch.zeros(
            batch,
            self.weights.shape[1],
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=x.size(-1))


class LPSDAFNO1d(nn.Module):
    def __init__(
        self,
        *,
        nx: int,
        time_history: int = 20,
        time_future: int = 20,
        modes: int = 32,
        width: int = 256,
        num_layers: int = 5,
    ):
        super().__init__()
        self.nx = int(nx)
        self.time_history = int(time_history)
        self.time_future = int(time_future)
        self.fc0 = nn.Linear(self.time_history + 2, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, self.time_future)
        self.fourier_layers = nn.ModuleList(
            [SpectralConv1dLPSDA(width, width, modes) for _ in range(num_layers)]
        )
        self.conv_layers = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(num_layers)])

    def forward(self, u: torch.Tensor, dx: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # u is [batch, x, time_history].
        dx_channel = dx[:, None, None].to(u.device).repeat(1, self.nx, 1)
        dt_channel = dt[:, None, None].to(u.device).repeat(1, self.nx, 1)
        x = torch.cat((u, dx_channel, dt_channel), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for fourier, conv in zip(self.fourier_layers, self.conv_layers):
            x = F.gelu(fourier(x) + conv(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def _create_window(
    trajectories: torch.Tensor,
    start_times: list[int],
    time_history: int,
    time_future: int,
    pf_steps: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = []
    labels = []
    for trajectory, start in zip(trajectories, start_times):
        end = start + time_history
        target_start = end + time_future * pf_steps
        target_end = target_start + time_future
        data.append(trajectory[start:end].unsqueeze(0))
        labels.append(trajectory[target_start:target_end].unsqueeze(0))
    return torch.cat(data, dim=0), torch.cat(labels, dim=0)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    batch_size: int,
    nx: int,
    nt_effective: int,
    time_history: int,
    time_future: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    criterion = nn.MSELoss(reduction="none")
    losses = []
    nlosses = []
    max_start_time = nt_effective - time_history - time_future
    rollout_time_steps = 0
    with torch.no_grad():
        for trajectories, dx, dt in loader:
            trajectories = trajectories.to(device)
            dx = dx.to(device)
            dt = dt.to(device)
            batch_losses = []
            batch_nlosses = []
            data = None
            pred = None
            batch_rollout_time_steps = 0
            for start in range(time_history, max_start_time + 1, time_future):
                end = start + time_history
                target_end = end + time_future
                if start == time_history:
                    data = trajectories[:, start:end].permute(0, 2, 1)
                else:
                    assert data is not None and pred is not None
                    data = torch.cat([data, pred], dim=-1)[..., -time_history:]
                labels = trajectories[:, end:target_end]
                pred = model(data, dx, dt)
                loss = criterion(pred.permute(0, 2, 1), labels)
                nlabels = torch.mean(labels**2, dim=-1, keepdim=True)
                nloss = loss / nlabels
                batch_losses.append(loss.sum() / nx / batch_size)
                batch_nlosses.append(nloss.sum() / nx / batch_size)
                batch_rollout_time_steps += labels.shape[1]
            if rollout_time_steps == 0:
                rollout_time_steps = batch_rollout_time_steps
            losses.append(torch.sum(torch.stack(batch_losses)))
            nlosses.append(torch.sum(torch.stack(batch_nlosses)))
    loss_tensor = torch.stack(losses).detach().cpu()
    nloss_tensor = torch.stack(nlosses).detach().cpu()
    if rollout_time_steps <= 0:
        raise RuntimeError("No rollout steps were evaluated.")
    loss_mean = loss_tensor / rollout_time_steps
    nloss_mean = nloss_tensor / rollout_time_steps
    return {
        "rollout_time_steps": int(rollout_time_steps),
        "unrolled_mse": float(loss_mean.mean()),
        "unrolled_mse_std": float(loss_mean.std(unbiased=True)) if len(loss_mean) > 1 else 0.0,
        "trajectory_nmse": float(nloss_mean.mean()),
        "trajectory_nmse_std": float(nloss_mean.std(unbiased=True)) if len(nloss_mean) > 1 else 0.0,
        "unrolled_mse_cumulative": float(loss_tensor.mean()),
        "unrolled_mse_cumulative_std": float(loss_tensor.std(unbiased=True)) if len(loss_tensor) > 1 else 0.0,
        "trajectory_nmse_cumulative": float(nloss_tensor.mean()),
        "trajectory_nmse_cumulative_std": float(nloss_tensor.std(unbiased=True)) if len(nloss_tensor) > 1 else 0.0,
    }


def _trajectory_nmse_sum(pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """LPSDA normalized MSE sum for ``pred`` [B, X, T] and labels [B, T, X]."""
    if pred.ndim != 3 or labels.ndim != 3:
        raise ValueError(f"Expected rank-3 pred/labels, got {tuple(pred.shape)} and {tuple(labels.shape)}")
    if pred.shape[0] != labels.shape[0] or pred.shape[1] != labels.shape[2] or pred.shape[2] != labels.shape[1]:
        raise ValueError(f"Incompatible pred/labels shapes: {tuple(pred.shape)} vs {tuple(labels.shape)}")
    loss = (pred.permute(0, 2, 1) - labels).pow(2)
    nlabels = torch.mean(labels.pow(2), dim=-1, keepdim=True).clamp_min(1e-12)
    return loss.div(nlabels).sum() / pred.shape[1] / pred.shape[0]


def _relative_defect(pred_t: torch.Tensor, pred_equiv: torch.Tensor) -> torch.Tensor:
    diff = (pred_t - pred_equiv).reshape(pred_t.shape[0], -1)
    ref = pred_equiv.reshape(pred_equiv.shape[0], -1)
    return torch.linalg.norm(diff, dim=-1) / torch.linalg.norm(ref, dim=-1).clamp_min(1e-8)


def _orbit_consistency_loss(
    model: nn.Module,
    data: torch.Tensor,
    dx: torch.Tensor,
    dt: torch.Tensor,
    base_pred: torch.Tensor,
    *,
    max_boost: float,
    normalize_by_epsilon: bool,
    eta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    boost, epsilon = _sample_kdv_boost(
        data.shape[0],
        max_boost=max_boost,
        device=data.device,
        dtype=data.dtype,
    )
    data_t = _apply_kdv_galilean_window(data, boost=boost, dx=dx, dt=dt, offset_steps=0)
    pred_t = model(data_t, dx, dt)
    pred_equiv = _apply_kdv_galilean_window(
        base_pred,
        boost=boost,
        dx=dx,
        dt=dt,
        offset_steps=data.shape[-1],
    )
    per_sample = (pred_t - pred_equiv).pow(2).mean(dim=(1, 2))
    if normalize_by_epsilon:
        per_sample = per_sample / (epsilon.pow(2) + eta)
    loss = per_sample.mean()
    defect = _relative_defect(pred_t, pred_equiv)
    return loss, {
        "orbit_loss": float(loss.detach().cpu()),
        "orbit_defect_relative": float(defect.detach().mean().cpu()),
        "epsilon_mean": float(epsilon.detach().mean().cpu()),
        "epsilon_max": float(epsilon.detach().max().cpu()),
    }


@torch.no_grad()
def _evaluate_orbit(
    model: nn.Module,
    loader: DataLoader,
    *,
    nx: int,
    nt_effective: int,
    time_history: int,
    time_future: int,
    device: torch.device,
    max_boost: float,
    orbit_samples: int,
    seed: int | None = None,
) -> dict[str, float]:
    if orbit_samples <= 0 or max_boost <= 0.0:
        return {}
    model.eval()
    max_start_time = nt_effective - time_history - time_future
    rollout_time_steps = 0
    orbit_nlosses: list[torch.Tensor] = []
    defect_values: list[torch.Tensor] = []
    epsilon_values: list[torch.Tensor] = []
    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    for trajectories, dx, dt in loader:
        trajectories = trajectories.to(device)
        dx = dx.to(device)
        dt = dt.to(device)
        batch = trajectories.shape[0]
        for _ in range(orbit_samples):
            if device.type == "cuda":
                boost = (2.0 * torch.rand(batch, device=device, dtype=trajectories.dtype) - 1.0) * max_boost
            else:
                boost = (
                    2.0
                    * torch.rand(batch, generator=generator, device=device, dtype=trajectories.dtype)
                    - 1.0
                ) * max_boost
            epsilon = boost.abs().clamp_min(1e-6)
            data = trajectories[:, time_history : 2 * time_history].permute(0, 2, 1)
            data_t = _apply_kdv_galilean_window(data, boost=boost, dx=dx, dt=dt, offset_steps=0)
            pred = None
            pred_t = None
            batch_nlosses = []
            batch_rollout_time_steps = 0
            for step_idx, start in enumerate(range(time_history, max_start_time + 1, time_future)):
                end = start + time_history
                target_end = end + time_future
                if step_idx > 0:
                    assert pred is not None and pred_t is not None
                    data = torch.cat([data, pred], dim=-1)[..., -time_history:]
                    data_t = torch.cat([data_t, pred_t], dim=-1)[..., -time_history:]
                labels = trajectories[:, end:target_end]
                pred = model(data, dx, dt)
                pred_t = model(data_t, dx, dt)
                offset_steps = time_history + step_idx * time_future
                labels_t = _apply_kdv_galilean_window(
                    labels.permute(0, 2, 1),
                    boost=boost,
                    dx=dx,
                    dt=dt,
                    offset_steps=offset_steps,
                )
                pred_equiv = _apply_kdv_galilean_window(
                    pred,
                    boost=boost,
                    dx=dx,
                    dt=dt,
                    offset_steps=offset_steps,
                )
                batch_nlosses.append(_trajectory_nmse_sum(pred_t, labels_t.permute(0, 2, 1)))
                defect_values.append(_relative_defect(pred_t, pred_equiv).detach().cpu())
                batch_rollout_time_steps += labels.shape[1]
            if rollout_time_steps == 0:
                rollout_time_steps = batch_rollout_time_steps
            orbit_nlosses.append(torch.sum(torch.stack(batch_nlosses)).detach().cpu())
            epsilon_values.append(epsilon.detach().cpu())
    if rollout_time_steps <= 0:
        raise RuntimeError("No orbit rollout steps were evaluated.")
    orbit_tensor = torch.stack(orbit_nlosses) / rollout_time_steps
    defect_tensor = torch.cat(defect_values)
    epsilon_tensor = torch.cat(epsilon_values)
    return {
        "orbit_rollout_time_steps": int(rollout_time_steps),
        "orbit_eval_samples": int(orbit_samples),
        "orbit_max_boost": float(max_boost),
        "orbit_ood_trajectory_nmse": float(orbit_tensor.mean()),
        "orbit_ood_trajectory_nmse_std": float(orbit_tensor.std(unbiased=True)) if len(orbit_tensor) > 1 else 0.0,
        "orbit_ood_trajectory_nmse_cumulative": float((orbit_tensor * rollout_time_steps).mean()),
        "equivariance_defect_relative": float(defect_tensor.mean()),
        "equivariance_defect_relative_std": float(defect_tensor.std(unbiased=True)) if len(defect_tensor) > 1 else 0.0,
        "epsilon_mean": float(epsilon_tensor.mean()),
        "epsilon_mean_std": float(epsilon_tensor.std(unbiased=True)) if len(epsilon_tensor) > 1 else 0.0,
    }


def _latency(
    model: nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    time_history: int,
    device: torch.device,
) -> dict[str, float]:
    trajectories, dx, dt = batch
    data = trajectories[:, time_history : 2 * time_history].permute(0, 2, 1).to(device)
    dx = dx.to(device)
    dt = dt.to(device)
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(data, dx, dt)
        start = time.perf_counter()
        repeats = 10
        for _ in range(repeats):
            model(data, dx, dt)
        elapsed = time.perf_counter() - start
    return {
        "latency_ms_per_batch": 1000.0 * elapsed / repeats,
        "latency_ms_per_sample": 1000.0 * elapsed / repeats / max(1, data.shape[0]),
    }


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    set_seed(seed, deterministic=False)
    random.seed(seed)
    np.random.seed(seed)
    device = get_device(config.get("runtime", {}).get("device", "auto"))
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.exists() or bool(config["dataset"].get("overwrite", False)):
        generate_dataset(config, dataset_path)

    runtime_cfg = config.get("runtime", {})
    run_dir = ensure_dir(runtime_cfg.get("run_dir", "runs/external/lpsda_kdv_faithful/default"))
    save_config(config, run_dir / "config.yaml")

    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    augmentation = dict(train_cfg.get("augmentation", {}))
    method = str(train_cfg.get("method", "baseline")).lower()
    use_aug = method in {"lpsda_aug", "aug", "augmentation", "aug_orbit", "orbit_aug"}
    use_orbit = method in {"orbit", "loco", "local_orbit", "aug_orbit", "orbit_aug"}
    lambda_orbit = float(train_cfg.get("lambda_orbit", 0.05))
    orbit_max_boost = float(train_cfg.get("orbit_max_boost", 0.2))
    normalize_by_epsilon = bool(train_cfg.get("normalize_by_epsilon", True))
    orbit_eta = float(train_cfg.get("orbit_eta", 1e-6))
    eval_orbit_samples = int(train_cfg.get("eval_orbit_samples", 4 if use_orbit else 0))
    eval_orbit_seed = train_cfg.get("eval_orbit_seed", seed + 17_123)
    train_dataset = LPSDAKdvDataset(dataset_path, "train", augmentation=augmentation if use_aug else None)
    valid_dataset = LPSDAKdvDataset(dataset_path, "valid")
    test_dataset = LPSDAKdvDataset(dataset_path, "test")
    batch_size = int(train_cfg.get("batch_size", 16))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    time_history = int(model_cfg.get("time_history", 20))
    time_future = int(model_cfg.get("time_future", 20))
    nx = int(config["dataset"].get("nx", 256))
    nt_effective = int(config["dataset"].get("nt_effective", 140))
    model = LPSDAFNO1d(
        nx=nx,
        time_history=time_history,
        time_future=time_future,
        modes=int(model_cfg.get("modes", 32)),
        width=int(model_cfg.get("width", 256)),
        num_layers=int(model_cfg.get("num_layers", 5)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-4)))
    unrolling = int(train_cfg.get("unrolling", 0))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[unrolling, 5, 10, 15],
        gamma=float(train_cfg.get("lr_decay", 0.4)),
    )
    criterion = nn.MSELoss(reduction="none")
    epochs = int(train_cfg.get("epochs", 20))
    cycles_multiplier = int(train_cfg.get("cycles_multiplier", 2))
    max_cycles = train_cfg.get("max_cycles_per_epoch", None)
    cycles_per_epoch = nt_effective * cycles_multiplier
    if max_cycles is not None:
        cycles_per_epoch = min(cycles_per_epoch, int(max_cycles))
    time_shift = bool(int(augmentation.get("time_shift", 0))) if use_aug else False

    metadata = {
        "method": method,
        "seed": seed,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "config_hash": stable_json_hash(config)[:12],
        "lambda_orbit": lambda_orbit if use_orbit else 0.0,
        "orbit_max_boost": orbit_max_boost if (use_orbit or eval_orbit_samples > 0) else 0.0,
        "eval_orbit_samples": eval_orbit_samples,
    }
    dump_json(metadata, run_dir / "meta.json")

    best_val = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    train_start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        max_unrolling = min(epoch, unrolling)
        unrolling_choices = list(range(max_unrolling + 1))
        epoch_losses = []
        epoch_orbit_losses = []
        epoch_orbit_defects = []
        for cycle in range(cycles_per_epoch):
            for trajectories, dx, dt in train_loader:
                trajectories = trajectories.to(device)
                dx = dx.to(device)
                dt = dt.to(device)
                pf_steps = random.choice(unrolling_choices)
                max_start_time = nt_effective - time_history - time_future * (1 + pf_steps)
                if time_shift:
                    starts = random.choices(list(range(max_start_time + 1)), k=trajectories.shape[0])
                else:
                    starts = random.choices(
                        list(range(time_history, max_start_time + 1, time_history)),
                        k=trajectories.shape[0],
                    )
                data, labels = _create_window(trajectories, starts, time_history, time_future, pf_steps)
                data = data.to(device).permute(0, 2, 1)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for _ in range(pf_steps):
                        pred_pf = model(data, dx, dt)
                        data = torch.cat([data, pred_pf], dim=-1)[..., -time_history:]
                pred = model(data, dx, dt)
                loss = criterion(pred.permute(0, 2, 1), labels).sum()
                if use_orbit:
                    orbit_loss, orbit_stats = _orbit_consistency_loss(
                        model,
                        data,
                        dx,
                        dt,
                        pred,
                        max_boost=orbit_max_boost,
                        normalize_by_epsilon=normalize_by_epsilon,
                        eta=orbit_eta,
                    )
                    loss = loss + lambda_orbit * orbit_loss
                    epoch_orbit_losses.append(orbit_stats["orbit_loss"])
                    epoch_orbit_defects.append(orbit_stats["orbit_defect_relative"])
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()) / batch_size)
            if cycle % int(train_cfg.get("print_interval", 20)) == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} cycle {cycle + 1}/{cycles_per_epoch} "
                    f"loss={np.mean(epoch_losses):.6g}",
                    flush=True,
                )
        scheduler.step()
        val_metrics = _evaluate(
            model,
            valid_loader,
            batch_size=batch_size,
            nx=nx,
            nt_effective=nt_effective,
            time_history=time_history,
            time_future=time_future,
            device=device,
        )
        record = {"epoch": epoch + 1, "train_loss": float(np.mean(epoch_losses)), **val_metrics}
        if epoch_orbit_losses:
            record["train_orbit_loss"] = float(np.mean(epoch_orbit_losses))
            record["train_orbit_defect_relative"] = float(np.mean(epoch_orbit_defects))
        with (run_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch + 1}/{epochs} val_nmse={val_metrics['trajectory_nmse']:.6g} "
            f"best={best_val:.6g}",
            flush=True,
        )
        if val_metrics["unrolled_mse"] < best_val:
            best_val = val_metrics["unrolled_mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, run_dir / "best_model.pt")

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = _evaluate(
        model,
        test_loader,
        batch_size=batch_size,
        nx=nx,
        nt_effective=nt_effective,
        time_history=time_history,
        time_future=time_future,
        device=device,
    )
    test_metrics.update(
        _evaluate_orbit(
            model,
            test_loader,
            nx=nx,
            nt_effective=nt_effective,
            time_history=time_history,
            time_future=time_future,
            device=device,
            max_boost=orbit_max_boost,
            orbit_samples=eval_orbit_samples,
            seed=int(eval_orbit_seed) if eval_orbit_seed is not None else None,
        )
    )
    latency = _latency(model, next(iter(test_loader)), time_history=time_history, device=device)
    results = {
        **metadata,
        **test_metrics,
        **latency,
        "best_val_unrolled_mse": best_val,
        "train_wall_seconds": time.perf_counter() - train_start,
        "cycles_per_epoch": cycles_per_epoch,
        "epochs": epochs,
    }
    dump_json(results, run_dir / "test_metrics.json")
    return results


def evaluate_checkpoint_from_config(
    config: dict[str, Any],
    *,
    checkpoint: str | None = None,
    output: str | None = None,
) -> dict[str, Any]:
    seed = int(config.get("seed", 0))
    set_seed(seed, deterministic=False)
    device = get_device(config.get("runtime", {}).get("device", "auto"))
    dataset_path = Path(config["dataset"]["path"])
    if not dataset_path.exists() or bool(config["dataset"].get("overwrite", False)):
        generate_dataset(config, dataset_path)

    runtime_cfg = config.get("runtime", {})
    run_dir = ensure_dir(runtime_cfg.get("run_dir", "runs/external/lpsda_kdv_faithful/default"))
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    batch_size = int(train_cfg.get("batch_size", 16))
    test_dataset = LPSDAKdvDataset(dataset_path, "test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    time_history = int(model_cfg.get("time_history", 20))
    time_future = int(model_cfg.get("time_future", 20))
    nx = int(config["dataset"].get("nx", 256))
    nt_effective = int(config["dataset"].get("nt_effective", 140))
    model = LPSDAFNO1d(
        nx=nx,
        time_history=time_history,
        time_future=time_future,
        modes=int(model_cfg.get("modes", 32)),
        width=int(model_cfg.get("width", 256)),
        num_layers=int(model_cfg.get("num_layers", 5)),
    ).to(device)
    checkpoint_path = Path(checkpoint) if checkpoint is not None else run_dir / "best_model.pt"
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    eval_orbit_samples = int(train_cfg.get("eval_orbit_samples", 4))
    eval_orbit_seed = train_cfg.get("eval_orbit_seed", seed + 17_123)
    orbit_max_boost = float(train_cfg.get("orbit_max_boost", 0.2))
    metadata = {
        "method": str(train_cfg.get("method", "baseline")).lower(),
        "seed": seed,
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "dataset_path": str(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "config_hash": stable_json_hash(config)[:12],
        "checkpoint_path": str(checkpoint_path),
        "orbit_max_boost": orbit_max_boost,
        "eval_orbit_samples": eval_orbit_samples,
    }
    test_metrics = _evaluate(
        model,
        test_loader,
        batch_size=batch_size,
        nx=nx,
        nt_effective=nt_effective,
        time_history=time_history,
        time_future=time_future,
        device=device,
    )
    test_metrics.update(
        _evaluate_orbit(
            model,
            test_loader,
            nx=nx,
            nt_effective=nt_effective,
            time_history=time_history,
            time_future=time_future,
            device=device,
            max_boost=orbit_max_boost,
            orbit_samples=eval_orbit_samples,
            seed=int(eval_orbit_seed) if eval_orbit_seed is not None else None,
        )
    )
    latency = _latency(model, next(iter(test_loader)), time_history=time_history, device=device)
    results = {**metadata, **test_metrics, **latency}
    output_path = Path(output) if output is not None else run_dir / "test_metrics_orbit.json"
    dump_json(results, output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Faithful independent LPSDA KdV replication.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved checkpoint without training.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path for --eval-only.")
    parser.add_argument("--output", default=None, help="Metrics output path for --eval-only.")
    args = parser.parse_args()
    config = recursive_update(load_config(args.config), parse_overrides(args.override))
    if args.eval_only:
        results = evaluate_checkpoint_from_config(config, checkpoint=args.checkpoint, output=args.output)
    else:
        results = train_from_config(config)
    for key, value in sorted(results.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
