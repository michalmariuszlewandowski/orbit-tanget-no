from __future__ import annotations

from pathlib import Path
from typing import Any

import copy
import hashlib
import json

import numpy as np
import torch

from otno.config import validate_config
from otno.data.solvers1d import (
    random_fourier_field_1d,
    solve_advection_1d,
    solve_burgers_1d,
)
from otno.data.solvers2d import random_fourier_field_2d, solve_navier_stokes_vorticity_2d


def dataset_config_fingerprint(config: dict[str, Any]) -> str:
    """Return a stable fingerprint of data-generating parameters.

    File-location and cache-control keys are excluded so the same dataset generated
    at a different path has the same fingerprint.
    """
    cfg = copy.deepcopy(config.get("dataset", config))
    for key in [
        "path",
        "generate_if_missing",
        "overwrite",
        "allow_stale",
        "source_path",
        "raw_dir",
        "download_if_missing",
        "download_url",
    ]:
        cfg.pop(key, None)
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metadata(kind: str, config: dict[str, Any]) -> dict[str, Any]:
    return {"kind": kind, "config_fingerprint": dataset_config_fingerprint(config), **config}


def validate_existing_dataset(path: str | Path, config: dict[str, Any]) -> None:
    """Raise when an existing dataset file was generated from different parameters."""
    path = Path(path)
    if not path.exists():
        return
    config = config.get("dataset", config)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    expected = dataset_config_fingerprint(config)
    actual = metadata.get("config_fingerprint")
    if actual is None:
        if bool(config.get("allow_stale", False)):
            return
        raise ValueError(
            f"Existing dataset {path} has no config_fingerprint. Regenerate it or set dataset.allow_stale=true."
        )
    if actual != expected:
        raise ValueError(
            f"Existing dataset {path} metadata does not match requested config: "
            f"actual fingerprint {actual}, expected {expected}. Use a new path or set overwrite=true."
        )


def _split_counts(config: dict[str, Any]) -> dict[str, int]:
    return {
        "train": int(config.get("num_train", 1024)),
        "val": int(config.get("num_val", 128)),
        "test": int(config.get("num_test", 128)),
    }


def _save_dataset(payload: dict[str, Any], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def generate_advection_1d(config: dict[str, Any], out_path: str | Path) -> Path:
    n = int(config.get("n", 128))
    final_time = float(config.get("final_time", 0.5))
    velocity = float(config.get("velocity", 1.0))
    seed = int(config.get("seed", 0))
    counts = _split_counts(config)
    payload: dict[str, Any] = {"metadata": _metadata("advection1d", config), "splits": {}}
    offset = 0
    for split, num in counts.items():
        a = random_fourier_field_1d(
            num,
            n,
            modes=int(config.get("modes", 8)),
            amplitude=float(config.get("amplitude", 1.0)),
            seed=seed + offset,
        )
        u = solve_advection_1d(a, velocity=velocity, final_time=final_time)
        payload["splits"][split] = {"a": a[..., None].contiguous(), "u": u[..., None].contiguous()}
        offset += 11
    return _save_dataset(payload, out_path)


def generate_burgers_1d(config: dict[str, Any], out_path: str | Path) -> Path:
    n = int(config.get("n", 128))
    final_time = float(config.get("final_time", 0.5))
    dt = float(config.get("dt", 1e-3))
    viscosity = float(config.get("viscosity", 0.01))
    seed = int(config.get("seed", 0))
    counts = _split_counts(config)
    batch_size = int(config.get("solver_batch_size", 64))
    payload: dict[str, Any] = {"metadata": _metadata("burgers1d", config), "splits": {}}
    offset = 0
    for split, num in counts.items():
        a_all = []
        u_all = []
        generated = 0
        while generated < num:
            b = min(batch_size, num - generated)
            a = random_fourier_field_1d(
                b,
                n,
                modes=int(config.get("modes", 8)),
                amplitude=float(config.get("amplitude", 0.8)),
                mean=float(config.get("mean", 0.0)),
                seed=seed + offset + generated,
            )
            u = solve_burgers_1d(
                a,
                viscosity=viscosity,
                final_time=final_time,
                dt=dt,
                dealias=bool(config.get("dealias", True)),
            )
            a_all.append(a.cpu())
            u_all.append(u.cpu())
            generated += b
        a_cat = torch.cat(a_all, dim=0)
        u_cat = torch.cat(u_all, dim=0)
        payload["splits"][split] = {"a": a_cat[..., None].contiguous(), "u": u_cat[..., None].contiguous()}
        offset += 100_000
    return _save_dataset(payload, out_path)


def random_sine_field_1d(
    num: int,
    n: int,
    *,
    modes: int = 8,
    amplitude: float = 1.0,
    seed: int | None = None,
) -> torch.Tensor:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    x = torch.linspace(0.0, 1.0, n)
    k = torch.arange(1, modes + 1, dtype=torch.float32)
    basis = torch.sin(torch.pi * k[:, None] * x[None, :])
    coeff = torch.randn(num, modes, generator=generator) / k[None, :]
    field = coeff @ basis
    field = field / (field.std(dim=-1, keepdim=True) + 1e-6)
    return amplitude * field


def solve_heat_1d_dirichlet(
    a: torch.Tensor,
    *,
    diffusivity: float = 0.01,
    final_time: float = 0.1,
    modes: int | None = None,
) -> torch.Tensor:
    if a.ndim != 2:
        raise ValueError(f"Expected [batch, n], got {tuple(a.shape)}")
    batch, n = a.shape
    m = int(modes or min(n - 2, 32))
    x = torch.linspace(0.0, 1.0, n, device=a.device, dtype=a.dtype)
    k = torch.arange(1, m + 1, device=a.device, dtype=a.dtype)
    basis = torch.sin(torch.pi * k[:, None] * x[None, :])
    coeff = (2.0 / max(1, n - 1)) * torch.einsum("bn,kn->bk", a, basis)
    decay = torch.exp(-diffusivity * (torch.pi * k).pow(2) * final_time)
    return torch.einsum("bk,kn->bn", coeff * decay[None, :], basis)


def generate_heat_1d_dirichlet(config: dict[str, Any], out_path: str | Path) -> Path:
    n = int(config.get("n", 128))
    final_time = float(config.get("final_time", 0.1))
    diffusivity = float(config.get("diffusivity", 0.01))
    modes = int(config.get("modes", 8))
    seed = int(config.get("seed", 0))
    counts = _split_counts(config)
    payload: dict[str, Any] = {"metadata": _metadata("heat1d_dirichlet", config), "splits": {}}
    offset = 0
    for split, num in counts.items():
        a = random_sine_field_1d(
            num,
            n,
            modes=modes,
            amplitude=float(config.get("amplitude", 1.0)),
            seed=seed + offset,
        )
        u = solve_heat_1d_dirichlet(
            a,
            diffusivity=diffusivity,
            final_time=final_time,
            modes=int(config.get("solve_modes", modes)),
        )
        payload["splits"][split] = {"a": a[..., None].contiguous(), "u": u[..., None].contiguous()}
        offset += 11
    return _save_dataset(payload, out_path)


def generate_navier_stokes_2d(config: dict[str, Any], out_path: str | Path) -> Path:
    n = int(config.get("n", 64))
    final_time = float(config.get("final_time", 0.5))
    dt = float(config.get("dt", 1e-3))
    viscosity = float(config.get("viscosity", 1e-3))
    seed = int(config.get("seed", 0))
    counts = _split_counts(config)
    batch_size = int(config.get("solver_batch_size", 8))
    payload: dict[str, Any] = {"metadata": _metadata("navier_stokes_vorticity2d", config), "splits": {}}
    offset = 0
    for split, num in counts.items():
        a_all = []
        u_all = []
        generated = 0
        while generated < num:
            b = min(batch_size, num - generated)
            a = random_fourier_field_2d(
                b,
                n,
                smoothness=float(config.get("smoothness", 8.0)),
                amplitude=float(config.get("amplitude", 1.0)),
                seed=seed + offset + generated,
            )
            u = solve_navier_stokes_vorticity_2d(
                a,
                viscosity=viscosity,
                final_time=final_time,
                dt=dt,
                dealias=bool(config.get("dealias", True)),
            )
            a_all.append(a.cpu())
            u_all.append(u.cpu())
            generated += b
        a_cat = torch.cat(a_all, dim=0)
        u_cat = torch.cat(u_all, dim=0)
        payload["splits"][split] = {"a": a_cat[..., None].contiguous(), "u": u_cat[..., None].contiguous()}
        offset += 100_000
    return _save_dataset(payload, out_path)


def generate_boosted_navier_stokes_2d(config: dict[str, Any], out_path: str | Path) -> Path:
    n = int(config.get("n", 64))
    final_time = float(config.get("final_time", 0.5))
    dt = float(config.get("dt", 1e-3))
    viscosity = float(config.get("viscosity", 1e-3))
    max_boost = float(config.get("max_boost", 0.5))
    seed = int(config.get("seed", 0))
    counts = _split_counts(config)
    batch_size = int(config.get("solver_batch_size", 8))
    payload: dict[str, Any] = {"metadata": _metadata("navier_stokes_vorticity2d_boosted", config), "splits": {}}
    offset = 0
    for split, num in counts.items():
        a_all = []
        u_all = []
        generated = 0
        while generated < num:
            b = min(batch_size, num - generated)
            sample_seed = seed + offset + generated
            omega0 = random_fourier_field_2d(
                b,
                n,
                smoothness=float(config.get("smoothness", 8.0)),
                amplitude=float(config.get("amplitude", 1.0)),
                seed=sample_seed,
            )
            generator = torch.Generator().manual_seed(sample_seed + 50_000)
            boost = (2 * torch.rand(b, 2, generator=generator) - 1) * max_boost
            u = solve_navier_stokes_vorticity_2d(
                omega0,
                viscosity=viscosity,
                final_time=final_time,
                dt=dt,
                ambient_velocity=boost,
                dealias=bool(config.get("dealias", True)),
            )
            boost_fields = boost[:, None, None, :].expand(b, n, n, 2)
            a_all.append(torch.cat([omega0[..., None], boost_fields], dim=-1).cpu())
            u_all.append(u[..., None].cpu())
            generated += b
        payload["splits"][split] = {
            "a": torch.cat(a_all, dim=0).contiguous(),
            "u": torch.cat(u_all, dim=0).contiguous(),
        }
        offset += 100_000
    return _save_dataset(payload, out_path)


def _find_rmd17_source(config: dict[str, Any]) -> Path:
    if "source_path" in config:
        source = Path(config["source_path"])
        if source.exists():
            return source
        raise FileNotFoundError(source)
    raw_dir = Path(config.get("raw_dir", "data/rmd17/raw"))
    molecule = str(config.get("molecule", "ethanol"))
    candidates = [
        raw_dir / f"rmd17_{molecule}.npz",
        raw_dir / f"{molecule}.npz",
        raw_dir / f"rmd17_{molecule}_dft.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No rMD17 source found for molecule={molecule!r}; checked "
        + ", ".join(str(path) for path in candidates)
    )


def generate_rmd17_force(config: dict[str, Any], out_path: str | Path) -> Path:
    source = _find_rmd17_source(config)
    raw = np.load(source)
    coords = torch.as_tensor(raw["coords"], dtype=torch.float32)
    forces = torch.as_tensor(raw["forces"], dtype=torch.float32)
    charges_key = "nuclear_charges" if "nuclear_charges" in raw else "z"
    charges = torch.as_tensor(raw[charges_key], dtype=torch.float32)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected coords [samples, atoms, 3], got {tuple(coords.shape)}")
    if forces.shape != coords.shape:
        raise ValueError(f"Expected forces shape {tuple(coords.shape)}, got {tuple(forces.shape)}")
    if bool(config.get("center_positions", True)):
        coords = coords - coords.mean(dim=1, keepdim=True)
    charge_scale = float(config.get("charge_scale", 10.0))
    charge_channel = (charges / charge_scale).view(1, -1, 1).expand(coords.shape[0], -1, 1)
    inputs = torch.cat([coords, charge_channel], dim=-1).contiguous()
    targets = forces.contiguous()

    counts = _split_counts(config)
    total = sum(counts.values())
    if total > inputs.shape[0]:
        raise ValueError(f"Requested {total} rMD17 samples but source only has {inputs.shape[0]}")
    generator = torch.Generator().manual_seed(int(config.get("split_seed", config.get("seed", 0))))
    perm = torch.randperm(inputs.shape[0], generator=generator)[:total]
    payload: dict[str, Any] = {
        "metadata": {
            **_metadata("rmd17_force", config),
            "source_path": str(source),
            "n_atoms": int(coords.shape[1]),
        },
        "splits": {},
    }
    start = 0
    for split, num in counts.items():
        idx = perm[start : start + num]
        payload["splits"][split] = {"a": inputs[idx].contiguous(), "u": targets[idx].contiguous()}
        start += num
    return _save_dataset(payload, out_path)


def generate_dataset_from_config(config: dict[str, Any], out_path: str | Path | None = None) -> Path:
    dataset_cfg = config.get("dataset", config)
    kind = str(dataset_cfg.get("kind", dataset_cfg.get("name", "burgers1d"))).lower()
    out_path = Path(out_path or dataset_cfg.get("path", f"data/{kind}.pt"))
    validate_config({"dataset": {**dataset_cfg, "kind": kind, "path": str(out_path)}})
    if out_path.exists() and not bool(dataset_cfg.get("overwrite", False)):
        validate_existing_dataset(out_path, dataset_cfg)
        return out_path
    if kind in {"advection1d", "1d_advection"}:
        return generate_advection_1d(dataset_cfg, out_path)
    if kind in {"burgers1d", "1d_burgers"}:
        return generate_burgers_1d(dataset_cfg, out_path)
    if kind in {"heat1d_dirichlet", "dirichlet_heat1d"}:
        return generate_heat_1d_dirichlet(dataset_cfg, out_path)
    if kind in {"navier_stokes_vorticity2d", "ns2d", "2d_navier_stokes"}:
        return generate_navier_stokes_2d(dataset_cfg, out_path)
    if kind in {
        "navier_stokes_vorticity2d_boosted",
        "boosted_navier_stokes_vorticity2d",
        "ns2d_boosted",
        "boosted_ns2d",
    }:
        return generate_boosted_navier_stokes_2d(dataset_cfg, out_path)
    if kind in {"rmd17_force", "rmd17"}:
        return generate_rmd17_force(dataset_cfg, out_path)
    raise ValueError(f"Unknown dataset kind: {kind}")
