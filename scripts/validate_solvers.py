#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import json

import torch

from otno.data.solvers1d import random_fourier_field_1d, solve_advection_1d, solve_burgers_1d
from otno.data.solvers2d import random_fourier_field_2d, solve_navier_stokes_vorticity_2d
from otno.symmetry.transforms import Burgers1DGalilean, D4Pseudoscalar2D, TransformSample, Translation1D, Translation2D
from otno.utils import dump_json, ensure_dir, set_seed


def _rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.linalg.norm((x - y).reshape(-1)) / torch.linalg.norm(y.reshape(-1)).clamp_min(1e-12))


def run_validation(seed: int = 123) -> dict[str, float]:
    set_seed(seed)
    metrics: dict[str, float] = {}

    u0 = random_fourier_field_1d(4, 64, modes=5, amplitude=0.5, seed=seed)[..., None]
    t1 = Translation1D(max_shift=0.2)
    sample_t = TransformSample(
        params={"shift": torch.tensor([0.1, -0.07, 0.03, 0.0])},
        epsilon=torch.ones(4),
        name="translation1d",
    )
    left = solve_advection_1d(t1.apply_input(u0, sample_t)[..., 0], velocity=0.8, final_time=0.25)[..., None]
    right = t1.apply_output(solve_advection_1d(u0[..., 0], velocity=0.8, final_time=0.25)[..., None], sample_t)
    metrics["advection_translation_rel"] = _rel(left, right)

    final_time = 0.02
    bg = Burgers1DGalilean(max_boost=0.1, final_time=final_time)
    sample_g = TransformSample(
        params={"boost": torch.tensor([0.08, -0.05, 0.02, -0.01])},
        epsilon=torch.ones(4),
        name="burgers1d_galilean",
    )
    left = solve_burgers_1d(bg.apply_input(u0, sample_g)[..., 0], viscosity=0.02, final_time=final_time, dt=5e-4)[..., None]
    right = bg.apply_output(solve_burgers_1d(u0[..., 0], viscosity=0.02, final_time=final_time, dt=5e-4)[..., None], sample_g)
    metrics["burgers_galilean_rel"] = _rel(left, right)
    coarse = solve_burgers_1d(u0[..., 0], viscosity=0.02, final_time=final_time, dt=2e-3)
    fine = solve_burgers_1d(u0[..., 0], viscosity=0.02, final_time=final_time, dt=1e-3)
    metrics["burgers_dt_halving_rel"] = _rel(coarse, fine)

    omega0 = random_fourier_field_2d(3, 16, smoothness=4.0, amplitude=0.1, seed=seed + 1)[..., None]
    t2 = Translation2D(max_shift=0.1)
    sample_2d = TransformSample(
        params={"shift": torch.tensor([[1.0 / 16.0, -2.0 / 16.0], [0.0, 1.0 / 16.0], [2.0 / 16.0, 0.0]])},
        epsilon=torch.ones(3),
        name="translation2d",
    )
    left = solve_navier_stokes_vorticity_2d(t2.apply_input(omega0, sample_2d)[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002)[..., None]
    right = t2.apply_output(
        solve_navier_stokes_vorticity_2d(omega0[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002)[..., None],
        sample_2d,
    )
    metrics["ns2d_translation_rel"] = _rel(left, right)
    d4 = D4Pseudoscalar2D()
    sample_d4 = TransformSample(
        params={"k": torch.tensor([1, 2, 0]), "flip": torch.tensor([False, False, True])},
        epsilon=torch.ones(3),
        name="d4_pseudoscalar2d",
    )
    left = solve_navier_stokes_vorticity_2d(d4.apply_input(omega0, sample_d4)[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002)[..., None]
    right = d4.apply_output(
        solve_navier_stokes_vorticity_2d(omega0[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002)[..., None],
        sample_d4,
    )
    metrics["ns2d_d4_pseudoscalar_rel"] = _rel(left, right)
    coarse = solve_navier_stokes_vorticity_2d(omega0[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002)
    fine = solve_navier_stokes_vorticity_2d(omega0[..., 0], viscosity=1e-2, final_time=0.01, dt=0.001)
    metrics["ns2d_dt_halving_rel"] = _rel(coarse, fine)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run numerical symmetry and solver-convergence checks.")
    parser.add_argument("--out", default="runs/quality/solver_validation.json")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    metrics = run_validation(args.seed)
    out = Path(args.out)
    ensure_dir(out.parent)
    dump_json(metrics, out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    thresholds = {
        "advection_translation_rel": 1e-6,
        "burgers_galilean_rel": 5e-3,
        "burgers_dt_halving_rel": 5e-4,
        "ns2d_translation_rel": 1e-5,
        "ns2d_d4_pseudoscalar_rel": 1e-5,
        "ns2d_dt_halving_rel": 5e-4,
    }
    failures = {key: value for key, value in metrics.items() if value > thresholds[key]}
    if failures:
        print("SOLVER VALIDATION FAILED")
        for key, value in failures.items():
            print(f"- {key}: {value:.3e} > {thresholds[key]:.3e}")
        raise SystemExit(1)
    print("SOLVER VALIDATION PASSED")


if __name__ == "__main__":
    main()
