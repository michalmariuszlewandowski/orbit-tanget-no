#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
import torch
import yaml

from otno.data.datasets import load_tensor_dataset
from otno.data.solvers2d import solve_navier_stokes_vorticity_2d
from otno.symmetry.registry import build_transform
from otno.symmetry.transforms import NavierStokes2DGalilean, TransformSample
from otno.utils import file_sha256


def _relative_l2_per_sample(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = (x - y).reshape(x.shape[0], -1)
    ref = y.reshape(y.shape[0], -1)
    return torch.linalg.norm(diff, dim=1) / torch.linalg.norm(ref, dim=1).clamp_min(1e-12)


def _sample_indices(total: int, count: int, seed: int) -> torch.Tensor:
    if count > total:
        raise ValueError(f"Requested {count} samples from split with only {total} examples")
    generator = torch.Generator().manual_seed(int(seed))
    return torch.randperm(total, generator=generator)[:count]


def _sample_boosts(count: int, max_boost: float, seed: int, dtype: torch.dtype) -> torch.Tensor:
    generator = torch.Generator().manual_seed(int(seed) + 10_000)
    return (2 * torch.rand(count, 2, generator=generator, dtype=dtype) - 1) * float(max_boost)


def _format_pm(mean: float, std: float) -> str:
    return rf"\({mean:.2e}\pm{std:.2e}\)"


def _write_tex(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Solver-level Galilean closure for the N64 2D Navier--Stokes dataset. "
            "The residual is "
            "$\\|S(T_g a)-T_gS(a)\\|_2/\\|T_gS(a)\\|_2$ at the training boost radius.}\n"
            "\\label{tab:n64-solver-closure}\n"
            "\\begin{tabular}{lrrrrr}\n"
            "\\toprule\n"
            "Setting & Samples & Max boost & Mean $\\pm$ std & P95 & Max \\\\\n"
            "\\midrule\n"
        )
        for _, row in summary.sort_values("max_boost").iterrows():
            label = "N64 Galilean train radius" if abs(float(row["max_boost"]) - 0.25) < 1e-12 else "N64 Galilean OOD radius"
            f.write(
                f"{label} & {int(row['num_samples'])} & "
                f"{row['max_boost']:.2f} & "
                f"{_format_pm(row['closure_residual_mean'], row['closure_residual_std'])} & "
                f"\\({row['closure_residual_p95']:.2e}\\) & "
                f"\\({row['closure_residual_max']:.2e}\\) \\\\\n"
            )
        f.write(
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "\\end{table}\n"
        )


def run_closure(config: dict[str, Any], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_cfg = config["dataset"]
    dataset_path = Path(args.dataset_path or dataset_cfg["path"])
    split = str(args.split)
    dataset, metadata = load_tensor_dataset(dataset_path, split)
    indices = _sample_indices(len(dataset), int(args.num_samples), int(args.seed))
    a = dataset.a[indices]
    u = dataset.u[indices]

    transform = build_transform(config.get("symmetry"))
    if not isinstance(transform, NavierStokes2DGalilean):
        raise TypeError("Solver closure table currently expects a NavierStokes2DGalilean symmetry")
    max_boost = float(args.max_boost if args.max_boost is not None else transform.max_boost)
    boosts = _sample_boosts(a.shape[0], max_boost, int(args.seed), a.dtype)

    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    batch_size = int(args.batch_size or dataset_cfg.get("solver_batch_size", 8))
    viscosity = float(args.viscosity if args.viscosity is not None else dataset_cfg.get("viscosity", 1e-3))
    final_time = float(args.final_time if args.final_time is not None else dataset_cfg.get("final_time", 0.5))
    dt = float(args.dt if args.dt is not None else dataset_cfg.get("dt", 1e-3))
    dealias = bool(dataset_cfg.get("dealias", True))

    for start in range(0, a.shape[0], batch_size):
        stop = min(start + batch_size, a.shape[0])
        a_b = a[start:stop].to(device)
        u_b = u[start:stop].to(device)
        boost_b = boosts[start:stop].to(device)
        sample = TransformSample(
            params={"boost": boost_b},
            epsilon=torch.linalg.norm(boost_b, dim=-1).clamp_min(1e-12),
            name=transform.name,
        )
        transformed_input = transform.apply_input(a_b, sample)
        solved_transformed = solve_navier_stokes_vorticity_2d(
            transformed_input[..., 0],
            viscosity=viscosity,
            final_time=final_time,
            dt=dt,
            ambient_velocity=transformed_input[:, 0, 0, 1:3],
            dealias=dealias,
        )[..., None]
        transformed_target = transform.apply_output(u_b, sample)
        residual = _relative_l2_per_sample(solved_transformed, transformed_target)
        abs_l2 = torch.linalg.norm((solved_transformed - transformed_target).reshape(stop - start, -1), dim=1)
        ref_l2 = torch.linalg.norm(transformed_target.reshape(stop - start, -1), dim=1)
        for local_idx, value in enumerate(residual.detach().cpu()):
            sample_idx = int(indices[start + local_idx].item())
            base_boost = a[start + local_idx, 0, 0, 1:3]
            delta = boosts[start + local_idx]
            rows.append(
                {
                    "split": split,
                    "max_boost": max_boost,
                    "sample_index": sample_idx,
                    "base_boost_x": float(base_boost[0]),
                    "base_boost_y": float(base_boost[1]),
                    "delta_boost_x": float(delta[0]),
                    "delta_boost_y": float(delta[1]),
                    "delta_boost_norm": float(torch.linalg.norm(delta)),
                    "closure_residual_relative": float(value),
                    "closure_abs_l2": float(abs_l2[local_idx].detach().cpu()),
                    "closure_reference_l2": float(ref_l2[local_idx].detach().cpu()),
                }
            )

    sample_df = pd.DataFrame(rows).sort_values("sample_index")
    residuals = sample_df["closure_residual_relative"]
    summary = pd.DataFrame(
        [
            {
                "problem": "n64_navier_stokes_galilean",
                "dataset_path": str(dataset_path),
                "dataset_sha256": file_sha256(dataset_path),
                "dataset_fingerprint": metadata.get("config_fingerprint"),
                "split": split,
                "num_samples": int(len(sample_df)),
                "seed": int(args.seed),
                "n": int(dataset_cfg.get("n", a.shape[1])),
                "final_time": final_time,
                "dt": dt,
                "viscosity": viscosity,
                "max_boost": max_boost,
                "closure_residual_mean": float(residuals.mean()),
                "closure_residual_std": float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0,
                "closure_residual_median": float(residuals.median()),
                "closure_residual_p95": float(residuals.quantile(0.95)),
                "closure_residual_max": float(residuals.max()),
            }
        ]
    )
    return sample_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper-facing solver-level Galilean closure tables."
    )
    parser.add_argument("--config", default="configs/pilot/2d_navier_stokes_galilean.yaml")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-boost", type=float, default=None)
    parser.add_argument(
        "--max-boosts",
        default=None,
        help="Comma-separated boost radii. Overrides --max-boost and writes one combined table.",
    )
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--viscosity", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-prefix", default="runs/paper_tables/2d_galilean_n64_solver_closure")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if args.max_boosts:
        boost_values = [float(value.strip()) for value in str(args.max_boosts).split(",") if value.strip()]
    else:
        boost_values = [args.max_boost]
    samples = []
    summaries = []
    for max_boost in boost_values:
        local_args = copy.copy(args)
        local_args.max_boost = max_boost
        sample_df, summary = run_closure(config, local_args)
        samples.append(sample_df)
        summaries.append(summary)
    sample_df = pd.concat(samples, ignore_index=True)
    summary = pd.concat(summaries, ignore_index=True)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(out_prefix.with_suffix(".samples.csv"), index=False)
    summary.to_csv(out_prefix.with_suffix(".summary.csv"), index=False)
    _write_tex(summary, out_prefix.with_suffix(".tex"))
    print(f"wrote {out_prefix.with_suffix('.samples.csv')}")
    print(f"wrote {out_prefix.with_suffix('.summary.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
