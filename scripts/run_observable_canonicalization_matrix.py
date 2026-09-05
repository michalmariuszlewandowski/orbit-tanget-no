#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from otno.data.datasets import load_tensor_dataset
from otno.models import ObservableGalileanCanonicalFNO2d, build_model
from otno.models.canonical import observed_galilean_boost_2d
from otno.symmetry.registry import build_transform
from otno.symmetry.transforms import NavierStokes2DGalilean, TransformSample
from otno.training.losses import relative_defect_per_sample, relative_l2_per_sample
from otno.training.metrics import _add_stats, _finalize, _rng_context, measure_inference_latency
from otno.utils import get_device
from run_ood_severity_matrix import (
    _evaluation_metadata, _jobs, _read_cached_metrics, _symmetry_value, _write_metrics,
)


def _observed_boost_sample(a: torch.Tensor, transform: NavierStokes2DGalilean) -> TransformSample:
    boost = observed_galilean_boost_2d(
        a,
        boost_x_channel=transform.boost_x_channel,
        boost_y_channel=transform.boost_y_channel,
    )
    eps = torch.linalg.norm(boost, dim=-1).clamp_min(1e-6)
    return TransformSample({"boost": boost}, eps, transform.name)


def _canonicalize_input(a: torch.Tensor, transform: NavierStokes2DGalilean) -> torch.Tensor:
    z = a.clone()
    z[..., transform.boost_x_channel] = 0.0
    z[..., transform.boost_y_channel] = 0.0
    return z


def _observable_wrapper(
    model: torch.nn.Module,
    cfg: dict[str, Any],
    transform: NavierStokes2DGalilean,
) -> ObservableGalileanCanonicalFNO2d:
    model_cfg = cfg.get("model", {})
    wrapper = ObservableGalileanCanonicalFNO2d(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        width=int(model_cfg.get("width", 64)),
        modes1=int(model_cfg.get("modes1", model_cfg.get("modes", 12))),
        modes2=int(model_cfg.get("modes2", model_cfg.get("modes", 12))),
        depth=int(model_cfg.get("depth", 4)),
        add_grid=bool(model_cfg.get("add_grid", True)),
        boost_x_channel=transform.boost_x_channel,
        boost_y_channel=transform.boost_y_channel,
        length=transform.length,
        final_time=transform.final_time,
    )
    wrapper.base = model
    return wrapper


@torch.no_grad()
def evaluate_observable_canonicalization(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    transform: NavierStokes2DGalilean,
    n_orbit_samples: int,
    seed: int | None,
) -> dict[str, Any]:
    model.eval()
    sums: dict[str, float] = defaultdict(float)
    sumsqs: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    start = time.perf_counter()
    num_samples = 0
    num_batches = 0

    with _rng_context(seed):
        if seed is not None:
            torch.manual_seed(seed)
        for batch in loader:
            a = batch["a"].to(device)
            u = batch["u"].to(device)
            base_sample = _observed_boost_sample(a, transform)
            canonical_pred = model(_canonicalize_input(a, transform))
            observed_pred = transform.apply_output(canonical_pred, base_sample)
            _add_stats("observable_canonical_relative_l2", relative_l2_per_sample(observed_pred, u), sums, sumsqs, counts)
            _add_stats("observed_boost_norm", base_sample.epsilon, sums, sumsqs, counts)
            num_samples += a.shape[0]
            num_batches += 1

            for _ in range(n_orbit_samples):
                delta_sample = transform.sample(a.shape[0], a.device, a.dtype)
                a_t = transform.apply_input(a, delta_sample)
                u_t = transform.apply_output(u, delta_sample)
                direct_pred = model(a_t)
                total_sample = _observed_boost_sample(a_t, transform)
                canonical_pred = model(_canonicalize_input(a_t, transform))
                restored_pred = transform.apply_output(canonical_pred, total_sample)
                mask = transform.output_mask(u_t, delta_sample)
                _add_stats("orbit_ood_relative_l2", relative_l2_per_sample(direct_pred, u_t, mask=mask), sums, sumsqs, counts)
                _add_stats(
                    "observable_canonical_ood_relative_l2",
                    relative_l2_per_sample(restored_pred, u_t, mask=mask),
                    sums,
                    sumsqs,
                    counts,
                )
                _add_stats(
                    "observable_canonical_gap_relative",
                    relative_defect_per_sample(direct_pred, restored_pred, mask=mask),
                    sums,
                    sumsqs,
                    counts,
                )
                _add_stats("delta_boost_norm", delta_sample.epsilon, sums, sumsqs, counts)
                _add_stats("total_boost_norm", total_sample.epsilon, sums, sumsqs, counts)

    metrics = _finalize(sums, sumsqs, counts)
    for source, target in [
        ("observed_boost_norm", "observed_boost_norm_mean"),
        ("delta_boost_norm", "epsilon_mean"),
        ("total_boost_norm", "total_boost_norm_mean"),
    ]:
        if source in metrics:
            metrics[target] = metrics.pop(source)
            metrics[f"{target}_std"] = metrics.pop(f"{source}_std")
            metrics[f"{target}_stderr"] = metrics.pop(f"{source}_stderr")
    metrics.update({"num_samples": num_samples, "num_batches": num_batches, "eval_seconds": time.perf_counter() - start})
    return metrics


def _aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = ["severity", "severity_scale", "max_boost", "method"]
    value_cols = [
        "orbit_ood_relative_l2",
        "observable_canonical_ood_relative_l2",
        "observable_canonical_gap_relative",
        "observable_canonical_relative_l2",
        "epsilon_mean",
        "total_boost_norm_mean",
        "direct_latency_ms_per_sample",
        "observable_canonical_latency_ms_per_sample",
        "observable_canonical_latency_overhead_pct",
    ]
    existing_values = [col for col in value_cols if col in df.columns]
    aggregate = (
        df.groupby(group_cols, dropna=False)[existing_values]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in col if part).rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in aggregate.columns
    ]
    if {
        "orbit_ood_relative_l2_mean",
        "observable_canonical_ood_relative_l2_mean",
    }.issubset(aggregate.columns):
        aggregate["observable_canonical_delta_pct"] = 100.0 * (
            aggregate["observable_canonical_ood_relative_l2_mean"]
            - aggregate["orbit_ood_relative_l2_mean"]
        ) / aggregate["orbit_ood_relative_l2_mean"].clip(lower=1e-12)
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate observable Galilean canonicalization on OOD severity jobs.")
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--out-prefix", default="runs/paper_tables/2d_galilean_n64_observable_canonicalization")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    device = get_device(args.device)
    rows: list[dict[str, Any]] = []
    for job in _jobs(Path(args.matrix)):
        checkpoint = ROOT / str(job["checkpoint"])
        out_dir = ROOT / str(job["out_dir"]) / "observable_canonicalization"
        out_path = out_dir / "metrics.json"
        overwrite = bool(args.overwrite or job.get("overwrite", False))
        print(f"observable canonical eval {checkpoint} -> {out_path}", flush=True)
        if args.dry_run:
            continue
        if out_path.exists() and not overwrite:
            metrics = _read_cached_metrics(job, out_path, ROOT, str(device))
        else:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            cfg = ckpt.get("config", ckpt.get("base_config"))
            if cfg is None:
                raise KeyError(f"Checkpoint missing config/base_config: {checkpoint}")
            metadata = _evaluation_metadata(
                job, cfg["dataset"]["path"], ROOT, str(device),
                expected_dataset_sha256=ckpt.get("meta", {}).get("dataset_sha256"),
            )
            model = build_model(cfg).to(device)
            model.load_state_dict(ckpt["model"])
            dataset, _ = load_tensor_dataset(ROOT / cfg["dataset"]["path"], str(job.get("split", "test")))
            loader = DataLoader(dataset, batch_size=int(job.get("batch_size", cfg.get("training", {}).get("batch_size", 32))), shuffle=False)
            transform = build_transform({"symmetry": job["symmetry"]})
            if not isinstance(transform, NavierStokes2DGalilean):
                raise TypeError(f"Observable canonicalization requires NavierStokes2DGalilean, got {type(transform).__name__}")
            metrics = evaluate_observable_canonicalization(
                model,
                loader,
                device=device,
                transform=transform,
                n_orbit_samples=int(job.get("orbit_samples", 4)),
                seed=int(job.get("eval_seed", int(job.get("seed", 0)) + 700_000)),
            )
            first_batch = next(iter(loader))["a"].to(device)
            training_cfg = cfg.get("training", {})
            latency_repeats = int(job.get("latency_repeats", training_cfg.get("latency_repeats", 20)))
            latency_warmup = int(job.get("latency_warmup", training_cfg.get("latency_warmup", 5)))
            direct_latency = measure_inference_latency(
                model,
                first_batch,
                repeats=latency_repeats,
                warmup=latency_warmup,
            )
            canonical_latency = measure_inference_latency(
                _observable_wrapper(model, cfg, transform),
                first_batch,
                repeats=latency_repeats,
                warmup=latency_warmup,
            )
            metrics.update(
                {
                    f"direct_{key}": value
                    for key, value in direct_latency.items()
                    if key.startswith("latency_")
                }
            )
            metrics.update(
                {
                    f"observable_canonical_{key}": value
                    for key, value in canonical_latency.items()
                    if key.startswith("latency_")
                }
            )
            direct_ms = metrics["direct_latency_ms_per_sample"]
            canonical_ms = metrics["observable_canonical_latency_ms_per_sample"]
            metrics["observable_canonical_latency_overhead_pct"] = 100.0 * (
                canonical_ms - direct_ms
            ) / max(direct_ms, 1e-12)
            _write_metrics(metrics, out_path, metadata)
        row = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "run_dir": str(out_dir.relative_to(ROOT)),
            "method": job.get("method"),
            "seed": job.get("seed"),
            "severity": job.get("severity"),
            "severity_scale": job.get("severity_scale"),
            "max_boost": _symmetry_value(job.get("symmetry", {}), "max_boost"),
            **metrics,
        }
        rows.append(row)

    if args.dry_run:
        return

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(rows).sort_values(["severity_scale", "method", "seed"]) if rows else pd.DataFrame()
    aggregate = _aggregate(rows)
    runs.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        if aggregate.empty:
            f.write("% No observable canonicalization evaluations found.\n")
        else:
            f.write(aggregate.to_latex(index=False, escape=False, float_format=lambda value: f"{value:.4f}"))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}")


if __name__ == "__main__":
    main()
