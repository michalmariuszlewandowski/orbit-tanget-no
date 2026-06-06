#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.training.metrics import evaluate_model
from otno.utils import dump_json, get_device


def _format_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [_format_value(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _format_value(item, context) for key, item in value.items()}
    return value


def _jobs(matrix_path: Path) -> list[dict[str, Any]]:
    with matrix_path.open("r", encoding="utf-8") as f:
        matrix = yaml.safe_load(f) or {}
    jobs: list[dict[str, Any]] = []
    for entry in matrix.get("evaluations", []):
        seeds = entry.get("seeds", [None])
        methods = entry.get("methods", [None])
        if not isinstance(seeds, list):
            seeds = [seeds]
        if not isinstance(methods, list):
            methods = [methods]
        for method in methods:
            for seed in seeds:
                context = dict(entry.get("format", {}))
                if method is not None:
                    context["method"] = method
                if seed is not None:
                    context["seed"] = int(seed)
                job = _format_value(dict(entry), context)
                job.pop("methods", None)
                job.pop("seeds", None)
                job.pop("format", None)
                if method is not None:
                    job["method"] = method
                if seed is not None:
                    job["seed"] = int(seed)
                jobs.append(job)
    return jobs


def _aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    group_cols = ["severity", "severity_scale", "method"]
    value_cols = [
        "relative_l2",
        "orbit_ood_relative_l2",
        "equivariance_defect_relative",
        "epsilon_mean",
    ]
    existing_values = [col for col in value_cols if col in df.columns]
    return (
        df.groupby(group_cols, dropna=False)[existing_values]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _symmetry_value(symmetry: dict[str, Any], key: str) -> Any:
    if key in symmetry:
        return symmetry.get(key)
    for transform in symmetry.get("transforms", []):
        if isinstance(transform, dict) and key in transform:
            return transform.get(key)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate checkpoints under OOD symmetry severity sweeps.")
    parser.add_argument("--matrix", required=True, help="Evaluation matrix YAML file")
    parser.add_argument("--out-prefix", default="runs/paper_tables/ood_severity")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    rows: list[dict[str, Any]] = []
    for job in _jobs(Path(args.matrix)):
        checkpoint = ROOT / str(job["checkpoint"])
        out_dir = ROOT / str(job["out_dir"])
        out_path = out_dir / "severity_metrics.json"
        overwrite = bool(job.get("overwrite", False))
        print(f"eval {checkpoint} -> {out_path}", flush=True)
        if args.dry_run:
            continue
        if out_path.exists() and not overwrite:
            metrics = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
        else:
            if not checkpoint.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            cfg = ckpt.get("config", ckpt.get("base_config"))
            if cfg is None:
                raise KeyError(f"Checkpoint missing config/base_config: {checkpoint}")
            model = build_model(cfg).to(device)
            model.load_state_dict(ckpt["model"])
            split = str(job.get("split", "test"))
            dataset, _ = load_tensor_dataset(cfg["dataset"]["path"], split)
            batch_size = int(job.get("batch_size", cfg.get("training", {}).get("batch_size", 32)))
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            transform = build_transform({"symmetry": job["symmetry"]})
            metrics = evaluate_model(
                model,
                loader,
                device=device,
                transform=transform,
                n_orbit_samples=int(job.get("orbit_samples", 8)),
                seed=int(job.get("eval_seed", int(job.get("seed", 0)) + 700_000)),
            )
            dump_json(metrics, out_path)
        row = {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "run_dir": str(out_dir.relative_to(ROOT)),
            "method": job.get("method"),
            "seed": job.get("seed"),
            "severity": job.get("severity"),
            "severity_scale": job.get("severity_scale"),
            "max_shift": _symmetry_value(job.get("symmetry", {}), "max_shift"),
            "max_boost": _symmetry_value(job.get("symmetry", {}), "max_boost"),
            **metrics,
        }
        rows.append(row)

    runs = pd.DataFrame(rows).sort_values(["severity_scale", "method", "seed"]) if rows else pd.DataFrame()
    runs.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    aggregate = _aggregate(rows)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        if aggregate.empty:
            f.write("% No severity evaluations found.\n")
        else:
            f.write(aggregate.to_latex(index=False, escape=False))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}")


if __name__ == "__main__":
    main()
