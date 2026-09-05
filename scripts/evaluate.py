#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import torch
from torch.utils.data import DataLoader

from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.training.metrics import evaluate_model, measure_inference_latency
from otno.utils import count_parameters, dump_json, file_sha256, get_device, make_torch_generator, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained neural operator checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset", default=None, help="Relocated dataset; its SHA-256 must match the checkpoint.")
    parser.add_argument("--orbit-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the checkpoint's test evaluation seed.")
    parser.add_argument("--latency-repeats", type=int, default=None)
    parser.add_argument("--latency-warmup", type=int, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    training_cfg = cfg.get("training", {})
    runtime_cfg = cfg.get("runtime", {})
    set_seed(int(cfg.get("seed", 0)), deterministic=bool(runtime_cfg.get("deterministic", False)))
    eval_seed = args.seed if args.seed is not None else int(training_cfg.get("test_eval_seed", int(cfg.get("seed", 0)) + 200_000))
    orbit_samples = args.orbit_samples if args.orbit_samples is not None else int(training_cfg.get("eval_orbit_samples", 4))
    dataset_path = Path(args.dataset or cfg["dataset"]["path"])
    dataset_sha256 = file_sha256(dataset_path)
    expected_sha256 = ckpt.get("meta", {}).get("dataset_sha256")
    if expected_sha256 and dataset_sha256 != expected_sha256:
        raise ValueError(f"Dataset SHA-256 does not match checkpoint: {dataset_path}")
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    dataset, _ = load_tensor_dataset(dataset_path, args.split)
    split_offset = {"train": 0, "val": 10_000, "test": 20_000}.get(args.split, 30_000)
    loader = DataLoader(dataset, batch_size=int(training_cfg.get("batch_size", 32)),
                        generator=make_torch_generator(int(cfg.get("seed", 0)) + split_offset))
    transform = build_transform(cfg.get("symmetry"))
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        transform=transform,
        n_orbit_samples=orbit_samples,
        seed=eval_seed,
    )
    first_batch = next(iter(loader))["a"].to(device)
    repeats = args.latency_repeats if args.latency_repeats is not None else int(training_cfg.get("latency_repeats", 20))
    warmup = args.latency_warmup if args.latency_warmup is not None else int(training_cfg.get("latency_warmup", 5))
    metrics.update(measure_inference_latency(model, first_batch, repeats=repeats, warmup=warmup))
    metrics.update({
        "best_val_relative_l2": ckpt.get("best_val"),
        "parameters": count_parameters(model),
        "method": training_cfg.get("method", "baseline"),
        "dataset_sha256": dataset_sha256,
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "eval_seed": eval_seed,
        "eval_orbit_samples": orbit_samples,
        "split": args.split,
    })
    if args.out:
        dump_json(metrics, args.out)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
