#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.symmetry.registry import build_transform
from otno.training.metrics import evaluate_model, measure_inference_latency
from otno.utils import dump_json, get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained neural operator checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--orbit-samples", type=int, default=4)
    parser.add_argument("--latency-repeats", type=int, default=None)
    parser.add_argument("--latency-warmup", type=int, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    dataset, _ = load_tensor_dataset(cfg["dataset"]["path"], args.split)
    loader = DataLoader(dataset, batch_size=int(cfg.get("training", {}).get("batch_size", 32)))
    transform = build_transform(cfg.get("symmetry"))
    metrics = evaluate_model(
        model,
        loader,
        device=device,
        transform=transform,
        n_orbit_samples=args.orbit_samples,
    )
    first_batch = next(iter(loader))["a"].to(device)
    training_cfg = cfg.get("training", {})
    repeats = args.latency_repeats if args.latency_repeats is not None else int(training_cfg.get("latency_repeats", 20))
    warmup = args.latency_warmup if args.latency_warmup is not None else int(training_cfg.get("latency_warmup", 5))
    metrics.update(measure_inference_latency(model, first_batch, repeats=repeats, warmup=warmup))
    if args.out:
        dump_json(metrics, args.out)
    for key, value in sorted(metrics.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
