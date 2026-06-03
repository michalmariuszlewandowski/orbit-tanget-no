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

from otno.training.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a tiny dataset and run a CPU smoke training job.")
    parser.add_argument("--work-dir", default="runs/smoke")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    work = Path(args.work_dir)
    cfg = {
        "seed": 7,
        "dataset": {
            "kind": "advection1d",
            "path": str(work / "data" / "advection1d_tiny.pt"),
            "generate_if_missing": True,
            "n": 16,
            "num_train": 4,
            "num_val": 2,
            "num_test": 2,
            "final_time": 0.25,
            "velocity": 0.7,
            "modes": 4,
            "amplitude": 0.8,
            "seed": 7,
        },
        "model": {
            "name": "fno1d",
            "in_channels": 1,
            "out_channels": 1,
            "width": 4,
            "modes": 3,
            "depth": 1,
            "add_grid": True,
        },
        "symmetry": {"name": "translation1d", "max_shift": 0.1},
        "training": {
            "method": "aug_orbit",
            "epochs": 1,
            "batch_size": 2,
            "lr": 0.002,
            "lambda_orbit": 0.05,
            "lambda_aug": 0.5,
            "eval_every": 1,
            "eval_orbit_samples": 1,
            "latency_repeats": 2,
            "latency_warmup": 1,
            "num_workers": 0,
        },
        "runtime": {"device": args.device, "run_dir": str(work / "run")},
    }
    results = train_from_config(cfg)
    print(results)


if __name__ == "__main__":
    main()
