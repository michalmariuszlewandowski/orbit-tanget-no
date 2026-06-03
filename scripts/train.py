#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

from otno.config import load_config, parse_overrides, recursive_update
from otno.training.trainer import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a neural operator with optional orbit consistency.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="Override config values: a.b=value")
    args = parser.parse_args()
    cfg = recursive_update(load_config(args.config), parse_overrides(args.override))
    results = train_from_config(cfg)
    for key, value in sorted(results.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
