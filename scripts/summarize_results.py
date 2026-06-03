#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import pandas as pd

from otno.reporting import collect_run_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect run_dir/test_metrics.json files into a CSV table.")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="runs/summary.csv")
    args = parser.parse_args()
    rows = collect_run_rows(args.runs)
    df = pd.DataFrame(rows).sort_values("run_dir") if rows else pd.DataFrame()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False) if len(df) else "No metrics found.")


if __name__ == "__main__":
    main()
