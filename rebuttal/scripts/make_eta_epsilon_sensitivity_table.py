#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from otno.reporting import aggregate_runs, collect_run_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate N64 Galilean eta/epsilon sensitivity runs.")
    parser.add_argument(
        "--run-root",
        default="runs/ablations/2d_galilean_n64_2pct_eta_epsilon_sensitivity",
    )
    parser.add_argument(
        "--out-prefix",
        default="runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity",
    )
    args = parser.parse_args()

    rows = collect_run_rows(args.run_root)
    if not rows:
        raise SystemExit(f"No completed test_metrics.json files found under {args.run_root}")
    df = pd.DataFrame(rows)
    df["orbit_eta"] = df["config.training.orbit_eta"].astype(float)
    df["step_max_boost"] = df["config.symmetry.max_boost"].astype(float)
    keep = [
        "run_dir",
        "seed",
        "method",
        "orbit_eta",
        "step_max_boost",
        "epsilon_mean",
        "relative_l2",
        "orbit_ood_relative_l2",
        "equivariance_defect_relative",
        "latency_ms_per_sample",
        "parameters",
    ]
    run_df = df[[col for col in keep if col in df.columns]].sort_values(
        ["orbit_eta", "step_max_boost", "seed"]
    )
    aggregate = aggregate_runs(
        run_df,
        group_cols=["method", "orbit_eta", "step_max_boost"],
    )
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    run_df.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        f.write(aggregate.to_latex(index=False, escape=False))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}")


if __name__ == "__main__":
    main()
