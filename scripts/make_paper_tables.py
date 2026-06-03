#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse
import re

import pandas as pd

from otno.reporting import aggregate_runs, collect_run_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect run metrics and write paper-table CSV/LaTeX drafts."
    )
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out-prefix", default="runs/paper_tables/main")
    parser.add_argument(
        "--group-cols",
        default=None,
        help="Comma-separated grouping columns for aggregate outputs.",
    )
    parser.add_argument(
        "--include-run-dir",
        default=None,
        help="Regex filter applied to normalized run_dir paths before aggregation.",
    )
    parser.add_argument(
        "--exclude-run-dir",
        default=None,
        help="Regex filter applied to normalized run_dir paths before aggregation.",
    )
    args = parser.parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_run_rows(args.runs)
    df = pd.DataFrame(rows).sort_values("run_dir") if rows else pd.DataFrame()
    if not df.empty and (args.include_run_dir or args.exclude_run_dir):
        normalized = df["run_dir"].astype(str).str.replace("\\", "/", regex=False)
        if args.include_run_dir:
            include = re.compile(args.include_run_dir)
            df = df[normalized.map(lambda value: bool(include.search(value)))]
            normalized = df["run_dir"].astype(str).str.replace("\\", "/", regex=False)
        if args.exclude_run_dir:
            exclude = re.compile(args.exclude_run_dir)
            df = df[~normalized.map(lambda value: bool(exclude.search(value)))]
    df.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    group_cols = (
        [col.strip() for col in args.group_cols.split(",") if col.strip()]
        if args.group_cols
        else None
    )
    missing_cols = sorted(set(group_cols or []) - set(df.columns)) if not df.empty else []
    if missing_cols:
        raise SystemExit(f"Unknown --group-cols entries: {', '.join(missing_cols)}")
    agg = aggregate_runs(df, group_cols=group_cols)
    agg.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        if agg.empty:
            f.write("% No completed runs found.\n")
        else:
            f.write(agg.to_latex(index=False, escape=False))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}")


if __name__ == "__main__":
    main()
