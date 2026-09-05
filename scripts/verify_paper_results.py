#!/usr/bin/env python
"""Recompute v11 table statistics from the released per-seed CSV records."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def verify(root: Path, reference: Path) -> tuple[int, int]:
    spec = yaml.safe_load(reference.read_text(encoding="utf-8"))
    rows_checked = 0
    values_checked = 0
    for table in spec["tables"]:
        data = pd.read_csv(root / table["csv"])
        metrics = table.get("metrics", spec["metrics"])
        for row in table["rows"]:
            selected = data
            for column, value in row["select"].items():
                selected = selected[selected[column] == value]
            seeds = row.get("seeds", table.get("seeds", spec["seeds"]))
            label = f"{table['table']} {row['select']}"
            if sorted(selected["seed"].tolist()) != sorted(seeds):
                raise ValueError(f"{label}: missing, duplicate, or unexpected seeds")
            if "parameters" in table and not (selected["parameters"] == table["parameters"]).all():
                raise ValueError(f"{label}: parameter count mismatch")
            if selected["dataset_sha256"].isna().any() or selected["dataset_sha256"].nunique() != 1:
                raise ValueError(f"{label}: missing or inconsistent dataset hashes")
            if selected["config_hash"].isna().any():
                raise ValueError(f"{label}: missing config hashes")
            if len(row["values"]) != len(metrics):
                raise ValueError(f"{label}: reference metric count mismatch")
            for metric, expected in zip(metrics, row["values"]):
                values = selected[metric].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"{label}: non-finite {metric}")
                actual = (float(values.mean()), float(values.std(ddof=1)))
                for statistic, observed, printed in zip(("mean", "std"), actual, expected):
                    # Expected values are strings so the manuscript's precision is explicit.
                    mantissa = printed.lower().split("e")[0]
                    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
                    exponent = int(printed.lower().split("e")[1]) if "e" in printed.lower() else 0
                    tolerance = 0.50001 * 10.0 ** (exponent - decimals)
                    if abs(observed - float(printed)) > tolerance:
                        raise ValueError(f"{label}: {metric} {statistic}={observed:.10g}, PDF={printed}")
                    values_checked += 1
            rows_checked += 1
    return rows_checked, values_checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--reference", default="figure_specs/paper_results_v11.yaml")
    args = parser.parse_args()
    root = args.root.resolve()
    rows, values = verify(root, root / args.reference)
    print(f"PAPER VALUES VERIFIED: {rows} rows, {values} mean/std values; exact seed sets and provenance checked")
    print("This verifies cached statistics; it does not rerun training or verify unlisted paper tables.")


if __name__ == "__main__":
    main()
