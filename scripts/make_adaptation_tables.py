#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from otno.config import load_config


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, dict) else {}


def _flatten(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_{key}": value
        for key, value in payload.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _row(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    result = _read_json(path)
    config_path = path.parent / "config.yaml"
    config = load_config(config_path) if config_path.exists() else {}
    adaptation = config.get("adaptation", {})
    row = {
        "run_dir": str(path.parent.relative_to(ROOT)),
        "checkpoint": result.get("checkpoint"),
        "source_method": adaptation.get("source_method"),
        "seed": config.get("seed"),
        "trainable": result.get("trainable", adaptation.get("trainable")),
        "epochs": adaptation.get("epochs"),
        "lr": adaptation.get("lr"),
        "beta_l2_initial": adaptation.get("beta_l2_initial"),
        "orbit_eta": adaptation.get("orbit_eta"),
    }
    row.update(_flatten("before", result.get("before", {})))
    row.update(_flatten("after", result.get("after", {})))
    if "before_relative_l2" in row and "after_relative_l2" in row:
        row["relative_l2_delta_pct"] = 100.0 * (
            float(row["after_relative_l2"]) / max(float(row["before_relative_l2"]), 1e-12) - 1.0
        )
    if "before_orbit_ood_relative_l2" in row and "after_orbit_ood_relative_l2" in row:
        row["orbit_ood_delta_pct"] = 100.0 * (
            float(row["after_orbit_ood_relative_l2"])
            / max(float(row["before_orbit_ood_relative_l2"]), 1e-12)
            - 1.0
        )
    if "before_equivariance_defect_relative" in row and "after_equivariance_defect_relative" in row:
        row["defect_delta_pct"] = 100.0 * (
            float(row["after_equivariance_defect_relative"])
            / max(float(row["before_equivariance_defect_relative"]), 1e-12)
            - 1.0
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect adaptation metrics and write table drafts.")
    parser.add_argument("--runs", default="runs/adaptation")
    parser.add_argument("--out-prefix", default="runs/paper_tables/adaptation")
    parser.add_argument(
        "--group-cols",
        default="source_method,trainable,beta_l2_initial,epochs",
        help="Comma-separated grouping columns for aggregate outputs.",
    )
    args = parser.parse_args()
    runs_root = Path(args.runs)
    if not runs_root.is_absolute():
        runs_root = ROOT / runs_root
    runs_root = runs_root.resolve()
    paths = sorted(runs_root.glob("**/adapt_results.json"))
    rows = [_row(path, runs_root) for path in paths]
    df = pd.DataFrame(rows).sort_values("run_dir") if rows else pd.DataFrame()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_prefix.with_suffix(".runs.csv"), index=False)

    group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]
    missing = sorted(set(group_cols) - set(df.columns)) if not df.empty else []
    if missing:
        raise SystemExit(f"Unknown --group-cols entries: {', '.join(missing)}")
    value_cols = [
        "before_relative_l2",
        "after_relative_l2",
        "relative_l2_delta_pct",
        "before_orbit_ood_relative_l2",
        "after_orbit_ood_relative_l2",
        "orbit_ood_delta_pct",
        "before_equivariance_defect_relative",
        "after_equivariance_defect_relative",
        "defect_delta_pct",
    ]
    if df.empty:
        aggregate = pd.DataFrame()
    else:
        existing_values = [col for col in value_cols if col in df.columns]
        aggregate = (
            df.groupby(group_cols, dropna=False)[existing_values]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    with out_prefix.with_suffix(".aggregate.tex").open("w", encoding="utf-8") as f:
        if aggregate.empty:
            f.write("% No adaptation runs found.\n")
        else:
            f.write(aggregate.to_latex(index=False, escape=False))
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.tex')}")


if __name__ == "__main__":
    main()
