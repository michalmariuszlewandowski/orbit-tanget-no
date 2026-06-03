#!/usr/bin/env python
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def _save_bar(df: pd.DataFrame, metric: str, out_dir: Path) -> None:
    if metric not in df.columns or "method" not in df.columns:
        return
    plot_df = df.dropna(subset=[metric]).copy()
    if plot_df.empty:
        return
    plot_df["label"] = plot_df.get("run_dir", plot_df["method"]).astype(str)
    fig = plt.figure(figsize=(max(6, 0.35 * len(plot_df)), 4))
    ax = fig.add_subplot(111)
    ax.bar(range(len(plot_df)), plot_df[metric])
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(plot_df["label"], rotation=60, ha="right")
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(out_dir / f"{metric}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple diagnostic plots from a summary CSV.")
    parser.add_argument("--summary", default="runs/summary.csv")
    parser.add_argument("--out-dir", default="runs/figures")
    args = parser.parse_args()
    df = pd.read_csv(args.summary)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric in [
        "relative_l2",
        "orbit_ood_relative_l2",
        "equivariance_defect_relative",
        "latency_ms_per_sample",
    ]:
        _save_bar(df, metric, out_dir)
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
