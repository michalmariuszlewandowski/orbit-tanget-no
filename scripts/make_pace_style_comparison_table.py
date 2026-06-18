#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _method_row(df: pd.DataFrame, method: str) -> pd.DataFrame:
    row = df[df["method"] == method].copy()
    if row.empty:
        raise ValueError(f"Missing method {method!r}")
    return row


def _pct_reduction(new: pd.Series, old: pd.Series) -> pd.Series:
    return 100.0 * (old - new) / old.clip(lower=1e-12)


def build_summary(aggregate: pd.DataFrame) -> pd.DataFrame:
    aug = _method_row(aggregate, "aug_steps_6").set_index("max_boost")
    loco = _method_row(aggregate, "aug_orbit_lambda_0.1_steps_4").set_index("max_boost")
    boosts = sorted(set(aug.index).intersection(set(loco.index)))
    aug = aug.loc[boosts]
    loco = loco.loc[boosts]
    out = pd.DataFrame(
        {
            "max_boost": boosts,
            "aug_direct_ood": aug["orbit_ood_relative_l2_mean"].to_numpy(),
            "loco_direct_ood": loco["orbit_ood_relative_l2_mean"].to_numpy(),
            "aug_observable_canonical_ood": aug[
                "observable_canonical_ood_relative_l2_mean"
            ].to_numpy(),
            "loco_observable_canonical_ood": loco[
                "observable_canonical_ood_relative_l2_mean"
            ].to_numpy(),
            "loco_direct_reduction_vs_aug_pct": _pct_reduction(
                loco["orbit_ood_relative_l2_mean"],
                aug["orbit_ood_relative_l2_mean"],
            ).to_numpy(),
            "loco_observable_reduction_vs_aug_observable_pct": _pct_reduction(
                loco["observable_canonical_ood_relative_l2_mean"],
                aug["observable_canonical_ood_relative_l2_mean"],
            ).to_numpy(),
            "aug_observable_gain_vs_aug_direct_pct": _pct_reduction(
                aug["observable_canonical_ood_relative_l2_mean"],
                aug["orbit_ood_relative_l2_mean"],
            ).to_numpy(),
            "loco_observable_gain_vs_loco_direct_pct": _pct_reduction(
                loco["observable_canonical_ood_relative_l2_mean"],
                loco["orbit_ood_relative_l2_mean"],
            ).to_numpy(),
            "aug_observable_latency_overhead_pct": aug[
                "observable_canonical_latency_overhead_pct_mean"
            ].to_numpy(),
            "loco_observable_latency_overhead_pct": loco[
                "observable_canonical_latency_overhead_pct_mean"
            ].to_numpy(),
            "seeds": aug["orbit_ood_relative_l2_count"].astype(int).to_numpy(),
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a compact LOCO vs PACE-style observable canonicalization table."
    )
    parser.add_argument(
        "--aggregate",
        default="runs/paper_tables/2d_galilean_n64_direct_pace_style_eval_fast.aggregate.csv",
    )
    parser.add_argument(
        "--out-prefix",
        default="runs/paper_tables/2d_galilean_n64_direct_pace_style_summary",
    )
    args = parser.parse_args()
    aggregate = pd.read_csv(args.aggregate)
    summary = build_summary(aggregate)
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_prefix.with_suffix(".csv"), index=False)
    summary.to_latex(
        out_prefix.with_suffix(".tex"),
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.4f}",
    )
    print(f"wrote {out_prefix.with_suffix('.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
