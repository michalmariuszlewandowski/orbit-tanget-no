#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from otno.reporting import collect_run_rows

METRICS = [
    ("force_mae", "Force MAE"),
    ("orbit_ood_force_mae", "OOD force MAE"),
    ("equivariance_defect_relative", "Eq. defect"),
    ("relative_l2", "ID L2"),
    ("orbit_ood_relative_l2", "OOD L2"),
]

METHOD_ORDER = [
    "baseline",
    "aug",
    "aug_orbit_no_output",
    "aug_orbit_shuffle",
    "aug_orbit",
]

METHOD_LABELS = {
    "baseline": "MLP",
    "aug": "MLP + aug.",
    "aug_orbit_no_output": "MLP + aug. + no output",
    "aug_orbit_shuffle": "MLP + aug. + shuffled orbit",
    "aug_orbit": "MLP + aug. + LOCO",
}

T_CRIT_95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636305,
    6: 2.4469118511449692,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
}


def _fmt(value: float) -> str:
    if abs(value) < 0.00005 and value != 0:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _fmt_pm(mean: float, std: float) -> str:
    return rf"\({_fmt(mean)}\pm{_fmt(std)}\)"


def _method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def _filter_runs(df: pd.DataFrame, run_root: str) -> pd.DataFrame:
    normalized = df["run_dir"].astype(str).str.replace("\\", "/", regex=False)
    root = run_root.replace("\\", "/").rstrip("/")
    out = df[normalized.str.contains(root, regex=False)].copy()
    if out.empty:
        raise SystemExit(f"No completed rMD17 runs found under {run_root!r}")
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in df.groupby("method", sort=False):
        row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed_count": int(group["seed"].nunique()),
        }
        for metric, _ in METRICS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        if "latency_ms_per_sample" in group:
            values = pd.to_numeric(group["latency_ms_per_sample"], errors="coerce").dropna()
            row["latency_ms_per_sample_mean"] = float(values.mean())
            row["latency_ms_per_sample_std"] = float(values.std(ddof=1)) if values.shape[0] > 1 else 0.0
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate["method_order"] = aggregate["method"].map(_method_sort_key)
    return aggregate.sort_values(["method_order", "method"]).drop(columns=["method_order"])


def _paired_rows(df: pd.DataFrame, reference: str = "aug", candidate: str = "aug_orbit") -> pd.DataFrame:
    rows = []
    ref = df[df["method"] == reference]
    cand = df[df["method"] == candidate]
    for metric, label in METRICS:
        paired = (
            ref[["seed", metric]]
            .rename(columns={metric: "reference"})
            .merge(cand[["seed", metric]].rename(columns={metric: "candidate"}), on="seed", how="inner")
            .sort_values("seed")
        )
        if paired.empty:
            continue
        diff = paired["candidate"] - paired["reference"]
        n = int(diff.shape[0])
        mean = float(diff.mean())
        std = float(diff.std(ddof=1)) if n > 1 else 0.0
        sem = std / math.sqrt(n) if n else 0.0
        tcrit = T_CRIT_95.get(n - 1, 1.96)
        ref_mean = float(paired["reference"].mean())
        cand_mean = float(paired["candidate"].mean())
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "seed_count": n,
                "reference_method": reference,
                "candidate_method": candidate,
                "reference_mean": ref_mean,
                "candidate_mean": cand_mean,
                "paired_delta_mean": mean,
                "paired_delta_ci95_low": mean - tcrit * sem,
                "paired_delta_ci95_high": mean + tcrit * sem,
                "relative_reduction_pct": 100.0 * (ref_mean - cand_mean) / ref_mean,
            }
        )
    return pd.DataFrame(rows)


def _write_latex_table(path: Path, aggregate: pd.DataFrame, *, caption: str, label: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\small\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{lrrrrrr}\n"
            "\\toprule\n"
            "Method & Force MAE & OOD force MAE & Eq. defect & ID $L^2$ & OOD $L^2$ & Seeds \\\\\n"
            "\\midrule\n"
        )
        for _, row in aggregate.iterrows():
            f.write(
                f"{row['method_label']} & "
                f"{_fmt_pm(row['force_mae_mean'], row['force_mae_std'])} & "
                f"{_fmt_pm(row['orbit_ood_force_mae_mean'], row['orbit_ood_force_mae_std'])} & "
                f"{_fmt_pm(row['equivariance_defect_relative_mean'], row['equivariance_defect_relative_std'])} & "
                f"{_fmt_pm(row['relative_l2_mean'], row['relative_l2_std'])} & "
                f"{_fmt_pm(row['orbit_ood_relative_l2_mean'], row['orbit_ood_relative_l2_std'])} & "
                f"{int(row['seed_count'])} \\\\\n"
            )
        f.write(
            "\\bottomrule\n"
            "\\end{tabular}\n"
            "}\n"
            "\\end{table*}\n"
        )


def _subset_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    return df[[col for col in columns if col in df.columns]].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rMD17 appendix tables from completed runs.")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--run-root", default="runs/appendix/rmd17_ethanol/labels_500")
    parser.add_argument("--out-prefix", default="runs/paper_tables/rmd17_ethanol_500_mechanism")
    parser.add_argument(
        "--caption",
        default=(
            "rMD17 ethanol force-prediction mechanism check at 500 labeled conformations. "
            "Entries are mean plus/minus sample standard deviation over matched seeds."
        ),
    )
    parser.add_argument("--label", default="tab:new-rmd17-ethanol-500")
    args = parser.parse_args()

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(collect_run_rows(args.runs))
    df = _filter_runs(df, args.run_root)
    df["method"] = df["method"].astype(str)
    df["method_order"] = df["method"].map(_method_sort_key)
    df = df.sort_values(["method_order", "seed", "run_dir"]).drop(columns=["method_order"])

    runs_columns = [
        "run_dir",
        "method",
        "seed",
        "force_mae",
        "orbit_ood_force_mae",
        "equivariance_defect_relative",
        "relative_l2",
        "orbit_ood_relative_l2",
        "latency_ms_per_sample",
        "train_wall_seconds",
        "dataset_sha256",
        "config_hash",
    ]
    _subset_columns(df, runs_columns).to_csv(out_prefix.with_suffix(".runs.csv"), index=False)

    aggregate = _aggregate(df)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)

    paired = _paired_rows(df)
    paired.to_csv(out_prefix.with_suffix(".paired.csv"), index=False)

    _write_latex_table(out_prefix.with_suffix(".tex"), aggregate, caption=args.caption, label=args.label)
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.paired.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
