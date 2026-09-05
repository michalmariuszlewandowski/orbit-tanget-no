#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from otno.reporting import RunValidation, collect_run_rows, paired_metric_summary

METRICS = [
    ("force_mae", "Force MAE"),
    ("orbit_ood_force_mae", "Transformed force MAE"),
    ("equivariance_defect_relative", "Eq. defect"),
    ("relative_l2", "ID L2"),
    ("orbit_ood_relative_l2", "Transformed L2"),
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
    if df.empty or "run_dir" not in df:
        raise SystemExit(f"No completed rMD17 runs found under {run_root!r}")
    root = Path(run_root).resolve()
    selected = df["run_dir"].map(lambda value: Path(str(value)).resolve().is_relative_to(root))
    out = df[selected].copy()
    if out.empty:
        raise SystemExit(f"No completed rMD17 runs found under {run_root!r}")
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    validation = RunValidation()
    if "dataset_sha256" in df:
        validation.nonempty_consistent(df, "dataset_sha256", "rMD17 campaign")
    metrics = [metric for metric, _ in METRICS]
    metrics.extend(metric for metric in ("latency_ms_per_sample", "train_wall_seconds") if metric in df)
    rows = []
    for method, group in df.groupby("method", sort=False):
        seeds = validation.seed_values(group, str(method))
        validation.finite_metrics(group, str(method), tuple(metrics))
        row: dict[str, object] = {
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "seed_count": len(seeds),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate["method_order"] = aggregate["method"].map(_method_sort_key)
    return aggregate.sort_values(["method_order", "method"]).drop(columns=["method_order"])


def _paired_rows(df: pd.DataFrame, reference: str = "aug", candidate: str = "aug_orbit") -> pd.DataFrame:
    rows = []
    ref = df[df["method"] == reference]
    cand = df[df["method"] == candidate]
    for metric, label in METRICS:
        summary = paired_metric_summary(ref, cand, metric, context=f"rMD17 {candidate} vs {reference}")
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "reference_method": reference,
                "candidate_method": candidate,
                **summary,
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
            "Method & Force MAE & Transformed force MAE & Eq. defect & ID $L^2$ & Transformed $L^2$ & Seeds \\\\\n"
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
    parser.add_argument(
        "--include-run-dir",
        default=None,
        help="Optional regular expression that selected normalized run paths must fully match.",
    )
    parser.add_argument(
        "--expected-seeds",
        default=None,
        help="Optional comma-separated seed set required exactly once for every selected method.",
    )
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
    if args.include_run_dir:
        normalized_run_dir = df["run_dir"].astype(str).str.replace("\\", "/", regex=False)
        df = df[normalized_run_dir.str.fullmatch(args.include_run_dir, na=False)].copy()
        if df.empty:
            raise SystemExit(
                f"No completed rMD17 runs match --include-run-dir={args.include_run_dir!r}"
            )
    df["method"] = df["method"].astype(str)
    if args.expected_seeds:
        expected_seeds = tuple(
            int(value.strip()) for value in args.expected_seeds.split(",") if value.strip()
        )
        for method, group in df.groupby("method", sort=False):
            RunValidation().seeds(group, str(method), expected_seeds)
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
        "config.training.normalize_by_epsilon",
        "config.training.orbit_eta",
        "config.training.lambda_orbit",
        "dataset_sha256",
        "config_hash",
    ]
    aggregate = _aggregate(df)
    paired = _paired_rows(df)
    _subset_columns(df, runs_columns).to_csv(out_prefix.with_suffix(".runs.csv"), index=False)
    aggregate.to_csv(out_prefix.with_suffix(".aggregate.csv"), index=False)
    paired.to_csv(out_prefix.with_suffix(".paired.csv"), index=False)

    _write_latex_table(out_prefix.with_suffix(".tex"), aggregate, caption=args.caption, label=args.label)
    print(f"wrote {out_prefix.with_suffix('.runs.csv')}")
    print(f"wrote {out_prefix.with_suffix('.aggregate.csv')}")
    print(f"wrote {out_prefix.with_suffix('.paired.csv')}")
    print(f"wrote {out_prefix.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
