#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path("runs/external/lpsda_kdv_galilean")
CLOSE_ROOT = Path("runs/external/lpsda_kdv_galilean_close/grid")
OUT_PREFIX = Path("runs/paper_tables/lpsda_kdv_galilean_pilot")
SEEDS = [23, 31, 47, 59, 71]


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _tex_num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    return rf"\({value:.{digits}f}\)"


def _tex_pm(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None:
        return "--"
    if std is None:
        return _tex_num(mean, digits=digits)
    return rf"\({mean:.{digits}f}\pm{std:.{digits}f}\)"


def main() -> None:
    run_groups = [
        (
            "ours (0.56M)",
            ROOT,
            [
                ("FNO", "baseline"),
                ("FNO + aug.", "aug"),
                (r"FNO + LOCO, \(\lambda=0.20\)", "orbit_lambda_0p2"),
                (r"FNO + aug. + LOCO, \(\lambda=0.20\)", "aug_orbit_lambda_0p2"),
            ],
        ),
        (
            "ours (1.09M)",
            CLOSE_ROOT,
            [
                ("FNO", "baseline"),
                (r"FNO + LOCO, \(\lambda=0.10\)", "orbit_lambda_0p10"),
            ],
        ),
    ]
    aggregate_rows = [
        {
            "source": "LPSDA Table 3",
            "method": "Published FNO(AR)",
            "samples": 64,
            "seeds": None,
            "trajectory_nmse": 0.0604,
            "trajectory_nmse_std": 0.0036,
            "orbit_ood_trajectory_nmse": None,
            "orbit_ood_trajectory_nmse_std": None,
            "equivariance_defect_relative": None,
            "equivariance_defect_relative_std": None,
            "latency_ms_per_sample": None,
            "latency_ms_per_sample_std": None,
        },
        {
            "source": "LPSDA Table 3",
            "method": "Published FNO(AR)+LPSDA",
            "samples": 64,
            "seeds": None,
            "trajectory_nmse": 0.0387,
            "trajectory_nmse_std": 0.0025,
            "orbit_ood_trajectory_nmse": None,
            "orbit_ood_trajectory_nmse_std": None,
            "equivariance_defect_relative": None,
            "equivariance_defect_relative_std": None,
            "latency_ms_per_sample": None,
            "latency_ms_per_sample_std": None,
        },
        {
            "source": "LPSDA Table 3",
            "method": "Published FNO(NO)",
            "samples": 64,
            "seeds": None,
            "trajectory_nmse": 0.1699,
            "trajectory_nmse_std": 0.0100,
            "orbit_ood_trajectory_nmse": None,
            "orbit_ood_trajectory_nmse_std": None,
            "equivariance_defect_relative": None,
            "equivariance_defect_relative_std": None,
            "latency_ms_per_sample": None,
            "latency_ms_per_sample_std": None,
        },
        {
            "source": "LPSDA Table 3",
            "method": "Published FNO(NO)+LPSDA",
            "samples": 64,
            "seeds": None,
            "trajectory_nmse": 0.0574,
            "trajectory_nmse_std": 0.0045,
            "orbit_ood_trajectory_nmse": None,
            "orbit_ood_trajectory_nmse_std": None,
            "equivariance_defect_relative": None,
            "equivariance_defect_relative_std": None,
            "latency_ms_per_sample": None,
            "latency_ms_per_sample_std": None,
        },
    ]
    run_rows = []
    for source, root, run_specs in run_groups:
        for method, run_name in run_specs:
            for seed in SEEDS:
                path = root / run_name / f"seed_{seed}" / "test_metrics.json"
                metrics = _load_metrics(path)
                run_rows.append(
                    {
                        "source": source,
                        "method": method,
                        "seed": seed,
                        "samples": int(metrics["num_samples"]),
                        "trajectory_nmse": float(metrics["trajectory_nmse"]),
                        "orbit_ood_trajectory_nmse": float(metrics["orbit_ood_trajectory_nmse"]),
                        "equivariance_defect_relative": float(metrics["equivariance_defect_relative"]),
                        "latency_ms_per_sample": float(metrics["latency_ms_per_sample"]),
                        "run_dir": str(path.parent),
                    }
                )
    run_df = pd.DataFrame(run_rows)
    for source, _, run_specs in run_groups:
        for method, _ in run_specs:
            method_runs = run_df[(run_df["source"] == source) & (run_df["method"] == method)]
            row = {
                "source": source,
                "method": method,
                "samples": int(method_runs["samples"].iloc[0]),
                "seeds": int(len(method_runs)),
            }
            for metric in [
                "trajectory_nmse",
                "orbit_ood_trajectory_nmse",
                "equivariance_defect_relative",
                "latency_ms_per_sample",
            ]:
                row[metric] = float(method_runs[metric].mean())
                row[f"{metric}_std"] = float(method_runs[metric].std(ddof=1))
            aggregate_rows.append(row)
    df = pd.DataFrame(aggregate_rows)
    OUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PREFIX.with_suffix(".csv"), index=False)
    run_df.to_csv(OUT_PREFIX.with_suffix(".runs.csv"), index=False)
    with OUT_PREFIX.with_suffix(".tex").open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{LPSDA-style KdV Galilean pilot at 64 training samples. "
            "Published rows are the KdV 20s FNO references from Brandstetter et al., Table 3; "
            "AR denotes their autoregressive solver and NO denotes their neural-operator rollout. "
            "Our rows use five training seeds on a fixed 64-sample KdV dataset; the 1.09M-parameter "
            "rows use a closer FNO with five Fourier blocks, 32 modes, width 80, and an internal grid coordinate. "
            "All rows report mean $\\pm$ "
            "sample standard deviation for the LPSDA normalized MSE, Galilean orbit-OOD NMSE, and "
            "equivariance defect; the Galilean frame is not an input channel.}\n"
            "\\label{tab:lpsda-kdv-galilean-pilot}\n"
            "\\begin{tabular}{llrrrrr}\n"
            "\\toprule\n"
            "Source & Method & Seeds & ID NMSE & Orbit NMSE & Eq. defect & Latency ms/sample \\\\\n"
            "\\midrule\n"
        )
        for row in aggregate_rows:
            f.write(
                f"{row['source']} & {row['method']} & "
                f"{'--' if row['seeds'] is None else int(row['seeds'])} & "
                f"{_tex_pm(row['trajectory_nmse'], row['trajectory_nmse_std'])} & "
                f"{_tex_pm(row['orbit_ood_trajectory_nmse'], row['orbit_ood_trajectory_nmse_std'])} & "
                f"{_tex_pm(row['equivariance_defect_relative'], row['equivariance_defect_relative_std'])} & "
                f"{_tex_pm(row['latency_ms_per_sample'], row['latency_ms_per_sample_std'], digits=2)} \\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(df.to_string(index=False))
    print(f"wrote {OUT_PREFIX.with_suffix('.csv')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.runs.csv')}")
    print(f"wrote {OUT_PREFIX.with_suffix('.tex')}")


if __name__ == "__main__":
    main()
