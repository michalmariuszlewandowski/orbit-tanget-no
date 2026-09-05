#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml
from scipy import stats


SEEDS = (23, 31, 47, 59, 71)
PUBLISHED_ROWS = (
    ("FNO (AR), no augmentation", "none", 0.1248, 0.0108),
    ("FNO (AR) + LPSDA (g1g2g3g4)", "g1g2g3g4", 0.0606, 0.0040),
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _rollout_time_steps(config: dict[str, Any]) -> int:
    dataset = config["dataset"]
    model = config["model"]
    nt_effective = int(dataset["nt_effective"])
    time_history = int(model["time_history"])
    time_future = int(model["time_future"])
    max_start_time = nt_effective - time_history - time_future
    return max(1, len(range(time_history, max_start_time + 1, time_future)) * time_future)


def _paper_nmse(metrics: dict[str, Any], rollout_steps: int) -> float:
    value = float(metrics["trajectory_nmse"])
    if "trajectory_nmse_cumulative" in metrics:
        return value
    return value / rollout_steps


def _optional_float(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def _local_run_rows(
    *, root: Path, method: str, method_label: str, expect_orbit: bool
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        run_dir = root / f"seed_{seed}"
        metrics_path = run_dir / "test_metrics.json"
        config_path = run_dir / "config.yaml"
        if not metrics_path.exists() or not config_path.exists():
            raise FileNotFoundError(f"incomplete seed {seed}: expected {metrics_path} and {config_path}")

        metrics = _load_json(metrics_path)
        config = _load_yaml(config_path)
        observed_method = str(metrics.get("method", ""))
        if observed_method != method:
            raise ValueError(
                f"{metrics_path}: expected method={method!r}, observed {observed_method!r}"
            )
        if int(metrics.get("seed", -1)) != seed:
            raise ValueError(f"{metrics_path}: seed does not match directory")
        if int(metrics.get("epochs", -1)) != 20:
            raise ValueError(f"{metrics_path}: expected a complete 20-epoch run")

        rollout_steps = int(metrics.get("rollout_time_steps") or _rollout_time_steps(config))
        orbit_ood = _optional_float(metrics, "orbit_ood_trajectory_nmse")
        equivariance_defect = _optional_float(metrics, "equivariance_defect_relative")
        if expect_orbit and (orbit_ood is None or equivariance_defect is None):
            raise ValueError(f"{metrics_path}: completed LOCO row is missing orbit metrics")

        training = config["training"]
        rows.append(
            {
                "evidence_scope": "local_independent_lpsda_aligned_protocol",
                "source": "Ours (independent protocol)",
                "method": method_label,
                "task": "KdV (40s)",
                "rollout_mode": "FNO (AR)",
                "symmetry_scope": "g3 Galilean only" if expect_orbit else "none",
                "uncertainty_scope": "sample SD across five training seeds",
                "seed": seed,
                "training_samples": int(config["dataset"]["train_samples"]),
                "epochs": int(metrics["epochs"]),
                "rollout_time_steps": rollout_steps,
                "trajectory_nmse": _paper_nmse(metrics, rollout_steps),
                "orbit_ood_trajectory_nmse": orbit_ood if expect_orbit else None,
                "equivariance_defect_relative": equivariance_defect if expect_orbit else None,
                "latency_ms_per_sample": float(metrics["latency_ms_per_sample"]),
                "parameters": int(metrics["parameters"]),
                "train_wall_seconds": float(metrics["train_wall_seconds"]),
                "dataset_sha256": str(metrics["dataset_sha256"]),
                "config_hash": str(metrics["config_hash"]),
                "orbit_eval_samples": int(metrics.get("orbit_eval_samples", 0))
                if expect_orbit
                else 0,
                "orbit_eval_seed": training.get("eval_orbit_seed") if expect_orbit else None,
                "metrics_path": str(metrics_path),
            }
        )
    return rows


def _mean(values: list[float]) -> float:
    return float(mean(values))


def _std(values: list[float]) -> float:
    return float(stdev(values))


def _aggregate_local(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dataset_hashes = {row["dataset_sha256"] for row in rows}
    parameters = {row["parameters"] for row in rows}
    if len(dataset_hashes) != 1 or len(parameters) != 1:
        raise ValueError("local aggregate mixes datasets or parameter counts")

    result: dict[str, Any] = {
        "evidence_scope": rows[0]["evidence_scope"],
        "source": rows[0]["source"],
        "method": rows[0]["method"],
        "task": rows[0]["task"],
        "rollout_mode": rows[0]["rollout_mode"],
        "symmetry_scope": rows[0]["symmetry_scope"],
        "uncertainty_scope": rows[0]["uncertainty_scope"],
        "training_samples": rows[0]["training_samples"],
        "seeds": len(rows),
        "parameters": rows[0]["parameters"],
        "trajectory_nmse": _mean([row["trajectory_nmse"] for row in rows]),
        "trajectory_nmse_std": _std([row["trajectory_nmse"] for row in rows]),
        "latency_ms_per_sample": _mean([row["latency_ms_per_sample"] for row in rows]),
        "latency_ms_per_sample_std": _std(
            [row["latency_ms_per_sample"] for row in rows]
        ),
        "train_wall_hours": _mean([row["train_wall_seconds"] / 3600.0 for row in rows]),
        "train_wall_hours_std": _std(
            [row["train_wall_seconds"] / 3600.0 for row in rows]
        ),
        "orbit_evaluated_seeds": sum(
            row["orbit_ood_trajectory_nmse"] is not None for row in rows
        ),
        "comparison_note": "matched local five-seed training",
    }
    for metric in ("orbit_ood_trajectory_nmse", "equivariance_defect_relative"):
        values = [float(row[metric]) for row in rows if row[metric] is not None]
        result[metric] = _mean(values) if len(values) == len(rows) else None
        result[f"{metric}_std"] = _std(values) if len(values) == len(rows) else None
    return result


def _published_aggregate_rows() -> list[dict[str, Any]]:
    rows = []
    for method, symmetry_scope, value, uncertainty in PUBLISHED_ROWS:
        rows.append(
            {
                "evidence_scope": "published_reference_not_locally_rerun",
                "source": "Published reference (Table 3)",
                "method": method,
                "task": "KdV (40s)",
                "rollout_mode": "FNO (AR)",
                "symmetry_scope": symmetry_scope,
                "uncertainty_scope": "published bootstrap +/-2 SD interval",
                "training_samples": 64,
                "seeds": None,
                "parameters": None,
                "trajectory_nmse": value,
                "trajectory_nmse_std": uncertainty,
                "orbit_ood_trajectory_nmse": None,
                "orbit_ood_trajectory_nmse_std": None,
                "equivariance_defect_relative": None,
                "equivariance_defect_relative_std": None,
                "latency_ms_per_sample": None,
                "latency_ms_per_sample_std": None,
                "train_wall_hours": None,
                "train_wall_hours_std": None,
                "orbit_evaluated_seeds": 0,
                "comparison_note": "literature value; not a local rerun",
            }
        )
    return rows


def _paired_summary(
    baseline_rows: list[dict[str, Any]], loco_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_seed = {row["seed"]: row for row in baseline_rows}
    loco_by_seed = {row["seed"]: row for row in loco_rows}
    paired_rows = []
    for seed in SEEDS:
        baseline = float(baseline_by_seed[seed]["trajectory_nmse"])
        loco = float(loco_by_seed[seed]["trajectory_nmse"])
        paired_rows.append(
            {
                "seed": seed,
                "baseline_trajectory_nmse": baseline,
                "loco_trajectory_nmse": loco,
                "loco_minus_baseline": loco - baseline,
            }
        )

    deltas = [row["loco_minus_baseline"] for row in paired_rows]
    delta_mean = _mean(deltas)
    interval = stats.t.interval(
        0.95,
        len(deltas) - 1,
        loc=delta_mean,
        scale=stats.sem(deltas),
    )
    baseline_mean = _mean([row["baseline_trajectory_nmse"] for row in paired_rows])
    loco_mean = _mean([row["loco_trajectory_nmse"] for row in paired_rows])
    summary = {
        "seeds": len(deltas),
        "baseline_mean": baseline_mean,
        "loco_mean": loco_mean,
        "paired_delta_mean": delta_mean,
        "paired_delta_std": _std(deltas),
        "paired_delta_ci95_low": float(interval[0]),
        "paired_delta_ci95_high": float(interval[1]),
        "paired_ttest_pvalue": float(
            stats.ttest_rel(
                [row["loco_trajectory_nmse"] for row in paired_rows],
                [row["baseline_trajectory_nmse"] for row in paired_rows],
            ).pvalue
        ),
        "relative_delta_percent": 100.0 * delta_mean / baseline_mean,
    }
    return paired_rows, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _tex_pm(value: float | None, uncertainty: float | None, digits: int = 4) -> str:
    if value is None:
        return "--"
    if uncertainty is None:
        return rf"\({value:.{digits}f}\)"
    return rf"\({value:.{digits}f}\pm{uncertainty:.{digits}f}\)"


def _write_tex(path: Path, aggregate_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\small\n"
            "\\caption{Published-reference comparison on the 64-trajectory KdV 40s FNO(AR) "
            "benchmark. Published values are transcribed from Brandstetter et al. (2022), Table 3, "
            "and were not rerun locally. Local rows use five paired training seeds of an independent "
            "implementation aligned to the released 20-to-20-step FNO code. LOCO applies Galilean "
            "consistency only ($g_3$, $\\lambda_{\\mathrm{orb}}=0.05$, "
            "$|\\epsilon|\\leq0.2$); published LPSDA applies $g_1g_2g_3g_4$. Published uncertainty "
            "is the reported bootstrap $\\pm2$ SD interval, whereas local uncertainty is sample "
            "standard deviation across training seeds. Orbit-OOD and equivariance-defect values are "
            "reported only where all five comparable evaluations exist. Dashes mean unavailable, "
            "not zero. The published and local blocks are protocol-aligned references, not a "
            "controlled method-only comparison.}\n"
            "\\label{tab:lpsda-kdv-40s-published-reference}\n"
            "\\resizebox{\\textwidth}{!}{%\n"
            "\\begin{tabular}{llrrrrrrr}\n"
            "\\toprule\n"
            "Evidence & Method & Seeds & Params (M) & ID NMSE & Orbit-OOD NMSE & Eq. defect & "
            "Train h & Latency ms/sample \\\\\n"
            "\\midrule\n"
        )
        for index, row in enumerate(aggregate_rows):
            if index == len(PUBLISHED_ROWS):
                f.write("\\midrule\n")
            source = (
                "Published ref."
                if row["evidence_scope"] == "published_reference_not_locally_rerun"
                else "Ours (independent)"
            )
            seeds = "--" if row["seeds"] is None else str(row["seeds"])
            params = "--" if row["parameters"] is None else f"{row['parameters'] / 1e6:.2f}"
            method = row["method"]
            if method == "FNO (AR) + LPSDA (g1g2g3g4)":
                method = r"FNO (AR) + LPSDA, \(g_1g_2g_3g_4\)"
            if method == "FNO (AR) + LOCO (g3 only; lambda=0.05)":
                method = r"FNO (AR) + LOCO, \(g_3\), \(\lambda=0.05\)"
            f.write(
                f"{source} & {method} & {seeds} & {params} & "
                f"{_tex_pm(row['trajectory_nmse'], row['trajectory_nmse_std'])} & "
                f"{_tex_pm(row['orbit_ood_trajectory_nmse'], row['orbit_ood_trajectory_nmse_std'])} & "
                f"{_tex_pm(row['equivariance_defect_relative'], row['equivariance_defect_relative_std'])} & "
                f"{_tex_pm(row['train_wall_hours'], row['train_wall_hours_std'], digits=2)} & "
                f"{_tex_pm(row['latency_ms_per_sample'], row['latency_ms_per_sample_std'], digits=2)} "
                "\\\\\n"
            )
        f.write("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n")


def _write_interpretation(
    path: Path,
    baseline: dict[str, Any],
    loco: dict[str, Any],
    paired: dict[str, Any],
) -> None:
    published_fno_ar = PUBLISHED_ROWS[0][2]
    published_lpsda_ar = PUBLISHED_ROWS[1][2]
    loco_vs_published_ratio = float(loco["trajectory_nmse"]) / published_lpsda_ar
    baseline_vs_published_reduction = (
        100.0 * (published_fno_ar - float(baseline["trajectory_nmse"])) / published_fno_ar
    )
    published_lpsda_reduction = (
        100.0 * (published_fno_ar - published_lpsda_ar) / published_fno_ar
    )
    training_ratio = float(loco["train_wall_hours"]) / float(baseline["train_wall_hours"])
    text = f"""# LPSDA-aligned KdV comparison: interpretation

## Evidence being compared

- Primary published source: https://proceedings.mlr.press/v162/brandstetter22a/brandstetter22a.pdf
- The two **published-reference** rows are the 64-sample KdV 40s FNO(AR) entries copied from Brandstetter et al., Table 3. They are not local reruns. Their reported uncertainty is a bootstrap $\\pm2$ SD interval, not an across-training-seed standard deviation.
- The two **ours (independent protocol)** rows are five matched local seeds on one cached dataset. Both use the same 10,856,084-parameter FNO. LOCO changes training only.
- The local implementation is aligned to the released 20-to-20-step autoregressive FNO protocol, not a bitwise reproduction. Local uncertainty is the sample standard deviation across five training seeds.
- Local LOCO constrains only the Galilean generator $g_3$ with $|\\epsilon|\\leq0.2$. Published full LPSDA augments with $g_1g_2g_3g_4$. Cross-block comparisons are therefore contextual rather than controlled method-only comparisons.

## Supported interpretation

- Against the matched local FNO calibration, LOCO changes ID NMSE from {baseline['trajectory_nmse']:.4f} to {loco['trajectory_nmse']:.4f}. The paired mean change (LOCO minus FNO) is {paired['paired_delta_mean']:.6f}, or {paired['relative_delta_percent']:.2f}%. Its two-sided 95% paired t-interval is [{paired['paired_delta_ci95_low']:.6f}, {paired['paired_delta_ci95_high']:.6f}] (p={paired['paired_ttest_pvalue']:.3f}). With five seeds, this is consistent with preserved ID accuracy, not evidence of an ID improvement.
- The local LOCO absolute Galilean metrics are {loco['orbit_ood_trajectory_nmse']:.4f} orbit-OOD NMSE and {loco['equivariance_defect_relative']:.4f} equivariance defect. No matched five-seed local FNO or published LPSDA values exist for these metrics, so this benchmark cannot establish a symmetry-robustness gain over either comparator.
- Published full LPSDA reduces ID NMSE by {published_lpsda_reduction:.1f}% within the source paper ({published_fno_ar:.4f} to {published_lpsda_ar:.4f}). Local LOCO ID NMSE is {loco_vs_published_ratio:.2f}x the published full-LPSDA value. Current Galilean-only LOCO therefore does not reproduce the published full-symmetry LPSDA label-efficiency gain.
- The local FNO calibration is {baseline_vs_published_reduction:.1f}% below the correct published FNO(AR) baseline and lies inside its reported interval. This supports broad protocol alignment, while the independent implementation and different uncertainty estimands still preclude a controlled cross-block method attribution.
- The published 40s augmentation ladder is informative but not a Galilean-only control: $g_1$ gives 0.0674, $g_1g_2$ gives 0.0673, $g_1g_2g_3$ gives 0.0550, and $g_1g_2g_3g_4$ gives 0.0606. Much of the gain appears before adding $g_3$, so the table cannot isolate LOCO versus supervised augmentation for the same generator.
- FNO and LOCO have identical parameter counts and inference graphs. Their measured latency means ({baseline['latency_ms_per_sample']:.2f} and {loco['latency_ms_per_sample']:.2f} ms/sample) must not be interpreted as a LOCO speedup; the defensible conclusion is that LOCO adds no inference-time module or theoretical inference cost.
- Training wall time increases from {baseline['train_wall_hours']:.2f} to {loco['train_wall_hours']:.2f} hours per seed on average ({training_ratio:.2f}x), so unchanged inference cost must not be confused with unchanged training cost.

## Bottom line

The external KdV run shows that five-seed, Galilean-only LOCO under an independent LPSDA-aligned FNO(AR) protocol preserves local ID accuracy with an unchanged inference architecture. It does **not** reproduce full LPSDA's published label-efficiency gain, does **not** provide a controlled same-generator LOCO-versus-LPSDA comparison, and does **not** provide a matched symmetry-metric comparison on this benchmark.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an accurately scoped LPSDA-aligned KdV comparison table."
    )
    parser.add_argument(
        "--baseline-root", default="runs/external/lpsda_kdv_faithful/baseline"
    )
    parser.add_argument(
        "--loco-root", default="runs/external/lpsda_kdv_faithful/loco_lambda_0p05"
    )
    parser.add_argument(
        "--out-prefix", default="runs/paper_tables/lpsda_kdv_faithful_comparison"
    )
    args = parser.parse_args()

    baseline_rows = _local_run_rows(
        root=Path(args.baseline_root),
        method="baseline",
        method_label="FNO (AR), local calibration",
        expect_orbit=False,
    )
    loco_rows = _local_run_rows(
        root=Path(args.loco_root),
        method="orbit",
        method_label="FNO (AR) + LOCO (g3 only; lambda=0.05)",
        expect_orbit=True,
    )
    baseline_aggregate = _aggregate_local(baseline_rows)
    loco_aggregate = _aggregate_local(loco_rows)
    aggregate_rows = _published_aggregate_rows() + [baseline_aggregate, loco_aggregate]
    paired_rows, paired_summary = _paired_summary(baseline_rows, loco_rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    aggregate_fields = [
        "evidence_scope",
        "source",
        "method",
        "task",
        "rollout_mode",
        "symmetry_scope",
        "uncertainty_scope",
        "training_samples",
        "seeds",
        "parameters",
        "trajectory_nmse",
        "trajectory_nmse_std",
        "orbit_ood_trajectory_nmse",
        "orbit_ood_trajectory_nmse_std",
        "equivariance_defect_relative",
        "equivariance_defect_relative_std",
        "latency_ms_per_sample",
        "latency_ms_per_sample_std",
        "train_wall_hours",
        "train_wall_hours_std",
        "orbit_evaluated_seeds",
        "comparison_note",
    ]
    run_fields = [
        "evidence_scope",
        "source",
        "method",
        "task",
        "rollout_mode",
        "symmetry_scope",
        "uncertainty_scope",
        "seed",
        "training_samples",
        "epochs",
        "rollout_time_steps",
        "trajectory_nmse",
        "orbit_ood_trajectory_nmse",
        "equivariance_defect_relative",
        "latency_ms_per_sample",
        "parameters",
        "train_wall_seconds",
        "dataset_sha256",
        "config_hash",
        "orbit_eval_samples",
        "orbit_eval_seed",
        "metrics_path",
    ]
    paired_fields = [
        "seed",
        "baseline_trajectory_nmse",
        "loco_trajectory_nmse",
        "loco_minus_baseline",
    ]
    _write_csv(out_prefix.with_suffix(".csv"), aggregate_rows, aggregate_fields)
    _write_csv(out_prefix.with_suffix(".runs.csv"), baseline_rows + loco_rows, run_fields)
    _write_csv(out_prefix.with_suffix(".paired.csv"), paired_rows, paired_fields)
    _write_csv(
        out_prefix.with_suffix(".paired_summary.csv"),
        [paired_summary],
        list(paired_summary),
    )
    _write_tex(out_prefix.with_suffix(".tex"), aggregate_rows)
    _write_interpretation(
        out_prefix.with_suffix(".interpretation.md"),
        baseline_aggregate,
        loco_aggregate,
        paired_summary,
    )
    print(f"wrote comparison artifacts under {out_prefix.parent}")


if __name__ == "__main__":
    main()
