#!/usr/bin/env python
"""Aggregate the cache-only finite-objective comparison requested by Reviewer 3."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from scipy.stats import t as student_t

from otno.reporting import collect_run_rows


DEFAULT_OUT_PREFIX = "runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison"
FIVE_SEEDS = (23, 31, 47, 59, 71)
RAW_SEEDS = FIVE_SEEDS
CORE_METRICS = (
    "relative_l2",
    "orbit_ood_relative_l2",
    "equivariance_defect_relative",
)
AGGREGATE_METRICS = CORE_METRICS + (
    "best_val_relative_l2",
    "latency_ms_per_sample",
)


class ComparisonValidationError(ValueError):
    """Raised when the recorded runs do not exactly match the audited protocol."""


@dataclass(frozen=True)
class ObjectiveSpec:
    objective: str
    method_label: str
    run_root: Path
    expected_seeds: tuple[int, ...]
    training_method: str
    steps_per_epoch: int
    weight_name: str
    weight_config_key: str
    weight: float
    normalization: str
    normalized_by_epsilon: bool | None
    normalization_eta: float | None
    uses_jvp: bool
    batch_forward_evaluations_per_epoch: int | None


def _build_specs(args: argparse.Namespace) -> tuple[ObjectiveSpec, ...]:
    return (
        ObjectiveSpec(
            objective="augmentation baseline",
            method_label="Augmentation",
            run_root=Path(args.aug_root),
            expected_seeds=FIVE_SEEDS,
            training_method="aug",
            steps_per_epoch=6,
            weight_name="lambda_aug",
            weight_config_key="config.training.lambda_aug",
            weight=1.0,
            normalization="not applicable",
            normalized_by_epsilon=None,
            normalization_eta=None,
            uses_jvp=False,
            batch_forward_evaluations_per_epoch=12,
        ),
        ObjectiveSpec(
            objective="tangent",
            method_label="Augmentation + tangent",
            run_root=Path(args.tangent_root),
            expected_seeds=FIVE_SEEDS,
            training_method="aug_tangent",
            steps_per_epoch=4,
            weight_name="lambda_tangent",
            weight_config_key="config.training.lambda_tangent",
            weight=0.1,
            normalization="not applicable (JVP tangent)",
            normalized_by_epsilon=None,
            normalization_eta=None,
            uses_jvp=True,
            batch_forward_evaluations_per_epoch=None,
        ),
        ObjectiveSpec(
            objective="normalized finite",
            method_label="Augmentation + normalized finite",
            run_root=Path(args.normalized_root),
            expected_seeds=FIVE_SEEDS,
            training_method="aug_orbit",
            steps_per_epoch=4,
            weight_name="lambda_orbit",
            weight_config_key="config.training.lambda_orbit",
            weight=0.1,
            normalization="divide by epsilon^2 + eta",
            normalized_by_epsilon=True,
            normalization_eta=1e-6,
            uses_jvp=False,
            batch_forward_evaluations_per_epoch=12,
        ),
        ObjectiveSpec(
            objective="raw finite",
            method_label="Augmentation + raw finite",
            run_root=Path(args.raw_root),
            expected_seeds=RAW_SEEDS,
            training_method="aug_orbit",
            steps_per_epoch=4,
            weight_name="lambda_raw",
            weight_config_key="config.training.lambda_orbit",
            weight=2.4,
            normalization="none (raw finite MSE)",
            normalized_by_epsilon=False,
            normalization_eta=None,
            uses_jvp=False,
            batch_forward_evaluations_per_epoch=12,
        ),
    )


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], objective: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ComparisonValidationError(
            f"{objective}: recorded rows are missing required columns {missing}"
        )


def _seed_as_int(value: Any, objective: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ComparisonValidationError(f"{objective}: invalid seed value {value!r}") from exc
    if not numeric.is_integer():
        raise ComparisonValidationError(f"{objective}: non-integral seed value {value!r}")
    return int(numeric)


def _validate_seed_set(df: pd.DataFrame, spec: ObjectiveSpec) -> list[int]:
    if "seed" not in df.columns:
        raise ComparisonValidationError(f"{spec.objective}: recorded rows have no seed column")
    seeds = [_seed_as_int(value, spec.objective) for value in df["seed"].tolist()]
    counts = Counter(seeds)
    duplicated = {seed: count for seed, count in sorted(counts.items()) if count != 1}
    if duplicated:
        raise ComparisonValidationError(
            f"{spec.objective}: duplicate completed runs by seed {duplicated}; "
            "expected exactly one test_metrics.json per configured seed"
        )
    expected = set(spec.expected_seeds)
    found = set(seeds)
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ComparisonValidationError(
            f"{spec.objective}: incomplete or unexpected seed set under {spec.run_root}; "
            f"expected {list(spec.expected_seeds)}, found {sorted(found)}, "
            f"missing {missing}, extra {extra}"
        )
    return seeds


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_constant(
    df: pd.DataFrame,
    *,
    column: str,
    expected: Any,
    objective: str,
) -> None:
    if column not in df.columns:
        raise ComparisonValidationError(f"{objective}: missing protocol field {column!r}")
    bad: list[str] = []
    for _, row in df.iterrows():
        if not _value_matches(row[column], expected):
            bad.append(f"seed={int(row['seed'])}: {row[column]!r}")
    if bad:
        raise ComparisonValidationError(
            f"{objective}: expected {column}={expected!r}, found " + "; ".join(bad)
        )


def _load_objective(spec: ObjectiveSpec) -> pd.DataFrame:
    rows = collect_run_rows(spec.run_root)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ComparisonValidationError(
            f"{spec.objective}: no completed test_metrics.json files under {spec.run_root}; "
            f"expected seeds {list(spec.expected_seeds)}"
        )
    _require_columns(
        df,
        (
            "run_dir",
            "config_path",
            "seed",
            "config.seed",
            "method",
            "config.training.method",
            "config.training.epochs",
            "config.training.batch_size",
            "config.training.lr",
            "config.training.weight_decay",
            "config.training.data_fraction",
            "config.training.steps_per_epoch",
            "config.training.lambda_aug",
            "config.symmetry.max_boost",
            "dataset_kind",
            "dataset_sha256",
            "model_name",
            "parameters",
        )
        + AGGREGATE_METRICS,
        spec.objective,
    )
    df = df.copy()
    df["seed"] = _validate_seed_set(df, spec)
    _validate_constant(
        df, column="method", expected=spec.training_method, objective=spec.objective
    )
    _validate_constant(
        df,
        column="config.training.method",
        expected=spec.training_method,
        objective=spec.objective,
    )
    _validate_constant(
        df,
        column="config.training.data_fraction",
        expected=0.02,
        objective=spec.objective,
    )
    _validate_constant(
        df, column="config.training.epochs", expected=150, objective=spec.objective
    )
    _validate_constant(
        df, column="config.training.batch_size", expected=16, objective=spec.objective
    )
    _validate_constant(
        df, column="config.training.lr", expected=0.001, objective=spec.objective
    )
    _validate_constant(
        df,
        column="config.training.weight_decay",
        expected=0.0001,
        objective=spec.objective,
    )
    _validate_constant(
        df,
        column="config.training.steps_per_epoch",
        expected=spec.steps_per_epoch,
        objective=spec.objective,
    )
    _validate_constant(
        df,
        column="config.training.lambda_aug",
        expected=1.0,
        objective=spec.objective,
    )
    _validate_constant(
        df,
        column=spec.weight_config_key,
        expected=spec.weight,
        objective=spec.objective,
    )
    _validate_constant(
        df,
        column="config.symmetry.max_boost",
        expected=0.25,
        objective=spec.objective,
    )
    if spec.normalized_by_epsilon is not None:
        _validate_constant(
            df,
            column="config.training.normalize_by_epsilon",
            expected=spec.normalized_by_epsilon,
            objective=spec.objective,
        )
    if spec.normalization_eta is not None:
        _validate_constant(
            df,
            column="config.training.orbit_eta",
            expected=spec.normalization_eta,
            objective=spec.objective,
        )
    for _, row in df.iterrows():
        if _seed_as_int(row["config.seed"], spec.objective) != int(row["seed"]):
            raise ComparisonValidationError(
                f"{spec.objective}: metrics/config seed mismatch in {row['run_dir']}"
            )
    for metric in AGGREGATE_METRICS:
        if metric not in df.columns:
            continue
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ComparisonValidationError(
                f"{spec.objective}: metric {metric!r} contains missing or non-finite values"
            )

    df["objective"] = spec.objective
    df["method_label"] = spec.method_label
    df["normalization"] = spec.normalization
    df["normalized_by_epsilon"] = (
        "not applicable"
        if spec.normalized_by_epsilon is None
        else str(spec.normalized_by_epsilon).lower()
    )
    df["normalization_eta"] = spec.normalization_eta
    df["uses_jvp"] = spec.uses_jvp
    df["weight_name"] = spec.weight_name
    df["weight"] = spec.weight
    df["seed_count"] = len(spec.expected_seeds)
    df["configured_seeds"] = ",".join(str(seed) for seed in spec.expected_seeds)
    df["batch_forward_evaluations_per_epoch"] = spec.batch_forward_evaluations_per_epoch
    return df.sort_values("seed").reset_index(drop=True)


def _validate_shared_protocol(run_df: pd.DataFrame) -> None:
    for column in ("dataset_sha256", "dataset_kind", "model_name", "parameters"):
        values = run_df[column].drop_duplicates().tolist()
        if len(values) != 1:
            raise ComparisonValidationError(
                f"selected objectives do not share one {column}: {values}"
            )


def _run_manifest(run_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "objective",
        "method_label",
        "method",
        "seed",
        "seed_count",
        "configured_seeds",
        "normalization",
        "normalized_by_epsilon",
        "normalization_eta",
        "uses_jvp",
        "weight_name",
        "weight",
        "steps_per_epoch",
        "batch_forward_evaluations_per_epoch",
        "relative_l2",
        "orbit_ood_relative_l2",
        "equivariance_defect_relative",
        "best_val_relative_l2",
        "latency_ms_per_sample",
        "parameters",
        "dataset_kind",
        "dataset_sha256",
        "config_hash",
        "run_dir",
        "config_path",
    ]
    return run_df[[column for column in columns if column in run_df.columns]].copy()


def _aggregate(run_df: pd.DataFrame, specs: tuple[ObjectiveSpec, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        group = run_df[run_df["objective"] == spec.objective].sort_values("seed")
        row: dict[str, Any] = {
            "objective": spec.objective,
            "method_label": spec.method_label,
            "normalization": spec.normalization,
            "normalized_by_epsilon": (
                "not applicable"
                if spec.normalized_by_epsilon is None
                else str(spec.normalized_by_epsilon).lower()
            ),
            "normalization_eta": spec.normalization_eta,
            "uses_jvp": spec.uses_jvp,
            "weight_name": spec.weight_name,
            "weight": spec.weight,
            "steps_per_epoch": spec.steps_per_epoch,
            "batch_forward_evaluations_per_epoch": (
                spec.batch_forward_evaluations_per_epoch
            ),
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in group["seed"]),
        }
        for metric in AGGREGATE_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_count"] = int(values.count())
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_metric_rows(
    run_df: pd.DataFrame,
    *,
    reference: str,
    candidate: str,
    expected_seeds: tuple[int, ...],
) -> list[dict[str, Any]]:
    reference_df = run_df[run_df["objective"] == reference]
    candidate_df = run_df[run_df["objective"] == candidate]
    metadata_columns = ("normalization", "uses_jvp", "weight_name", "weight")
    _require_columns(reference_df, metadata_columns, reference)
    _require_columns(candidate_df, metadata_columns, candidate)
    reference_metadata = {
        column: reference_df[column].iloc[0] for column in metadata_columns
    }
    candidate_metadata = {
        column: candidate_df[column].iloc[0] for column in metadata_columns
    }
    rows: list[dict[str, Any]] = []
    for metric in CORE_METRICS:
        left = reference_df[["seed", metric]].rename(columns={metric: "reference_value"})
        right = candidate_df[["seed", metric]].rename(columns={metric: "candidate_value"})
        paired = left.merge(right, on="seed", how="inner", validate="one_to_one")
        paired = paired[paired["seed"].isin(expected_seeds)].sort_values("seed")
        found = tuple(int(seed) for seed in paired["seed"])
        if found != expected_seeds:
            raise ComparisonValidationError(
                f"paired comparison {candidate} vs {reference} / {metric}: "
                f"expected seeds {list(expected_seeds)}, found {list(found)}"
            )
        differences = paired["candidate_value"] - paired["reference_value"]
        n = int(differences.count())
        mean = float(differences.mean())
        std = float(differences.std(ddof=1))
        sem = std / math.sqrt(n)
        tcrit = float(student_t.ppf(0.975, df=n - 1))
        half_width = tcrit * sem
        reference_mean = float(paired["reference_value"].mean())
        candidate_mean = float(paired["candidate_value"].mean())
        rows.append(
            {
                "comparison": f"{candidate} minus {reference}",
                "difference_direction": "candidate_minus_reference",
                "reference_objective": reference,
                "candidate_objective": candidate,
                "reference_normalization": reference_metadata["normalization"],
                "candidate_normalization": candidate_metadata["normalization"],
                "reference_uses_jvp": reference_metadata["uses_jvp"],
                "candidate_uses_jvp": candidate_metadata["uses_jvp"],
                "reference_weight_name": reference_metadata["weight_name"],
                "candidate_weight_name": candidate_metadata["weight_name"],
                "reference_weight": reference_metadata["weight"],
                "candidate_weight": candidate_metadata["weight"],
                "metric": metric,
                "seed_count": n,
                "seeds": ",".join(str(seed) for seed in expected_seeds),
                "reference_mean": reference_mean,
                "candidate_mean": candidate_mean,
                "paired_delta_mean": mean,
                "paired_delta_std": std,
                "paired_delta_sem": sem,
                "paired_delta_ci95_low": mean - half_width,
                "paired_delta_ci95_high": mean + half_width,
                "relative_change_pct": 100.0 * mean / reference_mean,
            }
        )
    return rows


def _paired(run_df: pd.DataFrame) -> pd.DataFrame:
    rows = _paired_metric_rows(
        run_df,
        reference="tangent",
        candidate="normalized finite",
        expected_seeds=FIVE_SEEDS,
    )
    rows.extend(
        _paired_metric_rows(
            run_df,
            reference="raw finite",
            candidate="normalized finite",
            expected_seeds=RAW_SEEDS,
        )
    )
    return pd.DataFrame(rows)


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pm(mean: float, std: float) -> str:
    return rf"\({_fmt(mean)}\pm{_fmt(std)}\)"


def _tex_table(
    aggregate: pd.DataFrame,
    run_df: pd.DataFrame | None = None,
) -> str:
    rows = {str(row["objective"]): row for _, row in aggregate.iterrows()}
    normalized_matched = rows["normalized finite"].copy()
    if run_df is not None:
        matched = run_df[
            (run_df["objective"] == "normalized finite")
            & (run_df["seed"].isin(RAW_SEEDS))
        ].sort_values("seed")
        found = tuple(int(seed) for seed in matched["seed"])
        if found != RAW_SEEDS:
            raise ComparisonValidationError(
                "normalized finite TeX panel: expected matched seeds "
                f"{list(RAW_SEEDS)}, found {list(found)}"
            )
        normalized_matched = normalized_matched.copy()
        normalized_matched["seed_count"] = len(RAW_SEEDS)
        for metric in CORE_METRICS:
            values = pd.to_numeric(matched[metric], errors="raise")
            normalized_matched[f"{metric}_mean"] = float(values.mean())
            normalized_matched[f"{metric}_std"] = float(values.std(ddof=1))

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{Finite-objective comparison on the N64 Galilean task at 2\% labels. "
            r"Entries are mean $\pm$ sample standard deviation within each panel. "
            r"Both panels use five matched seeds. "
            r"The augmentation baseline and both finite objectives use 12 batch-level "
            r"model-forward evaluations per epoch. The tangent objective uses an explicit "
            r"JVP and is not compute-matched. The raw finite weight $2.4$ is a leading-order "
            r"average-scale match to normalized finite weight $0.10$, not a tuned equivalence.}"
        ),
        r"\label{tab:n64-finite-objective-comparison}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Method & Weight & Step normalization & JVP & ID $L^2$ & Orbit-shift $L^2$ "
        r"& Eq. defect & Seeds \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\emph{Panel A: finite versus tangent, five matched seeds}} \\",
    ]

    def append_row(objective: str, row: pd.Series) -> None:
        normalization = {
            "augmentation baseline": "n/a",
            "tangent": "n/a",
            "normalized finite": r"$\epsilon^2+\eta$",
            "raw finite": "none",
        }[objective]
        jvp = "yes" if bool(row["uses_jvp"]) else "no"
        weight = {
            "augmentation baseline": r"$\lambda_{\rm aug}=1$",
            "tangent": r"$\lambda_{\rm tangent}=0.1$",
            "normalized finite": r"$\lambda_{\rm orb}=0.1$",
            "raw finite": r"$\lambda_{\rm raw}=2.4$",
        }[objective]
        defect = _fmt_pm(
            row["equivariance_defect_relative_mean"],
            row["equivariance_defect_relative_std"],
        )
        lines.append(
            f"{row['method_label']} & {weight} & {normalization} & {jvp} & "
            f"{_fmt_pm(row['relative_l2_mean'], row['relative_l2_std'])} & "
            f"{_fmt_pm(row['orbit_ood_relative_l2_mean'], row['orbit_ood_relative_l2_std'])} & "
            f"{defect} & "
            f"{int(row['seed_count'])} \\\\"
        )

    for objective in ("augmentation baseline", "tangent", "normalized finite"):
        append_row(objective, rows[objective])
    lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{8}{l}{\emph{Panel B: normalized versus raw finite, five matched seeds}} \\",
        ]
    )
    append_row("normalized finite", normalized_matched)
    append_row("raw finite", rows["raw finite"])
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _output_path(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def build_comparison(specs: tuple[ObjectiveSpec, ...]) -> tuple[pd.DataFrame, ...]:
    frames = [_load_objective(spec) for spec in specs]
    run_df = pd.concat(frames, ignore_index=True)
    objective_order = {spec.objective: index for index, spec in enumerate(specs)}
    run_df["_objective_order"] = run_df["objective"].map(objective_order)
    run_df = run_df.sort_values(["_objective_order", "seed"]).drop(columns="_objective_order")
    _validate_shared_protocol(run_df)
    return _run_manifest(run_df), _aggregate(run_df, specs), _paired(run_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate recorded N64 Galilean finite-objective controls; this script never "
            "trains or evaluates a model."
        )
    )
    parser.add_argument(
        "--aug-root",
        default=(
            "runs/ablations/2d_galilean_n64_label_efficiency_compute_matched/"
            "fraction_0.02/aug_steps_6"
        ),
    )
    parser.add_argument(
        "--tangent-root",
        default=(
            "runs/ablations/2d_galilean_n64_2pct_tangent_baseline/"
            "fraction_0.02/aug_tangent_lambda_0.1_steps_4"
        ),
    )
    parser.add_argument(
        "--normalized-root",
        default=(
            "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
            "fraction_0.02/aug_orbit_lambda_0.1_steps_4"
        ),
    )
    parser.add_argument(
        "--raw-root",
        default=(
            "runs/ablations/2d_galilean_n64_2pct_plain_finite_baseline/"
            "fraction_0.02/aug_plain_finite_lambda_2.4_steps_4"
        ),
    )
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = _build_specs(args)
    try:
        run_df, aggregate, paired = build_comparison(specs)
    except ComparisonValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": _output_path(out_prefix, ".runs.csv"),
        "aggregate": _output_path(out_prefix, ".aggregate.csv"),
        "paired": _output_path(out_prefix, ".paired.csv"),
        "tex": _output_path(out_prefix, ".tex"),
    }
    paths["runs"].write_text(run_df.to_csv(index=False), encoding="utf-8")
    paths["aggregate"].write_text(aggregate.to_csv(index=False), encoding="utf-8")
    paths["paired"].write_text(paired.to_csv(index=False), encoding="utf-8")
    paths["tex"].write_text(_tex_table(aggregate, run_df), encoding="utf-8")
    for path in paths.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
