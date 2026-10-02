"""Collect run records and share validation and statistics across paper reports.

Experiment scripts declare their own methods, metrics, and training protocols;
this module supplies the common arithmetic and provenance checks.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import t as student_t


def value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class RunValidation:
    """Shared checks for tables whose runs must follow one experiment protocol."""

    def __init__(self, error_type: type[ValueError] = ValueError) -> None:
        self.error_type = error_type

    def constant(self, df: pd.DataFrame, column: str, expected: Any, context: str) -> None:
        if column not in df.columns:
            raise self.error_type(f"{context}: missing protocol field {column!r}")
        failures = [
            f"seed={row.get('seed')}: {row[column]!r}"
            for _, row in df.iterrows()
            if not value_matches(row[column], expected)
        ]
        if failures:
            raise self.error_type(
                f"{context}: expected {column}={expected!r}; found " + "; ".join(failures)
            )

    def optional_constant(self, df: pd.DataFrame, column: str, expected: Any, context: str) -> None:
        if column not in df.columns:
            return
        failures = [
            f"seed={row.get('seed')}: {row[column]!r}"
            for _, row in df.iterrows()
            if not is_missing(row[column]) and not value_matches(row[column], expected)
        ]
        if failures:
            raise self.error_type(
                f"{context}: expected absent or {column}={expected!r}; found " + "; ".join(failures)
            )

    def absent(self, df: pd.DataFrame, column: str, context: str) -> None:
        if column not in df.columns:
            return
        failures = [
            f"seed={row.get('seed')}: {row[column]!r}"
            for _, row in df.iterrows()
            if not is_missing(row[column])
        ]
        if failures:
            raise self.error_type(
                f"{context}: expected {column!r} to be absent; found " + "; ".join(failures)
            )

    def nonempty_consistent(self, df: pd.DataFrame, column: str, context: str) -> None:
        if column not in df.columns:
            raise self.error_type(f"{context}: missing provenance field {column!r}")
        for _, row in df.iterrows():
            value = row[column]
            if is_missing(value) or (isinstance(value, str) and not value.strip()):
                raise self.error_type(
                    f"{context}: empty provenance field {column!r} for seed={row.get('seed')}"
                )
        self.consistent(df, column, context)

    def consistent(self, df: pd.DataFrame, column: str, context: str) -> None:
        if column not in df.columns:
            raise self.error_type(f"{context}: missing provenance field {column!r}")
        expected = df.iloc[0][column]
        failures = []
        for _, row in df.iloc[1:].iterrows():
            value = row[column]
            both_missing = is_missing(expected) and is_missing(value)
            if not both_missing and (
                is_missing(expected) or is_missing(value) or not value_matches(value, expected)
            ):
                failures.append(f"seed={row.get('seed')}: {value!r}")
        if failures:
            raise self.error_type(
                f"{context}: inconsistent {column!r}; expected {expected!r}; found "
                + "; ".join(failures)
            )

    def seed_values(self, df: pd.DataFrame, context: str) -> list[int]:
        if "seed" not in df.columns:
            raise self.error_type(f"{context}: missing seed field")
        seeds = []
        for value in df["seed"]:
            try:
                seed = int(value)
                if isinstance(value, bool) or (not isinstance(value, str) and value != seed):
                    raise ValueError
            except (TypeError, ValueError, OverflowError) as exc:
                raise self.error_type(f"{context}: non-integer seed value {value!r}") from exc
            seeds.append(seed)
        counts = Counter(seeds)
        duplicates = {seed: count for seed, count in counts.items() if count != 1}
        if duplicates:
            raise self.error_type(f"{context}: duplicate seed rows {duplicates}")
        return seeds

    def seeds(self, df: pd.DataFrame, context: str, expected_seeds: tuple[int, ...]) -> None:
        seeds = self.seed_values(df, context)
        if set(seeds) != set(expected_seeds):
            raise self.error_type(
                f"{context}: expected seeds {list(expected_seeds)}, found {sorted(seeds)}"
            )

    def finite_metrics(self, df: pd.DataFrame, context: str, metrics: tuple[str, ...]) -> None:
        for metric in metrics:
            if metric not in df.columns:
                raise self.error_type(f"{context}: missing metric {metric!r}")
            values = pd.to_numeric(df[metric], errors="coerce")
            if values.isna().any() or not values.map(math.isfinite).all():
                raise self.error_type(f"{context}: non-finite values in {metric!r}")

    def fno_reference_seeds(self, path: Path, methods: list[str]) -> tuple[int, ...]:
        """Read the seed set shared by all requested FNO reference methods."""
        if not path.is_file():
            raise self.error_type(f"FNO seed reference does not exist: {path}")
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise self.error_type(f"could not read FNO seed reference {path}: {exc}") from exc
        required = {"method", "model_name", "seed"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise self.error_type(f"FNO seed reference {path} is missing columns {missing}")
        reference: tuple[int, ...] | None = None
        for method in methods:
            group = df[df["method"] == method]
            context = f"FNO {method} seed reference in {path}"
            if group.empty:
                raise self.error_type(f"{context}: no rows")
            model_names = set(group["model_name"].dropna().astype(str))
            if model_names != {"fno2d"}:
                raise self.error_type(
                    f"{context}: expected model_name='fno2d', found {sorted(model_names)}"
                )
            seeds = self.seed_values(group, context)
            current = tuple(sorted(seeds))
            if reference is None:
                reference = current
            elif current != reference:
                raise self.error_type(
                    f"{context}: expected the shared FNO seeds {reference}, found {current}"
                )
        if reference is None:
            raise self.error_type(f"FNO seed reference {path} contains no methods")
        return reference


def _paired_difference_statistics(
    reference: pd.Series, candidate: pd.Series
) -> dict[str, float | int | bool]:
    """Compute a Student-t interval from already aligned independent seed pairs."""
    differences = candidate - reference
    n = int(differences.shape[0])
    delta = float(differences.mean())
    delta_std = float(differences.std(ddof=1))
    delta_sem = delta_std / math.sqrt(n)
    half_width = float(student_t.ppf(0.975, n - 1)) * delta_sem
    return {
        "seed_count": n,
        "reference_mean": float(reference.mean()),
        "candidate_mean": float(candidate.mean()),
        "paired_delta_mean": delta,
        "paired_delta_std": delta_std,
        "paired_delta_sem": delta_sem,
        "paired_delta_ci95_low": delta - half_width,
        "paired_delta_ci95_high": delta + half_width,
        "ci_excludes_zero": bool(delta - half_width > 0 or delta + half_width < 0),
    }


def paired_metric_summary(
    reference: pd.DataFrame, candidate: pd.DataFrame, metric: str, *, context: str
) -> dict[str, float | int]:
    """Summarize candidate-minus-reference differences over the same independent seeds."""
    validation = RunValidation()
    reference_seeds = validation.seed_values(reference, f"{context}: reference")
    candidate_seeds = validation.seed_values(candidate, f"{context}: candidate")
    if set(reference_seeds) != set(candidate_seeds):
        raise ValueError(f"{context}: reference and candidate seed sets differ")
    n = len(reference_seeds)
    if n < 2:
        raise ValueError(
            f"{context}: at least two paired seeds are required for a confidence interval"
        )
    validation.finite_metrics(reference, f"{context}: reference", (metric,))
    validation.finite_metrics(candidate, f"{context}: candidate", (metric,))
    if "dataset_sha256" in reference or "dataset_sha256" in candidate:
        validation.nonempty_consistent(
            pd.concat([reference, candidate], ignore_index=True), "dataset_sha256", context
        )
    left = pd.DataFrame(
        {"seed": reference_seeds, "reference": pd.to_numeric(reference[metric]).to_numpy()}
    )
    right = pd.DataFrame(
        {"seed": candidate_seeds, "candidate": pd.to_numeric(candidate[metric]).to_numpy()}
    )
    paired = left.merge(right, on="seed", validate="one_to_one").sort_values("seed")
    statistics = _paired_difference_statistics(paired["reference"], paired["candidate"])
    reference_mean = statistics["reference_mean"]
    candidate_mean = statistics["candidate_mean"]
    return {
        "seed_count": n,
        "reference_mean": reference_mean,
        "candidate_mean": candidate_mean,
        "paired_delta_mean": statistics["paired_delta_mean"],
        "paired_delta_ci95_low": statistics["paired_delta_ci95_low"],
        "paired_delta_ci95_high": statistics["paired_delta_ci95_high"],
        "relative_reduction_pct": (
            100.0 * (reference_mean - candidate_mean) / reference_mean
            if reference_mean != 0.0
            else math.nan
        ),
    }


def paired_protocol_comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    comparison: str,
    error_type: type[ValueError] = ValueError,
) -> pd.DataFrame:
    """Compare validated backbone runs using their declared metric and seed order.

    Run loaders validate the experiment protocol before this step. The merge still
    checks that each seed occurs once, and each metric must have the expected pairs.
    A zero reference mean remains an error for these relative-reduction tables.
    """
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        paired = (
            reference[["seed", metric]]
            .rename(columns={metric: "reference"})
            .merge(
                candidate[["seed", metric]].rename(columns={metric: "candidate"}),
                on="seed",
                how="inner",
                validate="one_to_one",
            )
            .sort_values("seed")
        )
        seeds = tuple(int(seed) for seed in paired["seed"])
        if seeds != expected_seeds:
            raise error_type(
                f"primary paired comparison for {metric}: expected {expected_seeds}, found {seeds}"
            )
        statistics = _paired_difference_statistics(paired["reference"], paired["candidate"])
        reference_mean = statistics["reference_mean"]
        candidate_mean = statistics["candidate_mean"]
        rows.append(
            {
                "comparison": comparison,
                "difference_direction": "candidate_minus_reference",
                "metric": metric,
                "seed_count": statistics["seed_count"],
                "seeds": ",".join(str(seed) for seed in expected_seeds),
                "reference_mean": reference_mean,
                "candidate_mean": candidate_mean,
                "paired_delta_mean": statistics["paired_delta_mean"],
                "paired_delta_std": statistics["paired_delta_std"],
                "paired_delta_sem": statistics["paired_delta_sem"],
                "paired_delta_ci95_low": statistics["paired_delta_ci95_low"],
                "paired_delta_ci95_high": statistics["paired_delta_ci95_high"],
                "relative_reduction_pct": (
                    100.0 * (reference_mean - candidate_mean) / reference_mean
                ),
                "ci_excludes_zero": statistics["ci_excludes_zero"],
            }
        )
    return pd.DataFrame(rows)


def backbone_training_summary(
    group: pd.DataFrame,
    *,
    method: str,
    label: str,
    epochs: int,
    steps_per_epoch: int,
    model_forwards_per_epoch: int,
) -> dict[str, Any]:
    """Describe a validated, seed-sorted method's training budget and seed set."""
    return {
        "method": method,
        "method_label": label,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": steps_per_epoch * epochs,
        "model_forwards_per_epoch": model_forwards_per_epoch,
        "model_forwards_total": model_forwards_per_epoch * epochs,
        "backward_passes_total": steps_per_epoch * epochs,
        "seed_count": int(group["seed"].nunique()),
        "seeds": ",".join(str(int(seed)) for seed in group["seed"]),
        "parameters": int(group["parameters"].iloc[0]),
    }


def metric_mean_std(group: pd.DataFrame, *, metrics: tuple[str, ...]) -> dict[str, float]:
    """Summarize declared metrics in order, using sample standard deviations."""
    summary = {}
    for metric in metrics:
        values = pd.to_numeric(group[metric], errors="raise")
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
    return summary


def drop_empty_config_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove config fields introduced by unrelated runs before a table was filtered."""
    columns = [col for col in df if col.startswith("config.") and df[col].isna().all()]
    return df.drop(columns=columns)


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    else:
        out[prefix] = value


def collect_run_rows(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Read completed run metrics and flatten their saved configs into table columns."""
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(Path(runs_dir).glob("**/test_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        row: dict[str, Any] = {"run_dir": str(metrics_path.parent)}
        _flatten("", payload, row)
        config_path = metrics_path.parent / "config.yaml"
        if config_path.exists():
            row["config_path"] = str(config_path)
            with config_path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
            config_flat: dict[str, Any] = {}
            _flatten("config", config, config_flat)
            row.update(config_flat)
            training = config.get("training", {}) if isinstance(config, dict) else {}
            if isinstance(training, dict):
                row.setdefault("data_fraction", float(training.get("data_fraction", 1.0)))
                if "steps_per_epoch" in training:
                    row.setdefault("steps_per_epoch", int(training["steps_per_epoch"]))
                if "orbit_data_fraction" in training:
                    row.setdefault("orbit_data_fraction", float(training["orbit_data_fraction"]))
                if "orbit_steps_per_epoch" in training:
                    row.setdefault("orbit_steps_per_epoch", int(training["orbit_steps_per_epoch"]))
                if "lambda_orbit" in training:
                    row.setdefault("lambda_orbit", float(training["lambda_orbit"]))
        rows.append(row)
    return rows


def validate_fresh_run_provenance(df: pd.DataFrame, *, source_root: str | Path) -> None:
    """Validate fresh campaigns from a source archive, including archives without Git."""
    from otno.utils import file_sha256, source_manifest, stable_json_hash

    root = Path(source_root)
    current_source = source_manifest(root)["source_sha256"]
    dataset_hashes: dict[Path, str] = {}
    campaign_dataset = None
    for _, row in df.iterrows():
        run_dir = root / str(row["run_dir"])
        manifest_path = run_dir / "source_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Missing or invalid source manifest: {manifest_path}") from exc
        recorded_source = row.get("source_sha256")
        if (
            not isinstance(manifest, dict)
            or not isinstance(manifest.get("files"), dict)
            or not manifest["files"]
            or stable_json_hash(manifest["files"]) != manifest.get("source_sha256")
            or recorded_source != manifest.get("source_sha256")
            or recorded_source != current_source
        ):
            raise ValueError(
                f"Fresh run source_sha256 does not match the actual release source: {run_dir}"
            )
        if row.get("environment.torch_num_threads") != 1:
            raise ValueError(f"Fresh campaign requires environment.torch_num_threads=1: {run_dir}")
        dataset_path = root / str(row["config.dataset.path"])
        if dataset_path not in dataset_hashes:
            dataset_hashes[dataset_path] = file_sha256(dataset_path)
        actual_dataset = dataset_hashes[dataset_path]
        if row.get("dataset_sha256") != actual_dataset:
            raise ValueError(
                f"Fresh run dataset_sha256 does not match the actual dataset: {run_dir}"
            )
        if campaign_dataset is not None and actual_dataset != campaign_dataset:
            raise ValueError("Fresh campaign contains inconsistent dataset_sha256 values")
        campaign_dataset = actual_dataset


def aggregate_runs(df: pd.DataFrame, *, group_cols: list[str] | None = None) -> pd.DataFrame:
    """Group general run tables into mean/std/count columns for available metrics.

    Backbone tables additionally disclose their experiment budgets and resume
    history, so their scripts use the explicit summary helpers above instead.
    """
    if df.empty:
        return df
    group_cols = group_cols or [
        col
        for col in ["method", "model_name", "dataset_kind", "data_fraction"]
        if col in df.columns
    ]
    metric_cols = [
        col
        for col in [
            "relative_l2",
            "orbit_ood_relative_l2",
            "oracle_canonical_ood_relative_l2",
            "equivariance_defect_relative",
            "latency_ms_per_sample",
            "latency_ms_per_batch",
        ]
        if col in df.columns
    ]
    if not group_cols or not metric_cols:
        return df
    agg = (
        df.groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(part) for part in col if part) if isinstance(col, tuple) else str(col)
        for col in agg.columns
    ]
    return agg
