#!/usr/bin/env python
"""Validate and aggregate the five-seed CNO2d backbone experiment."""

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

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from otno.reporting import collect_run_rows


EXPECTED_SEEDS = (23, 31, 47, 59, 71)
EXPECTED_DATASET_SHA256 = "defc9703dbc41494fabf9fcfe9e3408eb2e1c3890814e47fe03d0f6b849e3f50"
EXPECTED_PARAMETERS = 2_667_743
EXPECTED_EXECUTION = {
    "environment.omp_num_threads": "8",
    "environment.mkl_num_threads": "8",
    "device": "cpu",
    "deterministic": False,
}
NONEMPTY_PROVENANCE_FIELDS = (
    "environment.git_commit",
    "environment.platform",
    "environment.python",
    "environment.torch",
)
CONSISTENT_PROVENANCE_FIELDS = (
    "environment.cuda_available",
    "environment.cuda_version",
    "environment.cudnn_version",
)
DEFAULT_RUN_ROOT = Path(
    "runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/fraction_0.02"
)
DEFAULT_OUT_PREFIX = Path("runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed")
DEFAULT_FNO_RUNS = Path(
    "runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.runs.csv"
)
CORE_METRICS = (
    "relative_l2",
    "orbit_ood_relative_l2",
    "equivariance_defect_relative",
)
AGGREGATE_METRICS = CORE_METRICS + (
    "best_val_relative_l2",
    "latency_ms_per_sample",
)
FINITE_METRICS = AGGREGATE_METRICS + ("train_wall_seconds",)
RESUME_AUTHORIZATION_SOURCE = (
    "runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/PAUSED.md"
)


class CNO2dProtocolError(ValueError):
    """Raised when cached runs do not exactly match the declared protocol."""


@dataclass(frozen=True)
class MethodSpec:
    method: str
    label: str
    relative_dir: str
    steps_per_epoch: int
    model_forwards_per_step: int
    lambda_orbit: float | None

    @property
    def model_forwards_per_epoch(self) -> int:
        return self.steps_per_epoch * self.model_forwards_per_step


@dataclass(frozen=True)
class ResumeSpec:
    last_completed_epoch: int
    last_checkpoint_sha256: str
    best_checkpoint_sha256: str
    resumed_config_hash: str


METHOD_SPECS = (
    MethodSpec("baseline", "CNO2d", "baseline_steps_4", 4, 1, None),
    MethodSpec(
        "orbit",
        "CNO2d + normalized LOCO",
        "orbit_lambda_0.05_steps_4",
        4,
        2,
        0.05,
    ),
    MethodSpec("aug", "CNO2d + aug.", "aug_steps_6", 6, 2, None),
    MethodSpec(
        "aug_orbit",
        "CNO2d + aug. + normalized LOCO",
        "aug_loco_lambda_0.1_steps_4",
        4,
        3,
        0.10,
    ),
)


# These are the only checkpoint continuations authorized by the campaign's
# recorded pause state.  The checkpoint files are overwritten as training
# continues, so their pre-resume hashes are retained here for disclosure rather
# than presented as hashes of the final checkpoint files.
AUTHORIZED_RESUMES = {
    ("baseline", 23): ResumeSpec(
        last_completed_epoch=108,
        last_checkpoint_sha256=(
            "2bd7a1e0f067fe3e5975816c69a5547991a34bc9b0a179871ed057f64e28b208"
        ),
        best_checkpoint_sha256=(
            "9b9457c84e7e6925e842e2cfe0e38ac928340c9586ad50d86cda31b56019339e"
        ),
        resumed_config_hash="57d7580e4a21",
    ),
    ("aug_orbit", 59): ResumeSpec(
        last_completed_epoch=103,
        last_checkpoint_sha256=(
            "37579559adb2f66305a4502aa1a913eafacf042fe40aef7cc49d45ce7e387a70"
        ),
        best_checkpoint_sha256=(
            "a3031d0f375ff34ce1090203121b2c593f7ac47f8bcddd3390ebd6e78096eb02"
        ),
        resumed_config_hash="c3963fd57bc2",
    ),
}


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_constant(df: pd.DataFrame, column: str, expected: Any, context: str) -> None:
    if column not in df.columns:
        raise CNO2dProtocolError(f"{context}: missing protocol field {column!r}")
    failures = []
    for _, row in df.iterrows():
        if not _value_matches(row[column], expected):
            failures.append(f"seed={row.get('seed')}: {row[column]!r}")
    if failures:
        raise CNO2dProtocolError(
            f"{context}: expected {column}={expected!r}; found " + "; ".join(failures)
        )


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _validate_optional_constant(
    df: pd.DataFrame, column: str, expected: Any, context: str
) -> None:
    if column not in df.columns:
        return
    failures = []
    for _, row in df.iterrows():
        value = row[column]
        if not _is_missing(value) and not _value_matches(value, expected):
            failures.append(f"seed={row.get('seed')}: {value!r}")
    if failures:
        raise CNO2dProtocolError(
            f"{context}: expected absent or {column}={expected!r}; found "
            + "; ".join(failures)
        )


def _validate_absent(df: pd.DataFrame, column: str, context: str) -> None:
    if column not in df.columns:
        return
    failures = []
    for _, row in df.iterrows():
        value = row[column]
        if not _is_missing(value):
            failures.append(f"seed={row.get('seed')}: {value!r}")
    if failures:
        raise CNO2dProtocolError(
            f"{context}: expected {column!r} to be absent; found " + "; ".join(failures)
        )


def _annotate_authorized_resumes(
    df: pd.DataFrame, spec: MethodSpec, context: str, *, historical: bool = True
) -> pd.DataFrame:
    """Validate the exact campaign resume allowlist and add disclosure fields."""
    out = df.copy()
    expected = {
        (method, seed): resume_spec
        for (method, seed), resume_spec in AUTHORIZED_RESUMES.items()
        if historical and method == spec.method
    }
    observed: set[tuple[str, int]] = set()
    annotations: list[dict[str, Any]] = []

    for _, row in out.iterrows():
        value = row.get("config.runtime.resume")
        if _is_missing(value):
            resumed = False
        elif isinstance(value, (bool, np.bool_)):
            resumed = bool(value)
        else:
            raise CNO2dProtocolError(
                f"{context}: expected 'config.runtime.resume' to be a boolean or "
                f"absent for seed={row.get('seed')}; found {value!r}"
            )

        key = (str(row.get("method")), int(row.get("seed")))
        authorization = AUTHORIZED_RESUMES.get(key)
        if resumed:
            if authorization is None:
                raise CNO2dProtocolError(
                    f"{context}: unauthorized checkpoint resume in "
                    f"config.runtime.resume for "
                    f"method={key[0]!r}, seed={key[1]}"
                )
            observed.add(key)
            actual_hash = row.get("config_hash")
            if actual_hash != authorization.resumed_config_hash:
                raise CNO2dProtocolError(
                    f"{context}: resumed config hash mismatch for seed={key[1]}; "
                    f"expected {authorization.resumed_config_hash!r}, "
                    f"found {actual_hash!r}"
                )

        annotations.append(
            {
                "checkpoint_resumed": resumed,
                "resume_last_completed_epoch": (
                    authorization.last_completed_epoch if resumed else None
                ),
                "resume_source_last_checkpoint_sha256": (
                    authorization.last_checkpoint_sha256 if resumed else None
                ),
                "resume_source_best_checkpoint_sha256": (
                    authorization.best_checkpoint_sha256 if resumed else None
                ),
                "resume_authorization_source": (
                    RESUME_AUTHORIZATION_SOURCE if resumed else None
                ),
                "train_wall_seconds_scope": (
                    "post_resume_segment" if resumed else "full_run"
                ),
            }
        )

    missing = sorted(set(expected).difference(observed))
    if missing:
        rendered = ", ".join(f"{method}/seed_{seed}" for method, seed in missing)
        raise CNO2dProtocolError(
            f"{context}: expected authorized checkpoint resume(s) {rendered}"
        )

    for column in annotations[0] if annotations else ():
        out[column] = [annotation[column] for annotation in annotations]
    return out


def _validate_nonempty_consistent(
    df: pd.DataFrame, column: str, context: str
) -> None:
    if column not in df.columns:
        raise CNO2dProtocolError(f"{context}: missing provenance field {column!r}")
    values: list[tuple[Any, Any]] = []
    for _, row in df.iterrows():
        value = row[column]
        if _is_missing(value) or (isinstance(value, str) and not value.strip()):
            raise CNO2dProtocolError(
                f"{context}: empty provenance field {column!r} for seed={row.get('seed')}"
            )
        values.append((row.get("seed"), value))
    expected = values[0][1]
    failures = [
        f"seed={seed}: {value!r}"
        for seed, value in values[1:]
        if not _value_matches(value, expected)
    ]
    if failures:
        raise CNO2dProtocolError(
            f"{context}: inconsistent {column!r}; expected {expected!r}; found "
            + "; ".join(failures)
        )


def _validate_consistent(df: pd.DataFrame, column: str, context: str) -> None:
    if column not in df.columns:
        raise CNO2dProtocolError(f"{context}: missing provenance field {column!r}")
    expected = df.iloc[0][column]
    failures = []
    for _, row in df.iloc[1:].iterrows():
        value = row[column]
        both_missing = _is_missing(expected) and _is_missing(value)
        if not both_missing and (
            _is_missing(expected)
            or _is_missing(value)
            or not _value_matches(value, expected)
        ):
            failures.append(f"seed={row.get('seed')}: {value!r}")
    if failures:
        raise CNO2dProtocolError(
            f"{context}: inconsistent {column!r}; expected {expected!r}; found "
            + "; ".join(failures)
        )


def _validate_seeds(
    df: pd.DataFrame,
    context: str,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> None:
    if "seed" not in df.columns:
        raise CNO2dProtocolError(f"{context}: missing seed field")
    seeds = [int(seed) for seed in df["seed"]]
    counts = Counter(seeds)
    duplicates = {seed: count for seed, count in counts.items() if count != 1}
    if duplicates:
        raise CNO2dProtocolError(f"{context}: duplicate seed rows {duplicates}")
    if set(seeds) != set(expected_seeds):
        raise CNO2dProtocolError(
            f"{context}: expected seeds {list(expected_seeds)}, found {sorted(seeds)}"
        )


def load_fno_reference_seeds(path: Path) -> tuple[int, ...]:
    """Read the exact seed set shared by the four primary FNO rows."""
    if not path.is_file():
        raise CNO2dProtocolError(f"FNO seed reference does not exist: {path}")
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        raise CNO2dProtocolError(f"could not read FNO seed reference {path}: {exc}") from exc

    required = {"method", "model_name", "seed"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise CNO2dProtocolError(f"FNO seed reference {path} is missing columns {missing}")

    reference: tuple[int, ...] | None = None
    for spec in METHOD_SPECS:
        group = df[df["method"] == spec.method]
        context = f"FNO {spec.method} seed reference in {path}"
        if group.empty:
            raise CNO2dProtocolError(f"{context}: no rows")
        model_names = set(group["model_name"].dropna().astype(str))
        if model_names != {"fno2d"}:
            raise CNO2dProtocolError(
                f"{context}: expected model_name='fno2d', found {sorted(model_names)}"
            )
        try:
            seeds = [int(seed) for seed in group["seed"]]
        except (TypeError, ValueError) as exc:
            raise CNO2dProtocolError(f"{context}: non-integer seed") from exc
        counts = Counter(seeds)
        duplicates = {seed: count for seed, count in counts.items() if count != 1}
        if duplicates:
            raise CNO2dProtocolError(f"{context}: duplicate seed rows {duplicates}")
        current = tuple(sorted(seeds))
        if reference is None:
            reference = current
        elif current != reference:
            raise CNO2dProtocolError(
                f"{context}: expected the shared FNO seeds {reference}, found {current}"
            )
    if reference is None:
        raise CNO2dProtocolError(f"FNO seed reference {path} contains no methods")
    return reference


def _validate_finite_metrics(df: pd.DataFrame, context: str) -> None:
    for metric in FINITE_METRICS:
        if metric not in df.columns:
            raise CNO2dProtocolError(f"{context}: missing metric {metric!r}")
        values = pd.to_numeric(df[metric], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise CNO2dProtocolError(f"{context}: non-finite values in {metric!r}")


def _expected_protocol(spec: MethodSpec, *, fresh_runs: bool = False) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "method": spec.method,
        "model_name": "cno2d",
        "parameters": EXPECTED_PARAMETERS,
        "parameters_total": EXPECTED_PARAMETERS,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dataset_kind": "navier_stokes_vorticity2d_boosted",
        "orbit_control": "physical",
        "num_samples": 256,
        "num_batches": 16,
        "latency_repeats": 20,
        "latency_warmup": 5,
        "config.training.method": spec.method,
        "config.training.epochs": 150,
        "config.training.batch_size": 16,
        "config.training.data_fraction": 0.02,
        "config.training.steps_per_epoch": spec.steps_per_epoch,
        "config.training.lr": 0.001,
        "config.training.weight_decay": 0.0001,
        "config.training.lambda_aug": 1.0,
        "config.training.grad_clip": 1.0,
        "config.training.eval_every": 10,
        "config.training.eval_orbit_samples": 4,
        "config.training.num_workers": 0,
        "config.training.latency_repeats": 20,
        "config.training.latency_warmup": 5,
        "config.dataset.kind": "navier_stokes_vorticity2d_boosted",
        "config.dataset.path": "data/2d_ns_vorticity_galilean_n64.pt",
        "config.dataset.generate_if_missing": True,
        "config.dataset.n": 64,
        "config.dataset.num_train": 1024,
        "config.dataset.num_val": 128,
        "config.dataset.num_test": 256,
        "config.dataset.final_time": 0.5,
        "config.dataset.dt": 0.001,
        "config.dataset.viscosity": 0.001,
        "config.dataset.smoothness": 8.0,
        "config.dataset.amplitude": 1.0,
        "config.dataset.max_boost": 0.5,
        "config.dataset.solver_batch_size": 8,
        "config.dataset.dealias": True,
        "config.dataset.seed": 31,
        "config.symmetry.name": "navier_stokes2d_galilean",
        "config.symmetry.max_boost": 0.25,
        "config.symmetry.final_time": 0.5,
        "config.symmetry.length": 1.0,
        "config.symmetry.boost_x_channel": 1,
        "config.symmetry.boost_y_channel": 2,
        "config.model.name": "cno2d",
        "config.model.in_channels": 3,
        "config.model.out_channels": 1,
        "config.model.n_layers": 4,
        "config.model.n_res": 4,
        "config.model.n_res_neck": 3,
        "config.model.channel_multiplier": 20,
        "config.model.lift_project_channels": 64,
        "config.model.use_batch_norm": False,
        "config.model.resample_halo": 8,
        "config.model.negative_slope": 0.01,
        "config.runtime.deterministic": False,
        "config.runtime.device": "auto",
        "config.runtime.overwrite": False,
        "config.runtime.source_snapshot": (
            "runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/"
            "_provenance/source_snapshot.zip"
        ),
        "config.runtime.source_snapshot_sha256": (
            "73a64bfda1a3e6ba63eab3a372e1cd6b64d22fda76c0405dfd5facaa765fdb33"
        ),
        **EXPECTED_EXECUTION,
    }
    if spec.lambda_orbit is not None:
        protocol.update(
            {
                "config.training.lambda_orbit": spec.lambda_orbit,
                "config.training.normalize_by_epsilon": True,
                "config.training.orbit_eta": 1e-6,
            }
        )
    if fresh_runs:
        for key in (
            "dataset_sha256", "config.runtime.source_snapshot",
            "config.runtime.source_snapshot_sha256",
        ):
            protocol.pop(key)
        protocol["environment.omp_num_threads"] = "1"
        protocol["environment.mkl_num_threads"] = "1"
    return protocol


def _load_method(
    run_root: Path,
    spec: MethodSpec,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
    *, fresh_runs: bool = False,
) -> pd.DataFrame:
    method_root = run_root / spec.relative_dir
    df = pd.DataFrame(collect_run_rows(method_root))
    context = f"{spec.method} under {method_root}"
    if df.empty:
        raise CNO2dProtocolError(f"{context}: no completed test_metrics.json files")
    _validate_seeds(df, context, expected_seeds)
    for column, expected in _expected_protocol(spec, fresh_runs=fresh_runs).items():
        _validate_constant(df, column, expected, context)
    _validate_absent(df, "config.training.stop_after_epochs", context)
    for column in NONEMPTY_PROVENANCE_FIELDS:
        if fresh_runs and column == "environment.git_commit":
            continue
        _validate_nonempty_consistent(df, column, context)
    for column in CONSISTENT_PROVENANCE_FIELDS:
        _validate_consistent(df, column, context)
    for _, row in df.iterrows():
        if int(row["seed"]) != int(row["config.seed"]):
            raise CNO2dProtocolError(
                f"{context}: metrics/config seed mismatch in {row['run_dir']}"
            )
        if row.get("config_hash") != row.get("environment.config_hash"):
            raise CNO2dProtocolError(
                f"{context}: metrics/environment config hash mismatch in {row['run_dir']}"
            )
    _validate_finite_metrics(df, context)
    if fresh_runs:
        _validate_optional_constant(df, "config.runtime.resume", False, context)
        _validate_absent(df, "config.runtime.source_snapshot", context)
        _validate_absent(df, "config.runtime.source_snapshot_sha256", context)
        _validate_nonempty_consistent(df, "source_sha256", context)
    out = _annotate_authorized_resumes(df, spec, context, historical=not fresh_runs)
    out["method_label"] = spec.label
    out["optimizer_steps"] = spec.steps_per_epoch * 150
    out["model_forwards_per_epoch"] = spec.model_forwards_per_epoch
    out["model_forwards_total"] = spec.model_forwards_per_epoch * 150
    out["backward_passes_total"] = spec.steps_per_epoch * 150
    return out.sort_values("seed").reset_index(drop=True)


def build_run_frame(
    run_root: Path,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
    *, fresh_runs: bool = False,
) -> pd.DataFrame:
    frames = [
        _load_method(run_root, spec, expected_seeds=expected_seeds, fresh_runs=fresh_runs)
        for spec in METHOD_SPECS
    ]
    df = pd.concat(frames, ignore_index=True)
    for column in NONEMPTY_PROVENANCE_FIELDS:
        if fresh_runs and column == "environment.git_commit":
            continue
        _validate_nonempty_consistent(df, column, "CNO2d campaign")
    for column in CONSISTENT_PROVENANCE_FIELDS:
        _validate_consistent(df, column, "CNO2d campaign")
    if fresh_runs:
        from otno.reporting import validate_fresh_run_provenance

        try:
            validate_fresh_run_provenance(df, source_root=ROOT)
        except (OSError, ValueError) as exc:
            raise CNO2dProtocolError(str(exc)) from exc
    order = {spec.method: index for index, spec in enumerate(METHOD_SPECS)}
    df["_order"] = df["method"].map(order)
    return df.sort_values(["_order", "seed"]).drop(columns="_order")


def aggregate_runs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        group = run_df[run_df["method"] == spec.method].sort_values("seed")
        row: dict[str, Any] = {
            "method": spec.method,
            "method_label": spec.label,
            "steps_per_epoch": spec.steps_per_epoch,
            "optimizer_steps": spec.steps_per_epoch * 150,
            "model_forwards_per_epoch": spec.model_forwards_per_epoch,
            "model_forwards_total": spec.model_forwards_per_epoch * 150,
            "backward_passes_total": spec.steps_per_epoch * 150,
            "seed_count": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in group["seed"]),
            "parameters": int(group["parameters"].iloc[0]),
            "checkpoint_resumed_count": int(group["checkpoint_resumed"].sum()),
            "checkpoint_resumed_seeds": ",".join(
                str(int(seed))
                for seed in group.loc[group["checkpoint_resumed"], "seed"]
            ),
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        uninterrupted_wall = pd.to_numeric(
            group.loc[~group["checkpoint_resumed"], "train_wall_seconds"],
            errors="raise",
        )
        row["train_wall_seconds_uninterrupted_count"] = int(
            uninterrupted_wall.shape[0]
        )
        row["train_wall_seconds_uninterrupted_mean"] = float(
            uninterrupted_wall.mean()
        )
        row["train_wall_seconds_uninterrupted_std"] = float(
            uninterrupted_wall.std(ddof=1)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def paired_primary(
    run_df: pd.DataFrame,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> pd.DataFrame:
    reference = run_df[run_df["method"] == "aug"]
    candidate = run_df[run_df["method"] == "aug_orbit"]
    reference_resumed_seeds = ",".join(
        str(int(seed))
        for seed in reference.loc[reference["checkpoint_resumed"], "seed"]
    )
    candidate_resumed_seeds = ",".join(
        str(int(seed))
        for seed in candidate.loc[candidate["checkpoint_resumed"], "seed"]
    )
    rows: list[dict[str, Any]] = []
    for metric in CORE_METRICS:
        paired = (
            reference[["seed", metric]]
            .rename(columns={metric: "augmentation"})
            .merge(
                candidate[["seed", metric]].rename(columns={metric: "augmentation_loco"}),
                on="seed",
                how="inner",
                validate="one_to_one",
            )
            .sort_values("seed")
        )
        seeds = tuple(int(seed) for seed in paired["seed"])
        if seeds != expected_seeds:
            raise CNO2dProtocolError(
                f"primary paired comparison for {metric}: expected "
                f"{expected_seeds}, found {seeds}"
            )
        differences = paired["augmentation_loco"] - paired["augmentation"]
        n = int(differences.shape[0])
        delta = float(differences.mean())
        delta_std = float(differences.std(ddof=1))
        delta_sem = delta_std / math.sqrt(n)
        half_width = float(student_t.ppf(0.975, n - 1)) * delta_sem
        reference_mean = float(paired["augmentation"].mean())
        candidate_mean = float(paired["augmentation_loco"].mean())
        rows.append(
            {
                "comparison": "augmentation_plus_loco_minus_augmentation",
                "difference_direction": "candidate_minus_reference",
                "metric": metric,
                "seed_count": n,
                "seeds": ",".join(str(seed) for seed in expected_seeds),
                "reference_checkpoint_resumed_seeds": reference_resumed_seeds,
                "candidate_checkpoint_resumed_seeds": candidate_resumed_seeds,
                "reference_mean": reference_mean,
                "candidate_mean": candidate_mean,
                "paired_delta_mean": delta,
                "paired_delta_std": delta_std,
                "paired_delta_sem": delta_sem,
                "paired_delta_ci95_low": delta - half_width,
                "paired_delta_ci95_high": delta + half_width,
                "relative_reduction_pct": (
                    100.0 * (reference_mean - candidate_mean) / reference_mean
                ),
                "ci_excludes_zero": bool(delta - half_width > 0 or delta + half_width < 0),
            }
        )
    return pd.DataFrame(rows)


def _fmt_pm(mean: float, std: float, *, bold: bool = False) -> str:
    value = rf"{mean:.4f}\pm{std:.4f}"
    return rf"\(\mathbf{{{value}}}\)" if bold else rf"\({value}\)"


def latex_table(aggregate: pd.DataFrame) -> str:
    rows = {str(row["method"]): row for _, row in aggregate.iterrows()}
    seed_values = str(rows["baseline"]["seeds"]).replace(",", ", ")
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        (
            r"\caption{CNO2d reproduction on the 2\%-label N64 Galilean task over "
            f"the same training seeds as the FNO comparison, "
            rf"\(\{{{seed_values}\}}\). All rows use the same 2,667,743-parameter "
            r"architecture. The augmentation and augmentation-plus-normalized-LOCO "
            r"rows are matched at 12 batch-level model evaluations per epoch; optimizer "
            r"and backward-pass counts differ and are shown. Entries are mean $\pm$ "
            r"sample standard deviation. Latency and wall time remain available in the "
            r"per-run manifest; these eight-thread CPU training runs are not used for "
            r"cross-backbone efficiency comparisons. Baseline seed 23 and "
            r"augmentation-plus-LOCO seed 59 were resumed from fully saved checkpoints "
            r"after epochs 108 and 103, respectively. Their post-resume minibatch "
            r"streams need not reproduce uninterrupted runs, and their segment-local "
            r"wall times are excluded from timing summaries.}"
        ),
        r"\label{tab:cno2d-backbone}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        "Method & Steps/epoch & Model fwds., total & Bwd. passes, total & ID $L^2$ & "
        "Orbit OOD $L^2$ & Eq. defect \\\\",
        r"\midrule",
    ]
    for spec in METHOD_SPECS:
        row = rows[spec.method]
        bold = spec.method == "aug_orbit"
        relative = _fmt_pm(row["relative_l2_mean"], row["relative_l2_std"], bold=bold)
        orbit_ood = _fmt_pm(
            row["orbit_ood_relative_l2_mean"],
            row["orbit_ood_relative_l2_std"],
            bold=bold,
        )
        defect = _fmt_pm(
            row["equivariance_defect_relative_mean"],
            row["equivariance_defect_relative_std"],
            bold=bold,
        )
        lines.append(
            f"{row['method_label']} & {int(row['steps_per_epoch'])} & "
            f"{int(row['model_forwards_total'])} & {int(row['backward_passes_total'])} & "
            f"{relative} & {orbit_ood} & {defect} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table*}", ""])
    return "\n".join(lines)


def _manifest(run_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "run_dir",
        "method",
        "method_label",
        "seed",
        "relative_l2",
        "orbit_ood_relative_l2",
        "equivariance_defect_relative",
        "best_val_relative_l2",
        "latency_ms_per_sample",
        "train_wall_seconds",
        "train_wall_seconds_scope",
        "checkpoint_resumed",
        "resume_last_completed_epoch",
        "resume_source_last_checkpoint_sha256",
        "resume_source_best_checkpoint_sha256",
        "resume_authorization_source",
        "optimizer_steps",
        "model_forwards_total",
        "backward_passes_total",
        "parameters",
        "dataset_sha256",
        "config_hash",
        "environment.git_commit",
        "environment.platform",
        "environment.python",
        "environment.torch",
        "environment.cuda_available",
        "environment.cuda_version",
        "environment.cudnn_version",
        "environment.omp_num_threads",
        "environment.mkl_num_threads",
        "device",
        "deterministic",
        "config_path",
    ]
    if "source_sha256" in run_df.columns:
        columns.append("source_sha256")
    return run_df[columns].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate completed CNO2d backbone runs; never trains or evaluates models."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--fresh-runs", action="store_true",
        help="Audit new direct CPU/one-thread runs against their actual source and dataset hashes.",
    )
    parser.add_argument(
        "--fno-runs",
        type=Path,
        default=DEFAULT_FNO_RUNS,
        help="Primary FNO per-run CSV whose exact seed set CNO2d must match.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        fno_seeds = load_fno_reference_seeds(args.fno_runs)
        run_df = build_run_frame(args.run_root, expected_seeds=fno_seeds, fresh_runs=args.fresh_runs)
        aggregate = aggregate_runs(run_df)
        paired = paired_primary(run_df, expected_seeds=fno_seeds)
    except CNO2dProtocolError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "runs": Path(f"{args.out_prefix}.runs.csv"),
        "aggregate": Path(f"{args.out_prefix}.aggregate.csv"),
        "paired": Path(f"{args.out_prefix}.paired.csv"),
        "tex": Path(f"{args.out_prefix}.tex"),
    }
    outputs["runs"].write_text(
        _manifest(run_df).to_csv(index=False, lineterminator="\n"), encoding="utf-8"
    )
    outputs["aggregate"].write_text(
        aggregate.to_csv(index=False, lineterminator="\n"), encoding="utf-8"
    )
    outputs["paired"].write_text(
        paired.to_csv(index=False, lineterminator="\n"), encoding="utf-8"
    )
    outputs["tex"].write_text(latex_table(aggregate), encoding="utf-8")
    for path in outputs.values():
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
