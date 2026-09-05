#!/usr/bin/env python
"""Validate and aggregate the five-seed DeepONet backbone experiment."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from scipy.stats import t as student_t

from otno.reporting import RunValidation, collect_run_rows, validate_fresh_run_provenance


EXPECTED_SEEDS = (23, 31, 47, 59, 71)
EXPECTED_DATASET_SHA256 = "defc9703dbc41494fabf9fcfe9e3408eb2e1c3890814e47fe03d0f6b849e3f50"
EXPECTED_EXECUTION = {
    "environment.omp_num_threads": "1",
    "environment.mkl_num_threads": "1",
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
DEFAULT_RUN_ROOT = Path("runs/ablations/2d_galilean_n64_2pct_deeponet_5seed/fraction_0.02")
DEFAULT_OUT_PREFIX = Path("runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed")
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
    "train_wall_seconds",
)


class DeepONetProtocolError(ValueError):
    """Raised when cached runs do not exactly match the declared protocol."""


validation = RunValidation(DeepONetProtocolError)


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


METHOD_SPECS = (
    MethodSpec("baseline", "DeepONet", "baseline_steps_4", 4, 1, None),
    MethodSpec(
        "orbit",
        "DeepONet + normalized LOCO",
        "orbit_lambda_0.05_steps_4",
        4,
        2,
        0.05,
    ),
    MethodSpec("aug", "DeepONet + aug.", "aug_steps_6", 6, 2, None),
    MethodSpec(
        "aug_orbit",
        "DeepONet + aug. + normalized LOCO",
        "aug_loco_lambda_0.1_steps_4",
        4,
        3,
        0.10,
    ),
)


def load_fno_reference_seeds(path: Path) -> tuple[int, ...]:
    return validation.fno_reference_seeds(path, [spec.method for spec in METHOD_SPECS])


def _expected_protocol(spec: MethodSpec) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "method": spec.method,
        "model_name": "deeponet2d",
        "parameters": 2_688_033,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "dataset_kind": "navier_stokes_vorticity2d_boosted",
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
        "config.model.name": "deeponet2d",
        "config.model.in_channels": 3,
        "config.model.out_channels": 1,
        "config.model.grid_height": 64,
        "config.model.grid_width": 64,
        "config.model.branch_channels": [32, 64, 128, 256],
        "config.model.branch_fc_hidden": 480,
        "config.model.trunk_hidden": 256,
        "config.model.trunk_depth": 3,
        "config.model.latent_dim": 256,
        "config.model.coordinate_modes": 12,
        "config.runtime.deterministic": False,
        "config.runtime.device": "auto",
        "config.runtime.overwrite": False,
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
    return protocol


def _load_method(
    run_root: Path,
    spec: MethodSpec,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
    *,
    fresh_runs: bool = False,
) -> pd.DataFrame:
    method_root = run_root / spec.relative_dir
    df = pd.DataFrame(collect_run_rows(method_root))
    context = f"{spec.method} under {method_root}"
    if df.empty:
        raise DeepONetProtocolError(f"{context}: no completed test_metrics.json files")
    validation.seeds(df, context, expected_seeds)
    protocol = _expected_protocol(spec)
    if fresh_runs:
        protocol.pop("dataset_sha256")
    for column, expected in protocol.items():
        validation.constant(df, column, expected, context)
    validation.optional_constant(df, "config.runtime.resume", False, context)
    validation.absent(df, "config.training.stop_after_epochs", context)
    for column in NONEMPTY_PROVENANCE_FIELDS:
        if fresh_runs and column == "environment.git_commit":
            continue
        validation.nonempty_consistent(df, column, context)
    for column in CONSISTENT_PROVENANCE_FIELDS:
        validation.consistent(df, column, context)
    for _, row in df.iterrows():
        if int(row["seed"]) != int(row["config.seed"]):
            raise DeepONetProtocolError(
                f"{context}: metrics/config seed mismatch in {row['run_dir']}"
            )
    validation.finite_metrics(df, context, AGGREGATE_METRICS)
    out = df.copy()
    out["method_label"] = spec.label
    out["optimizer_steps"] = spec.steps_per_epoch * 150
    out["model_forwards_per_epoch"] = spec.model_forwards_per_epoch
    out["model_forwards_total"] = spec.model_forwards_per_epoch * 150
    out["backward_passes_total"] = spec.steps_per_epoch * 150
    return out.sort_values("seed").reset_index(drop=True)


def build_run_frame(
    run_root: Path,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
    *,
    fresh_runs: bool = False,
) -> pd.DataFrame:
    frames = [
        _load_method(run_root, spec, expected_seeds=expected_seeds, fresh_runs=fresh_runs)
        for spec in METHOD_SPECS
    ]
    df = pd.concat(frames, ignore_index=True)
    for column in NONEMPTY_PROVENANCE_FIELDS:
        if fresh_runs and column == "environment.git_commit":
            continue
        validation.nonempty_consistent(df, column, "DeepONet campaign")
    for column in CONSISTENT_PROVENANCE_FIELDS:
        validation.consistent(df, column, "DeepONet campaign")
    if fresh_runs:
        try:
            validate_fresh_run_provenance(df, source_root=ROOT)
        except (OSError, ValueError) as exc:
            raise DeepONetProtocolError(str(exc)) from exc
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
        }
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(group[metric], errors="raise")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_primary(
    run_df: pd.DataFrame,
    expected_seeds: tuple[int, ...] = EXPECTED_SEEDS,
) -> pd.DataFrame:
    reference = run_df[run_df["method"] == "aug"]
    candidate = run_df[run_df["method"] == "aug_orbit"]
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
            raise DeepONetProtocolError(
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
            r"\caption{DeepONet reproduction on the 2\%-label N64 Galilean task over "
            f"the same training seeds as the FNO comparison, "
            rf"\(\{{{seed_values}\}}\). All rows use the same 2,688,033-parameter "
            r"architecture. The "
            r"augmentation and augmentation-plus-normalized-LOCO rows are matched at 12 "
            r"batch-level model evaluations per epoch; optimizer and backward-pass counts "
            r"differ and are shown. Entries are mean $\pm$ sample standard deviation. "
            r"Latency and wall time remain available in the per-run manifest; seed shards "
            r"ran concurrently, so they are not used as efficiency comparisons.}"
        ),
        r"\label{tab:deeponet-backbone}",
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
        relative = _fmt_pm(
            row["relative_l2_mean"], row["relative_l2_std"], bold=bold
        )
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
            f"{relative} & {orbit_ood} & {defect} \\\\"
        )
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
        description="Validate and aggregate recorded DeepONet backbone metrics."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    parser.add_argument(
        "--fresh-runs", action="store_true",
        help="Validate new runs against their source manifest and dataset hashes.",
    )
    parser.add_argument(
        "--fno-runs",
        type=Path,
        default=DEFAULT_FNO_RUNS,
        help="Primary FNO per-run CSV whose exact seed set DeepONet must match.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        fno_seeds = load_fno_reference_seeds(args.fno_runs)
        run_df = build_run_frame(args.run_root, expected_seeds=fno_seeds, fresh_runs=args.fresh_runs)
        aggregate = aggregate_runs(run_df)
        paired = paired_primary(run_df, expected_seeds=fno_seeds)
    except DeepONetProtocolError as exc:
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
