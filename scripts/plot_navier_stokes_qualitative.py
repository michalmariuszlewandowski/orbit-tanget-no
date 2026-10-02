#!/usr/bin/env python
"""Generate Navier--Stokes field figures from a declared checkpoint protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from otno.data.datasets import load_tensor_dataset
from otno.models import build_model
from otno.plotting.field_analysis import (
    gradient_error_summary,
    predict_fields,
    relative_defects,
    relative_errors,
    representative_index,
)
from otno.plotting.field_diagnostics import (
    plot_equivariance_residuals,
    plot_fixed_stress_diagnostics,
)
from otno.plotting.field_panels import plot_prediction_fields
from otno.symmetry.transforms import NavierStokes2DGalilean
from otno.utils import file_sha256, get_device

DEFAULT_AUG_CHECKPOINT = (
    "runs/ablations/2d_galilean_n64_label_efficiency_compute_matched/"
    "fraction_0.02/aug_steps_6/seed_23/checkpoints/best.pt"
)
DEFAULT_LOCO_CHECKPOINT = (
    "runs/ablations/2d_galilean_n64_2pct_lambda_robustness/"
    "fraction_0.02/aug_orbit_lambda_0.1_steps_4/seed_23/checkpoints/best.pt"
)
DEFAULT_FIGURE_CONFIG = "figure_specs/2d_galilean_n64_id_ood_seed23.yaml"


def _load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", checkpoint.get("base_config"))
    if config is None:
        raise KeyError(f"Checkpoint has no config: {checkpoint_path}")
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    checkpoint_info = {
        "epoch": int(checkpoint.get("epoch", -1)),
        "best_val_relative_l2": float(checkpoint.get("best_val", float("nan"))),
        "config_hash": checkpoint.get("meta", {}).get("config_hash"),
    }
    return model, config, checkpoint_info


def _validate_checkpoint_configs(
    aug_config: dict[str, Any], loco_config: dict[str, Any]
) -> dict[str, Any]:
    aug_training = aug_config.get("training", {})
    loco_training = loco_config.get("training", {})
    if str(aug_training.get("method", "")).lower() not in {"aug", "augmentation"}:
        raise ValueError("The augmentation checkpoint is not an augmentation-only run")
    if str(loco_training.get("method", "")).lower() not in {"aug_orbit", "orbit_aug"}:
        raise ValueError("The LOCO checkpoint is not an augmentation-plus-orbit run")
    for key in ["seed", "dataset", "model", "symmetry"]:
        if aug_config.get(key) != loco_config.get(key):
            raise ValueError(f"Checkpoint configs disagree on {key!r}")
    aug_fraction = float(aug_training.get("data_fraction", 1.0))
    loco_fraction = float(loco_training.get("data_fraction", 1.0))
    if not np.isclose(aug_fraction, loco_fraction):
        raise ValueError("Checkpoint configs use different labeled fractions")
    return {
        "seed": int(aug_config.get("seed", -1)),
        "data_fraction": aug_fraction,
        "model_name": str(aug_config.get("model", {}).get("name", "")),
        "augmentation_method": str(aug_training.get("method")),
        "augmentation_epochs": int(aug_training.get("epochs", -1)),
        "augmentation_steps_per_epoch": int(aug_training.get("steps_per_epoch", -1)),
        "augmentation_lambda_aug": float(aug_training.get("lambda_aug", float("nan"))),
        "loco_method": str(loco_training.get("method")),
        "loco_epochs": int(loco_training.get("epochs", -1)),
        "loco_steps_per_epoch": int(loco_training.get("steps_per_epoch", -1)),
        "loco_lambda_aug": float(loco_training.get("lambda_aug", float("nan"))),
        "loco_lambda_orbit": float(loco_training.get("lambda_orbit", float("nan"))),
        "loco_normalize_by_epsilon": bool(loco_training.get("normalize_by_epsilon", False)),
        "loco_orbit_eta": float(loco_training.get("orbit_eta", float("nan"))),
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _metadata(
    *,
    fields: dict[str, torch.Tensor],
    errors: dict[str, torch.Tensor],
    index: int,
    median_improvement: float,
    boost: tuple[float, float],
    dataset_path: Path,
    aug_checkpoint: Path,
    loco_checkpoint: Path,
    aug_checkpoint_info: dict[str, Any],
    loco_checkpoint_info: dict[str, Any],
    run_spec: dict[str, Any],
    dataset_metadata: dict[str, Any],
    base_frame: tuple[float, float],
    plot_limits: dict[str, float],
    batch_size: int,
    device: torch.device,
    figure_config_path: Path,
) -> dict[str, Any]:
    selected = {name: float(values[index]) for name, values in errors.items()}
    selected["ood_error_reduction"] = selected["aug_ood"] - selected["loco_ood"]
    selected["id_error_reduction"] = selected["aug_id"] - selected["loco_id"]
    means = {name: float(values.mean()) for name, values in errors.items()}
    means["ood_error_reduction"] = means["aug_ood"] - means["loco_ood"]
    means["id_error_reduction"] = means["aug_id"] - means["loco_id"]
    means["aug_ood_to_id_ratio"] = means["aug_ood"] / means["aug_id"]
    means["loco_ood_to_id_ratio"] = means["loco_ood"] / means["loco_id"]
    improvement = errors["aug_ood"] - errors["loco_ood"]
    selected_error_maps = {
        "aug_id": (fields["aug_id"][index] - fields["target_id"][index]).abs(),
        "loco_id": (fields["loco_id"][index] - fields["target_id"][index]).abs(),
        "aug_ood": (fields["aug_ood"][index] - fields["target_ood"][index]).abs(),
        "loco_ood": (fields["loco_ood"][index] - fields["target_ood"][index]).abs(),
    }
    pixel_summaries = {}
    for regime in ["id", "ood"]:
        aug_error = selected_error_maps[f"aug_{regime}"]
        loco_error = selected_error_maps[f"loco_{regime}"]
        pixel_summaries[regime] = {
            "fraction_loco_lower_absolute_error": float((loco_error < aug_error).float().mean()),
            "aug_mean_absolute_error": float(aug_error.mean()),
            "loco_mean_absolute_error": float(loco_error.mean()),
            "aug_p95_absolute_error": float(torch.quantile(aug_error.reshape(-1), 0.95)),
            "loco_p95_absolute_error": float(torch.quantile(loco_error.reshape(-1), 0.95)),
        }
    return {
        "selection_rule": (
            f"Training seed {run_spec['seed']}; outcome-conditioned selection of the held-out "
            "test example whose fixed-boost OOD relative-L2 improvement is closest to the "
            "median over all test examples. The example is illustrative, not inferential."
        ),
        "selection_is_outcome_conditioned": True,
        "selected_test_index_zero_based": index,
        "num_test_examples": int(fields["target_id"].shape[0]),
        "fixed_relative_boost": [float(boost[0]), float(boost[1])],
        "fixed_relative_boost_norm": float(np.linalg.norm(boost)),
        "base_ambient_velocity": [float(base_frame[0]), float(base_frame[1])],
        "transformed_ambient_velocity": [
            float(base_frame[0] + boost[0]),
            float(base_frame[1] + boost[1]),
        ],
        "median_ood_relative_l2_absolute_reduction": median_improvement,
        "ood_relative_l2_reduction_quantiles": {
            "q10": float(torch.quantile(improvement, 0.10)),
            "q50": float(torch.quantile(improvement, 0.50)),
            "q90": float(torch.quantile(improvement, 0.90)),
        },
        "selected_relative_l2": selected,
        "full_test_fixed_boost_mean_relative_l2": means,
        "selected_pixel_error_summary": pixel_summaries,
        "selected_gradient_error_summary": {
            "aug_id": gradient_error_summary(fields["target_id"][index], fields["aug_id"][index]),
            "loco_id": gradient_error_summary(fields["target_id"][index], fields["loco_id"][index]),
            "aug_ood": gradient_error_summary(
                fields["target_ood"][index], fields["aug_ood"][index]
            ),
            "loco_ood": gradient_error_summary(
                fields["target_ood"][index], fields["loco_ood"][index]
            ),
        },
        "dataset_split": "test",
        "dataset_tensor_shapes": {
            "input": list(fields["target_id"].shape[:-1]) + [3],
            "target": list(fields["target_id"].shape),
        },
        "dataset_metadata": dataset_metadata,
        "dataset": _repo_relative(dataset_path),
        "dataset_sha256": file_sha256(dataset_path),
        "augmentation_checkpoint": _repo_relative(aug_checkpoint),
        "augmentation_checkpoint_sha256": file_sha256(aug_checkpoint),
        "augmentation_checkpoint_info": aug_checkpoint_info,
        "loco_checkpoint": _repo_relative(loco_checkpoint),
        "loco_checkpoint_sha256": file_sha256(loco_checkpoint),
        "loco_checkpoint_info": loco_checkpoint_info,
        "validated_run_spec": run_spec,
        "generation_device": str(device),
        "generation_batch_size": int(batch_size),
        "figure_config": _repo_relative(figure_config_path),
        "figure_config_sha256": file_sha256(figure_config_path),
        "plot_script": _repo_relative(Path(__file__)),
        "plot_script_sha256": file_sha256(Path(__file__)),
        "plot_color_limits": plot_limits,
        "target_construction": (
            "The OOD target is the deterministic periodic Galilean evaluation output "
            "action applied to the cached held-out target; solver closure is validated "
            "separately."
        ),
    }


def _figure_output_records(prefixes: list[Path]) -> list[dict[str, Any]]:
    records = []
    for prefix in prefixes:
        for suffix in [".png", ".pdf"]:
            path = prefix.with_suffix(suffix)
            record: dict[str, Any] = {
                "path": _repo_relative(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            if suffix == ".png":
                record["pixel_shape"] = list(plt.imread(path).shape)
            records.append(record)
    return records


def _value_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, float):
        try:
            return bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            return False
    return actual == expected


def _validate_frozen_figure_inputs(
    figure_config: dict[str, Any],
    *,
    aug_checkpoint: Path,
    loco_checkpoint: Path,
    dataset_path: Path,
    run_spec: dict[str, Any],
) -> None:
    expected_hashes = {
        "augmentation_checkpoint_sha256": file_sha256(aug_checkpoint),
        "loco_checkpoint_sha256": file_sha256(loco_checkpoint),
        "dataset_sha256": file_sha256(dataset_path),
    }
    for key, actual in expected_hashes.items():
        expected = figure_config.get(key)
        if not expected:
            raise ValueError(f"Figure config is missing frozen input hash {key!r}")
        if actual != expected:
            raise ValueError(f"Figure config {key}={expected!r}, but the file hash is {actual!r}")

    expected_run_spec = figure_config.get("expected_run_spec")
    if not isinstance(expected_run_spec, dict) or not expected_run_spec:
        raise ValueError("Figure config must define a non-empty expected_run_spec mapping")
    failures = []
    for key, expected in expected_run_spec.items():
        actual = run_spec.get(key)
        if not _value_matches(actual, expected):
            failures.append(f"{key}: expected {expected!r}, found {actual!r}")
    if failures:
        raise ValueError("Figure checkpoint protocol mismatch: " + "; ".join(failures))


def _parse_arguments() -> argparse.Namespace:
    """Keep command options separate from checkpoint evaluation and rendering."""
    parser = argparse.ArgumentParser(
        description="Plot representative ID/OOD Navier--Stokes predictions and error heatmaps."
    )
    parser.add_argument("--figure-config", default=DEFAULT_FIGURE_CONFIG)
    parser.add_argument("--aug-checkpoint")
    parser.add_argument("--loco-checkpoint")
    parser.add_argument("--boost-x", type=float)
    parser.add_argument("--boost-y", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--out-prefix",
    )
    parser.add_argument(
        "--metadata-out",
    )
    parser.add_argument("--diagnostics-out-prefix")
    parser.add_argument("--equivariance-out-prefix")
    return parser.parse_args()


def _figure_config(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    """Resolve command overrides against the saved figure specification."""

    figure_config_path = ROOT / args.figure_config
    with figure_config_path.open("r", encoding="utf-8") as handle:
        figure_config = yaml.safe_load(handle) or {}
    return figure_config_path, figure_config


def _write_metadata(metadata: dict[str, Any], path: Path, prefixes: list[Path]) -> None:
    """Record the code and rendered files used to generate this figure set."""
    metadata["software_versions"] = {
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "yaml": yaml.__version__,
    }
    sources = [Path(__file__)] + sorted((SRC / "otno" / "plotting").glob("*.py"))
    metadata["plot_sources"] = {_repo_relative(source): file_sha256(source) for source in sources}
    metadata["generated_figure_outputs"] = _figure_output_records(prefixes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = _parse_arguments()
    figure_config_path, figure_config = _figure_config(args)
    configured_boost = figure_config.get("fixed_relative_boost", [0.50, 0.00])
    aug_checkpoint_arg = args.aug_checkpoint or figure_config.get(
        "augmentation_checkpoint", DEFAULT_AUG_CHECKPOINT
    )
    loco_checkpoint_arg = args.loco_checkpoint or figure_config.get(
        "loco_checkpoint", DEFAULT_LOCO_CHECKPOINT
    )
    boost = (
        float(args.boost_x if args.boost_x is not None else configured_boost[0]),
        float(args.boost_y if args.boost_y is not None else configured_boost[1]),
    )
    batch_size = int(args.batch_size or figure_config.get("batch_size", 16))
    device_name = args.device or figure_config.get("device", "cpu")
    out_prefix_arg = args.out_prefix or figure_config.get(
        "out_prefix", "runs/figures/2d_galilean_n64_id_ood_fields"
    )
    metadata_out_arg = args.metadata_out or figure_config.get(
        "metadata_out", "runs/paper_tables/2d_galilean_n64_id_ood_fields.json"
    )
    diagnostics_out_arg = args.diagnostics_out_prefix or figure_config.get(
        "diagnostics_out_prefix",
        "runs/figures/2d_galilean_n64_fixed_stress_diagnostics",
    )
    equivariance_out_arg = args.equivariance_out_prefix or figure_config.get(
        "equivariance_out_prefix",
        "runs/figures/2d_galilean_n64_equivariance_residuals",
    )

    device = get_device(device_name)
    aug_checkpoint = ROOT / aug_checkpoint_arg
    loco_checkpoint = ROOT / loco_checkpoint_arg
    aug_model, aug_config, aug_checkpoint_info = _load_model(aug_checkpoint, device)
    loco_model, loco_config, loco_checkpoint_info = _load_model(loco_checkpoint, device)
    run_spec = _validate_checkpoint_configs(aug_config, loco_config)
    aug_dataset_path = ROOT / aug_config["dataset"]["path"]
    loco_dataset_path = ROOT / loco_config["dataset"]["path"]
    if aug_dataset_path.resolve() != loco_dataset_path.resolve():
        raise ValueError("The two checkpoints refer to different datasets")
    _validate_frozen_figure_inputs(
        figure_config,
        aug_checkpoint=aug_checkpoint,
        loco_checkpoint=loco_checkpoint,
        dataset_path=aug_dataset_path,
        run_spec=run_spec,
    )
    dataset, dataset_metadata = load_tensor_dataset(aug_dataset_path, "test")
    symmetry = aug_config["symmetry"]
    transform = NavierStokes2DGalilean(
        max_boost=max(abs(boost[0]), abs(boost[1])),
        final_time=float(symmetry.get("final_time", 0.5)),
        length=float(symmetry.get("length", 1.0)),
        boost_x_channel=int(symmetry.get("boost_x_channel", 1)),
        boost_y_channel=int(symmetry.get("boost_y_channel", 2)),
    )
    fields = predict_fields(
        aug_model,
        loco_model,
        dataset,
        transform,
        boost=boost,
        batch_size=batch_size,
        device=device,
    )
    errors = relative_errors(fields)
    defects = relative_defects(fields)
    index, median_improvement = representative_index(errors)
    out_prefix = ROOT / out_prefix_arg
    plot_limits = plot_prediction_fields(
        fields, errors, index=index, boost=boost, out_prefix=out_prefix
    )
    diagnostics_out_prefix = ROOT / diagnostics_out_arg
    diagnostics = plot_fixed_stress_diagnostics(
        fields,
        errors,
        index=index,
        boost=boost,
        out_prefix=diagnostics_out_prefix,
    )
    equivariance_out_prefix = ROOT / equivariance_out_arg
    equivariance_diagnostics = plot_equivariance_residuals(
        fields,
        defects,
        index=index,
        boost=boost,
        out_prefix=equivariance_out_prefix,
    )
    base_frame = (
        float(dataset.a[index, 0, 0, int(symmetry.get("boost_x_channel", 1))]),
        float(dataset.a[index, 0, 0, int(symmetry.get("boost_y_channel", 2))]),
    )

    metadata = _metadata(
        fields=fields,
        errors=errors,
        index=index,
        median_improvement=median_improvement,
        boost=boost,
        dataset_path=aug_dataset_path,
        aug_checkpoint=aug_checkpoint,
        loco_checkpoint=loco_checkpoint,
        aug_checkpoint_info=aug_checkpoint_info,
        loco_checkpoint_info=loco_checkpoint_info,
        run_spec=run_spec,
        dataset_metadata=dataset_metadata,
        base_frame=base_frame,
        plot_limits=plot_limits,
        batch_size=batch_size,
        device=device,
        figure_config_path=figure_config_path,
    )
    metadata["fixed_stress_diagnostics"] = diagnostics
    metadata["equivariance_diagnostics"] = equivariance_diagnostics
    metadata_path = ROOT / metadata_out_arg
    _write_metadata(
        metadata, metadata_path, [out_prefix, diagnostics_out_prefix, equivariance_out_prefix]
    )
    print(f"selected held-out test index {index}")
    print(f"wrote {out_prefix.with_suffix('.png')}")
    print(f"wrote {out_prefix.with_suffix('.pdf')}")
    print(f"wrote {diagnostics_out_prefix.with_suffix('.png')}")
    print(f"wrote {diagnostics_out_prefix.with_suffix('.pdf')}")
    print(f"wrote {equivariance_out_prefix.with_suffix('.png')}")
    print(f"wrote {equivariance_out_prefix.with_suffix('.pdf')}")
    print(f"wrote {metadata_path}")


if __name__ == "__main__":
    main()
