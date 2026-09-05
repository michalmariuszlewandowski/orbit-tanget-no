from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_DATASETS = {
    "advection1d",
    "1d_advection",
    "burgers1d",
    "1d_burgers",
    "heat1d_dirichlet",
    "dirichlet_heat1d",
    "kdv1d",
    "1d_kdv",
    "lpsda_kdv1d",
    "navier_stokes_vorticity2d",
    "navier_stokes_vorticity2d_boosted",
    "boosted_navier_stokes_vorticity2d",
    "ns2d",
    "ns2d_boosted",
    "boosted_ns2d",
    "2d_navier_stokes",
    "rmd17_force",
}
_ALLOWED_MODELS = {
    "fno1d",
    "fno_1d",
    "fno2d",
    "fno_2d",
    "deeponet2d",
    "deeponet_2d",
    "cno2d",
    "cno_2d",
    "canonical_fno1d",
    "canonical_fno_1d",
    "translation_canonical_fno1d",
    "phase_canonical_fno1d",
    "galilean_canonical_fno1d",
    "burgers_galilean_canonical_fno1d",
    "canonical_fno2d",
    "canonical_fno_2d",
    "observable_galilean_canonical_fno2d",
    "galilean_canonical_fno2d",
    "pace_style_fno2d",
    "d4_gfno2d",
    "d4_gfno_2d",
    "gfno2d",
    "g_fno2d",
    "g-fno2d",
    "molecule_mlp",
    "molecular_mlp",
}
_ALLOWED_METHODS = {
    "baseline",
    "aug",
    "augmentation",
    "orbit",
    "orb",
    "aug_orbit",
    "orbit_aug",
    "aug_orbit_shuffle",
    "aug_orbit_shuffled",
    "aug_orbit_no_output",
    "aug_orbit_input_only",
    "semi_aug_orbit",
    "mixed_semi_aug_orbit",
    "split_semi_aug_orbit",
    "tangent",
    "tangent_prop",
    "tangent_propagation",
    "aug_tangent",
    "tangent_aug",
}
_ALLOWED_TRANSFORMS = {
    "translation1d",
    "translation_1d",
    "nonperiodic_translation1d",
    "nonperiodic_translation_1d",
    "dirichlet_translation1d",
    "translation2d",
    "translation_2d",
    "burgers1d_galilean",
    "galilean1d",
    "burgers_galilean",
    "kdv1d_galilean",
    "kdv_galilean",
    "navier_stokes2d_galilean",
    "ns2d_galilean",
    "galilean2d",
    "vorticity2d_galilean",
    "d4_scalar2d",
    "d4_scalar_2d",
    "d4_pseudoscalar2d",
    "d4_pseudoscalar_2d",
    "d4_vorticity2d",
    "d4_vorticity_2d",
    "molecular_rigid_motion",
    "rigid_motion3d",
    "se3_molecular",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def recursive_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of base updated recursively by updates."""
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = recursive_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def get_by_path(config: dict[str, Any], dotted_path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    if not dotted_path or dotted_path.startswith(".") or dotted_path.endswith("."):
        raise ValueError(f"Invalid override key: {dotted_path!r}")
    cursor = config
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        if not part:
            raise ValueError(f"Invalid override key: {dotted_path!r}")
        existing = cursor.get(part)
        if existing is None:
            existing = {}
            cursor[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set nested key under non-dict path: {dotted_path!r}")
        cursor = existing
    cursor[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> dict[str, Any]:
    """Parse CLI overrides of the form key.path=value using YAML values."""
    out: dict[str, Any] = {}
    if not items:
        return out
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must have form key=value, got: {item}")
        key, raw_value = item.split("=", 1)
        set_by_path(out, key, yaml.safe_load(raw_value))
    return out


def dotted_overrides_to_nested(overrides: dict[str, Any]) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        set_by_path(nested, key, value)
    return nested


def apply_dotted_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    return recursive_update(base, dotted_overrides_to_nested(overrides))


def _check_positive_int(section: dict[str, Any], section_name: str, key: str) -> None:
    if key in section and int(section[key]) < 1:
        raise ValueError(f"{section_name}.{key} must be a positive integer")


def _check_nonnegative_int(section: dict[str, Any], section_name: str, key: str) -> None:
    if key in section and int(section[key]) < 0:
        raise ValueError(f"{section_name}.{key} must be a non-negative integer")


def _check_nonnegative_float(section: dict[str, Any], section_name: str, key: str) -> None:
    if key in section and float(section[key]) < 0:
        raise ValueError(f"{section_name}.{key} must be non-negative")


def _check_positive_float(section: dict[str, Any], section_name: str, key: str) -> None:
    if key in section and float(section[key]) <= 0:
        raise ValueError(f"{section_name}.{key} must be positive")


def _iter_transform_cfgs(symmetry: dict[str, Any]) -> list[dict[str, Any]]:
    if "transforms" in symmetry:
        return list(symmetry["transforms"])
    return [symmetry]


def validate_config(config: dict[str, Any], *, require_dataset: bool = True) -> None:
    """Fail early on configuration mistakes that would otherwise produce ambiguous runs."""
    dataset: dict[str, Any] = {}
    kind = ""
    if require_dataset:
        if "dataset" not in config:
            raise ValueError("Config requires a dataset section")
        dataset = config["dataset"]
        for key in ("kind", "path"):
            if key not in dataset:
                raise ValueError(f"dataset.{key} is required")
        kind = str(dataset.get("kind", "")).lower()
        if kind and kind not in _ALLOWED_DATASETS:
            raise ValueError(f"Unknown dataset.kind={kind!r}")
        for key in ("n", "num_train", "num_val", "num_test", "solver_batch_size"):
            _check_positive_int(dataset, "dataset", key)
        _check_nonnegative_float(dataset, "dataset", "final_time")
        _check_nonnegative_float(dataset, "dataset", "viscosity")
        if kind in {
            "burgers1d",
            "1d_burgers",
            "kdv1d",
            "1d_kdv",
            "lpsda_kdv1d",
            "navier_stokes_vorticity2d",
            "navier_stokes_vorticity2d_boosted",
            "boosted_navier_stokes_vorticity2d",
            "ns2d",
            "ns2d_boosted",
            "boosted_ns2d",
            "2d_navier_stokes",
        }:
            _check_positive_float(dataset, "dataset", "dt")
        elif "dt" in dataset:
            _check_nonnegative_float(dataset, "dataset", "dt")
        if kind in {
            "navier_stokes_vorticity2d_boosted",
            "boosted_navier_stokes_vorticity2d",
            "ns2d_boosted",
            "boosted_ns2d",
        }:
            _check_nonnegative_float(dataset, "dataset", "max_boost")

    model = config.get("model", {})
    name = str(model.get("name", "fno1d")).lower()
    if name not in _ALLOWED_MODELS:
        raise ValueError(f"Unknown model.name={name!r}")
    for key in (
        "in_channels",
        "out_channels",
        "width",
        "hidden",
        "depth",
        "modes",
        "modes1",
        "modes2",
        "translation_mode",
        "boost_x_channel",
        "boost_y_channel",
        "n_atoms",
        "atoms",
        "grid_size",
        "grid_height",
        "grid_width",
        "branch_fc_hidden",
        "trunk_hidden",
        "trunk_depth",
        "latent_dim",
        "coordinate_modes",
        "n_layers",
        "n_res",
        "n_res_neck",
        "channel_multiplier",
        "lift_project_channels",
        "resample_halo",
    ):
        _check_positive_int(model, "model", key)
    _check_nonnegative_float(model, "model", "negative_slope")
    if "branch_channels" in model:
        channels = model["branch_channels"]
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError("model.branch_channels must be a non-empty sequence")
        if any(int(channel) < 1 for channel in channels):
            raise ValueError("model.branch_channels values must be positive integers")

    training = config.get("training", {})
    method = str(training.get("method", "baseline")).lower()
    if method not in _ALLOWED_METHODS:
        raise ValueError(f"Unknown training.method={method!r}")
    for key in (
        "epochs",
        "batch_size",
        "eval_every",
        "latency_repeats",
        "stop_after_epochs",
        "steps_per_epoch",
    ):
        _check_positive_int(training, "training", key)
    for key in ("eval_orbit_samples", "latency_warmup", "num_workers"):
        _check_nonnegative_int(training, "training", key)
    data_fraction = float(training.get("data_fraction", 1.0))
    if data_fraction <= 0 or data_fraction > 1:
        raise ValueError("training.data_fraction must lie in (0, 1]")
    for key in ("lr", "weight_decay", "lambda_orbit", "lambda_aug", "lambda_tangent", "orbit_eta"):
        _check_nonnegative_float(training, "training", key)
    orbit_control = str(training.get("orbit_control", "physical")).lower()
    if orbit_control not in {
        "physical",
        "correct",
        "shuffle",
        "shuffled",
        "shuffle_output",
        "shuffled_output",
        "no_output",
        "no_output_transform",
        "input_only",
        "identity_output",
    }:
        raise ValueError(f"Unknown training.orbit_control={orbit_control!r}")

    symmetry = config.get("symmetry")
    if method in {
        "aug",
        "augmentation",
        "orbit",
        "orb",
        "aug_orbit",
        "orbit_aug",
        "aug_orbit_shuffle",
        "aug_orbit_shuffled",
        "aug_orbit_no_output",
        "aug_orbit_input_only",
        "semi_aug_orbit",
        "mixed_semi_aug_orbit",
        "split_semi_aug_orbit",
        "tangent",
        "tangent_prop",
        "tangent_propagation",
        "aug_tangent",
        "tangent_aug",
    } and not symmetry:
        raise ValueError(f"training.method={method!r} requires a symmetry section")
    if symmetry:
        if symmetry.get("enabled", True) is False:
            return
        for transform in _iter_transform_cfgs(symmetry):
            tname = str(transform.get("name", "translation1d")).lower()
            if tname not in _ALLOWED_TRANSFORMS:
                raise ValueError(f"Unknown symmetry transform={tname!r}")
            for key in ("max_shift", "max_boost", "length"):
                _check_positive_float(transform, "symmetry", key)
            for key in ("final_time",):
                _check_nonnegative_float(transform, "symmetry", key)
            for key in ("min_shift", "min_boost"):
                _check_nonnegative_float(transform, "symmetry", key)
        if "probabilities" in symmetry and "transforms" in symmetry:
            if len(symmetry["probabilities"]) != len(symmetry["transforms"]):
                raise ValueError("symmetry.probabilities must match symmetry.transforms length")
            if sum(float(x) for x in symmetry["probabilities"]) <= 0:
                raise ValueError("symmetry.probabilities must have positive sum")
