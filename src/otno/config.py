"""YAML loading, matrix overrides, and validation of experiment settings.

Validation checks each domain without rewriting values or applying defaults to
the saved config. Model and transform factories remain responsible for their
construction defaults.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml

from otno.definitions import (
    DATASET_ALIASES,
    METHOD_ALIASES,
    MODEL_ALIASES,
    TANGENT_METHODS,
    TANGENT_TRANSFORMS,
    TRANSFORM_ALIASES,
    canonical_name,
    canonicalizer_names,
)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping, treating an empty file as an empty configuration."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save values in their existing order, creating the destination directory."""
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
    """Read a nested key, returning default if any part of the path is absent."""
    cursor: Any = config
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def set_by_path(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Set a nested key in place, creating missing mapping sections."""
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
    """Expand a mapping of dotted CLI/matrix keys into nested sections."""
    nested: dict[str, Any] = {}
    for key, value in overrides.items():
        set_by_path(nested, key, value)
    return nested


def apply_dotted_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return an independent configuration with dotted overrides applied."""
    return recursive_update(base, dotted_overrides_to_nested(overrides))


def format_config_values(value: Any, context: dict[str, Any]) -> Any:
    """Substitute matrix fields such as {seed} in nested configuration values."""
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [format_config_values(item, context) for item in value]
    if isinstance(value, dict):
        return {key: format_config_values(item, context) for key, item in value.items()}
    return value


def _check_number(
    section: dict[str, Any],
    section_name: str,
    key: str,
    *,
    integer: bool = False,
    positive: bool = False,
) -> None:
    if key not in section:
        return
    value = section[key]
    try:
        number = float(value)
        valid = not isinstance(value, bool) and math.isfinite(number)
        valid = valid and (number > 0 if positive else number >= 0)
        if integer:
            valid = valid and int(value) == number
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        bound = "positive" if positive else "non-negative"
        kind = "integer" if integer else "finite number"
        raise ValueError(f"{section_name}.{key} must be a {bound} {kind}")


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name, {})
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    return section


def _validate_dataset(dataset: dict[str, Any]) -> None:
    """Check sample counts and solver parameters before data generation."""
    for key in ("kind", "path"):
        if key not in dataset:
            raise ValueError(f"dataset.{key} is required")
    kind = canonical_name(dataset["kind"], DATASET_ALIASES, "dataset.kind")
    for key in ("n", "num_train", "num_val", "num_test", "solver_batch_size"):
        _check_number(dataset, "dataset", key, integer=True, positive=True)
    for key in ("final_time", "viscosity", "diffusivity"):
        _check_number(dataset, "dataset", key)
    _check_number(dataset, "dataset", "charge_scale", positive=True)
    time_stepped = {"burgers1d", "navier_stokes_vorticity2d", "navier_stokes_vorticity2d_boosted"}
    _check_number(dataset, "dataset", "dt", positive=kind in time_stepped)
    if kind == "navier_stokes_vorticity2d_boosted":
        _check_number(dataset, "dataset", "max_boost")


def _validate_model(model: dict[str, Any]) -> None:
    """Check architecture sizes separately from zero-based channel indices."""
    name = canonical_name(model.get("name", "fno1d"), MODEL_ALIASES, "model.name")
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
        _check_number(model, "model", key, integer=True, positive=True)
    for key in ("canonical_channel", "boost_x_channel", "boost_y_channel"):
        _check_number(model, "model", key, integer=True)
    _check_number(model, "model", "negative_slope")
    if "branch_channels" in model:
        channels = model["branch_channels"]
        if not isinstance(channels, (list, tuple)) or not channels:
            raise ValueError("model.branch_channels must be a non-empty sequence")
        for channel in channels:
            _check_number(
                {"branch_channels": channel},
                "model",
                "branch_channels",
                integer=True,
                positive=True,
            )

    if name in {"canonical_fno1d", "canonical_fno2d"}:
        _check_number(model, "model", "length", positive=True)
        _check_number(model, "model", "final_time")
    if name == "canonical_fno1d":
        canonicalizer_names(model.get("canonicalizers"))
    if name == "canonical_fno2d" and model.get("boost_reduction", "mean") not in {"mean", "corner"}:
        raise ValueError("model.boost_reduction must be 'mean' or 'corner'")


def _validate_training(training: dict[str, Any]) -> str:
    """Check optimization settings and return the method's canonical name."""
    method = canonical_name(training.get("method", "baseline"), METHOD_ALIASES, "training.method")
    for key in (
        "epochs",
        "batch_size",
        "eval_every",
        "latency_repeats",
        "stop_after_epochs",
        "steps_per_epoch",
        "orbit_steps_per_epoch",
        "orbit_batch_size",
    ):
        _check_number(training, "training", key, integer=True, positive=True)
    for key in ("eval_orbit_samples", "latency_warmup", "num_workers"):
        _check_number(training, "training", key, integer=True)
    for key in ("data_fraction", "orbit_data_fraction"):
        _check_number(training, "training", key, positive=True)
        if float(training.get(key, 1.0)) > 1:
            raise ValueError(f"training.{key} must lie in (0, 1]")
    for key in ("lr", "weight_decay", "lambda_orbit", "lambda_aug", "lambda_tangent", "orbit_eta"):
        _check_number(training, "training", key)
    if training.get("grad_clip") is not None:
        _check_number(training, "training", "grad_clip")
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
    return method


def _iter_transform_cfgs(symmetry: dict[str, Any]) -> list[dict[str, Any]]:
    if "transforms" not in symmetry:
        return [symmetry]
    transforms = symmetry["transforms"]
    if not isinstance(transforms, (list, tuple)) or not transforms:
        raise ValueError("symmetry.transforms must be a non-empty sequence of mappings")
    if any(not isinstance(transform, dict) for transform in transforms):
        raise ValueError("symmetry.transforms entries must be mappings")
    return list(transforms)


def _validate_probabilities(symmetry: dict[str, Any], transform_count: int) -> None:
    values = symmetry["probabilities"]
    message = "symmetry.probabilities must be finite, non-negative, and have positive sum"
    if not isinstance(values, (list, tuple)) or len(values) != transform_count:
        raise ValueError("symmetry.probabilities must match symmetry.transforms length")
    try:
        probabilities = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
    if (
        any(not math.isfinite(value) or value < 0 for value in probabilities)
        or not 0 < sum(probabilities) < math.inf
    ):
        raise ValueError(message)


def _validate_symmetry(symmetry: dict[str, Any] | None, method: str) -> None:
    """Check finite actions and reject unsupported JVP objectives before a run."""
    if symmetry is not None and not isinstance(symmetry, dict):
        raise ValueError("symmetry must be a mapping")
    if method != "baseline" and (not symmetry or symmetry.get("enabled", True) is False):
        raise ValueError(f"training.method={method!r} requires an enabled symmetry section")
    if not symmetry or symmetry.get("enabled", True) is False:
        return
    transforms = _iter_transform_cfgs(symmetry)
    for transform in transforms:
        name = canonical_name(
            transform.get("name", "translation1d"), TRANSFORM_ALIASES, "symmetry transform"
        )
        for key in ("max_shift", "max_boost", "length"):
            _check_number(transform, "symmetry", key, positive=True)
        for key in (
            "final_time",
            "min_shift",
            "min_boost",
            "max_angle",
            "max_translation",
            "mask_margin",
        ):
            _check_number(transform, "symmetry", key)
        for key in ("channel", "boost_x_channel", "boost_y_channel"):
            _check_number(transform, "symmetry", key, integer=True)
        if method in TANGENT_METHODS and name not in TANGENT_TRANSFORMS:
            raise ValueError(
                f"training.method={method!r} requires infinitesimal actions; "
                f"symmetry transform={name!r} does not implement them"
            )
    if "probabilities" in symmetry and "transforms" in symmetry:
        _validate_probabilities(symmetry, len(transforms))


def validate_config(config: dict[str, Any], *, require_dataset: bool = True) -> None:
    """Validate experiment settings without changing config values or defaults.

    Set ``require_dataset=False`` when checking a model/training/symmetry fragment.
    Shape compatibility that depends on actual data is checked by the models and
    transforms when they consume tensors.
    """
    if not isinstance(config, dict):
        raise ValueError("Config must be a mapping")
    if require_dataset:
        if "dataset" not in config:
            raise ValueError("Config requires a dataset section")
        _validate_dataset(_section(config, "dataset"))
    _validate_model(_section(config, "model"))
    method = _validate_training(_section(config, "training"))
    _validate_symmetry(config.get("symmetry"), method)
