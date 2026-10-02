"""Construct symmetry actions from experiment sections or evaluation overrides."""

from __future__ import annotations

from typing import Any

from otno.definitions import TRANSFORM_ALIASES, canonical_name
from otno.symmetry.transforms import (
    BaseTransform,
    Burgers1DGalilean,
    CompositeTransform,
    D4Pseudoscalar2D,
    D4Scalar2D,
    MolecularRigidMotion,
    NavierStokes2DGalilean,
    NonPeriodicTranslation1D,
    Translation1D,
    Translation2D,
)


def _build_one(cfg: dict[str, Any]) -> BaseTransform:
    name = canonical_name(cfg.get("name", "translation1d"), TRANSFORM_ALIASES, "symmetry transform")
    if name == "translation1d":
        return Translation1D(max_shift=cfg.get("max_shift", 0.25), length=cfg.get("length", 1.0))
    if name == "nonperiodic_translation1d":
        return NonPeriodicTranslation1D(
            max_shift=cfg.get("max_shift", 0.1),
            length=cfg.get("length", 1.0),
            use_mask=cfg.get("use_mask", True),
            mask_margin=cfg.get("mask_margin", 0.0),
        )
    if name == "translation2d":
        return Translation2D(max_shift=cfg.get("max_shift", 0.25), length=cfg.get("length", 1.0))
    if name == "burgers1d_galilean":
        return Burgers1DGalilean(
            max_boost=cfg.get("max_boost", 0.5),
            final_time=cfg.get("final_time", 0.5),
            channel=cfg.get("channel", 0),
            length=cfg.get("length", 1.0),
        )
    if name == "navier_stokes2d_galilean":
        return NavierStokes2DGalilean(
            max_boost=cfg.get("max_boost", 0.5),
            final_time=cfg.get("final_time", 0.5),
            length=cfg.get("length", 1.0),
            boost_x_channel=cfg.get("boost_x_channel", 1),
            boost_y_channel=cfg.get("boost_y_channel", 2),
        )
    if name == "d4_scalar2d":
        return D4Scalar2D()
    if name == "d4_pseudoscalar2d":
        return D4Pseudoscalar2D()
    if name == "molecular_rigid_motion":
        return MolecularRigidMotion(
            max_angle=cfg.get("max_angle", 3.141592653589793),
            max_translation=cfg.get("max_translation", 1.0),
        )
    raise ValueError(f"Unknown transform: {name}")


def build_transform(config: dict[str, Any] | None) -> BaseTransform | None:
    """Build one action or a minibatch-sampled mixture of actions.

    Accept either the full experiment config or its symmetry section. Missing,
    empty, or explicitly disabled symmetry returns ``None``. A ``transforms``
    sequence samples one member for each minibatch; it does not compose actions.
    """
    if not config:
        return None
    cfg = config.get("symmetry", config)
    if cfg is None or cfg.get("enabled", True) is False:
        return None
    if "transforms" in cfg:
        transforms = [_build_one(item) for item in cfg["transforms"]]
        return CompositeTransform(transforms, cfg.get("probabilities"))
    return _build_one(cfg)
