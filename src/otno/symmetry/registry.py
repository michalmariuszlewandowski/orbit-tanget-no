from __future__ import annotations

from otno.symmetry.transforms import (
    BaseTransform,
    Burgers1DGalilean,
    CompositeTransform,
    D4Pseudoscalar2D,
    D4Scalar2D,
    MolecularRigidMotion,
    NavierStokes2DGalilean,
    Translation1D,
    Translation2D,
)


def _build_one(cfg: dict) -> BaseTransform:
    name = str(cfg.get("name", "translation1d")).lower()
    if name in {"translation1d", "translation_1d"}:
        return Translation1D(max_shift=cfg.get("max_shift", 0.25), length=cfg.get("length", 1.0))
    if name in {"translation2d", "translation_2d"}:
        return Translation2D(max_shift=cfg.get("max_shift", 0.25), length=cfg.get("length", 1.0))
    if name in {"burgers1d_galilean", "galilean1d", "burgers_galilean"}:
        return Burgers1DGalilean(
            max_boost=cfg.get("max_boost", 0.5),
            final_time=cfg.get("final_time", 0.5),
            channel=cfg.get("channel", 0),
            length=cfg.get("length", 1.0),
        )
    if name in {"navier_stokes2d_galilean", "ns2d_galilean", "galilean2d", "vorticity2d_galilean"}:
        return NavierStokes2DGalilean(
            max_boost=cfg.get("max_boost", 0.5),
            final_time=cfg.get("final_time", 0.5),
            length=cfg.get("length", 1.0),
            boost_x_channel=cfg.get("boost_x_channel", 1),
            boost_y_channel=cfg.get("boost_y_channel", 2),
        )
    if name in {"d4_scalar2d", "d4"}:
        return D4Scalar2D()
    if name in {"d4_pseudoscalar2d", "d4_vorticity2d"}:
        return D4Pseudoscalar2D()
    if name in {"molecular_rigid_motion", "molecule_rigid_motion", "se3_molecular", "rmd17_rigid_motion"}:
        return MolecularRigidMotion(
            max_angle=cfg.get("max_angle", 0.5),
            max_translation=cfg.get("max_translation", 0.5),
            coord_start=cfg.get("coord_start", 0),
            force_start=cfg.get("force_start", 0),
            rotate=cfg.get("rotate", True),
            translate=cfg.get("translate", True),
        )
    raise ValueError(f"Unknown transform: {name}")


def build_transform(config: dict | None) -> BaseTransform | None:
    if not config:
        return None
    cfg = config.get("symmetry", config)
    if cfg is None or cfg.get("enabled", True) is False:
        return None
    if "transforms" in cfg:
        transforms = [_build_one(item) for item in cfg["transforms"]]
        return CompositeTransform(transforms, cfg.get("probabilities"))
    return _build_one(cfg)
