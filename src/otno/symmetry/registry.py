from __future__ import annotations

from otno.symmetry.transforms import (
    BaseTransform,
    Burgers1DGalilean,
    CompositeTransform,
    D4Pseudoscalar2D,
    D4Scalar2D,
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
    if name in {"d4_scalar2d", "d4"}:
        return D4Scalar2D()
    if name in {"d4_pseudoscalar2d", "d4_vorticity2d"}:
        return D4Pseudoscalar2D()
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
