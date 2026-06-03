from __future__ import annotations

from .canonical import CanonicalFNO1d
from .fno import FNO1d, FNO2d


def _canonicalizers(value) -> list[str]:
    if value is None:
        return ["translation_first_mode"]
    if isinstance(value, str):
        return [value]
    return list(value)


def build_model(config: dict):
    model_cfg = config.get("model", config)
    name = str(model_cfg.get("name", "fno1d")).lower()
    if name in {"fno1d", "fno_1d"}:
        return FNO1d(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            width=int(model_cfg.get("width", 64)),
            modes=int(model_cfg.get("modes", 16)),
            depth=int(model_cfg.get("depth", 4)),
            add_grid=bool(model_cfg.get("add_grid", True)),
        )
    if name in {"canonical_fno1d", "canonical_fno_1d"}:
        return CanonicalFNO1d(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            width=int(model_cfg.get("width", 64)),
            modes=int(model_cfg.get("modes", 16)),
            depth=int(model_cfg.get("depth", 4)),
            add_grid=bool(model_cfg.get("add_grid", True)),
            canonicalizers=_canonicalizers(model_cfg.get("canonicalizers")),
            canonical_channel=int(model_cfg.get("canonical_channel", 0)),
            translation_mode=int(model_cfg.get("translation_mode", 1)),
            length=float(model_cfg.get("length", 1.0)),
            final_time=float(model_cfg.get("final_time", 0.5)),
        )
    if name in {"fno2d", "fno_2d"}:
        return FNO2d(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            width=int(model_cfg.get("width", 64)),
            modes1=int(model_cfg.get("modes1", model_cfg.get("modes", 12))),
            modes2=int(model_cfg.get("modes2", model_cfg.get("modes", 12))),
            depth=int(model_cfg.get("depth", 4)),
            add_grid=bool(model_cfg.get("add_grid", True)),
        )
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["FNO1d", "FNO2d", "CanonicalFNO1d", "build_model"]
