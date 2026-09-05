from __future__ import annotations

from .canonical import CanonicalFNO1d, ObservableGalileanCanonicalFNO2d
from .deeponet import DeepONet2d
from .fno import FNO1d, FNO2d
from .gfno import D4GFNO2d
from .molecular import MoleculeMLP


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
    if name in {
        "canonical_fno2d",
        "canonical_fno_2d",
        "observable_galilean_canonical_fno2d",
        "galilean_canonical_fno2d",
        "pace_style_fno2d",
    }:
        return ObservableGalileanCanonicalFNO2d(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            width=int(model_cfg.get("width", 64)),
            modes1=int(model_cfg.get("modes1", model_cfg.get("modes", 12))),
            modes2=int(model_cfg.get("modes2", model_cfg.get("modes", 12))),
            depth=int(model_cfg.get("depth", 4)),
            add_grid=bool(model_cfg.get("add_grid", True)),
            boost_x_channel=int(model_cfg.get("boost_x_channel", 1)),
            boost_y_channel=int(model_cfg.get("boost_y_channel", 2)),
            length=float(model_cfg.get("length", 1.0)),
            final_time=float(model_cfg.get("final_time", 0.5)),
            boost_reduction=str(model_cfg.get("boost_reduction", "mean")),
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
    if name in {"deeponet2d", "deeponet_2d"}:
        return DeepONet2d(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            grid_height=int(model_cfg.get("grid_height", model_cfg.get("grid_size", 64))),
            grid_width=int(model_cfg.get("grid_width", model_cfg.get("grid_size", 64))),
            branch_channels=tuple(model_cfg.get("branch_channels", (32, 64, 128, 256))),
            branch_fc_hidden=int(model_cfg.get("branch_fc_hidden", 480)),
            trunk_hidden=int(model_cfg.get("trunk_hidden", 256)),
            trunk_depth=int(model_cfg.get("trunk_depth", 3)),
            latent_dim=int(model_cfg.get("latent_dim", 256)),
            coordinate_modes=int(model_cfg.get("coordinate_modes", 12)),
        )
    if name in {"d4_gfno2d", "d4_gfno_2d", "gfno2d", "g_fno2d", "g-fno2d"}:
        return D4GFNO2d(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            width=int(model_cfg.get("width", 12)),
            modes1=int(model_cfg.get("modes1", model_cfg.get("modes", 8))),
            modes2=int(model_cfg.get("modes2", model_cfg.get("modes", 8))),
            depth=int(model_cfg.get("depth", 4)),
            add_grid=bool(model_cfg.get("add_grid", False)),
            input_pseudoscalar=bool(model_cfg.get("input_pseudoscalar", True)),
            output_pseudoscalar=bool(model_cfg.get("output_pseudoscalar", True)),
        )
    if name in {"molecule_mlp", "molecular_mlp"}:
        return MoleculeMLP(
            n_atoms=int(model_cfg.get("n_atoms", model_cfg.get("atoms", 9))),
            in_channels=int(model_cfg.get("in_channels", 4)),
            out_channels=int(model_cfg.get("out_channels", 3)),
            hidden=int(model_cfg.get("hidden", 128)),
            depth=int(model_cfg.get("depth", 4)),
        )
    raise ValueError(f"Unknown model name: {name}")


__all__ = [
    "FNO1d",
    "FNO2d",
    "DeepONet2d",
    "D4GFNO2d",
    "MoleculeMLP",
    "CanonicalFNO1d",
    "ObservableGalileanCanonicalFNO2d",
    "build_model",
]
