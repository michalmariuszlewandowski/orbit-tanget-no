"""Accepted experiment names shared by validation and construction.

Aliases are resolved locally; the user's configuration is never rewritten. This
keeps saved configs, provenance hashes, and historical experiment names intact.
"""

from collections.abc import Mapping, Sequence
from typing import Any


def _aliases(groups: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    return {
        name: canonical
        for canonical, alternatives in groups.items()
        for name in (canonical, *alternatives)
    }


DATASET_ALIASES = _aliases(
    {
        "advection1d": ("1d_advection",),
        "burgers1d": ("1d_burgers",),
        "heat1d_dirichlet": ("dirichlet_heat1d",),
        "navier_stokes_vorticity2d": ("ns2d", "2d_navier_stokes"),
        "navier_stokes_vorticity2d_boosted": (
            "boosted_navier_stokes_vorticity2d",
            "ns2d_boosted",
            "boosted_ns2d",
        ),
        "rmd17_force": ("rmd17",),
    }
)

MODEL_ALIASES = _aliases(
    {
        "fno1d": ("fno_1d",),
        "fno2d": ("fno_2d",),
        "deeponet2d": ("deeponet_2d",),
        "cno2d": ("cno_2d",),
        "canonical_fno1d": ("canonical_fno_1d",),
        "canonical_fno2d": (
            "canonical_fno_2d",
            "observable_galilean_canonical_fno2d",
            "galilean_canonical_fno2d",
            "pace_style_fno2d",
        ),
        "d4_gfno2d": ("d4_gfno_2d", "gfno2d", "g_fno2d", "g-fno2d"),
        "molecule_mlp": ("molecular_mlp",),
    }
)

METHOD_ALIASES = _aliases(
    {
        "baseline": (),
        "aug": ("augmentation",),
        "orbit": ("orb",),
        "aug_orbit": ("orbit_aug",),
        "aug_orbit_shuffle": ("aug_orbit_shuffled",),
        "aug_orbit_no_output": ("aug_orbit_input_only",),
        "semi_aug_orbit": (),
        "tangent": ("tangent_prop", "tangent_propagation"),
        "aug_tangent": ("tangent_aug",),
    }
)

TRANSFORM_ALIASES = _aliases(
    {
        "translation1d": ("translation_1d",),
        "nonperiodic_translation1d": (
            "nonperiodic_translation_1d",
            "dirichlet_translation1d",
        ),
        "translation2d": ("translation_2d",),
        "burgers1d_galilean": ("galilean1d", "burgers_galilean"),
        "navier_stokes2d_galilean": (
            "ns2d_galilean",
            "galilean2d",
            "vorticity2d_galilean",
        ),
        "d4_scalar2d": ("d4_scalar_2d",),
        "d4_pseudoscalar2d": (
            "d4_pseudoscalar_2d",
            "d4_vorticity2d",
            "d4_vorticity_2d",
        ),
        "molecular_rigid_motion": ("rigid_motion3d", "se3_molecular"),
    }
)

# These transforms implement both infinitesimal actions used by the JVP loss.
TANGENT_TRANSFORMS = frozenset(
    {"translation1d", "translation2d", "burgers1d_galilean", "navier_stokes2d_galilean"}
)
TANGENT_METHODS = frozenset({"tangent", "aug_tangent"})
CANONICALIZERS_1D = frozenset({"translation_first_mode", "galilean_mean"})


def canonical_name(value: Any, aliases: Mapping[str, str], field: str) -> str:
    """Resolve a case-insensitive name, reporting the config field on failure."""
    name = str(value).lower()
    try:
        return aliases[name]
    except KeyError:
        raise ValueError(f"Unknown {field}={name!r}") from None


def canonicalizer_names(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Validate 1D frame estimators; an explicit empty sequence disables them."""
    if value is None:
        names = ("translation_first_mode",)
    elif isinstance(value, str):
        names = (value,)
    elif isinstance(value, Sequence):
        names = tuple(value)
    else:
        raise ValueError("model.canonicalizers must be a name or a sequence of names")
    for name in names:
        if not isinstance(name, str) or name not in CANONICALIZERS_1D:
            raise ValueError(f"Unknown model.canonicalizers entry={name!r}")
    return names
