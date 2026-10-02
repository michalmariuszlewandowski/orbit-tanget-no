"""Regression checks for config contracts that can change scientific behavior."""

import copy

import pytest
import torch

from otno.config import load_config, validate_config
from otno.models import build_model
from otno.models.canonical import CanonicalFNO1d
from otno.symmetry.registry import build_transform
from otno.training.trainer import train_from_config


@pytest.mark.parametrize("canonicalizers", ["translation_first_mdoe", ["galilean_mean", "typo"]])
def test_unknown_canonicalizer_cannot_silently_select_plain_backbone(canonicalizers):
    model_config = {
        "name": "canonical_fno1d",
        "width": 4,
        "depth": 1,
        "modes": 2,
        "canonicalizers": canonicalizers,
    }
    with pytest.raises(ValueError, match="canonicalizers"):
        validate_config({"model": model_config}, require_dataset=False)
    with pytest.raises(ValueError, match="canonicalizers"):
        build_model(model_config)
    with pytest.raises(ValueError, match="canonicalizers"):
        CanonicalFNO1d(width=4, depth=1, modes=2, canonicalizers=canonicalizers)


def test_explicit_empty_canonicalizers_still_disable_frame_estimators():
    model = CanonicalFNO1d(width=4, depth=1, modes=2, canonicalizers=[])
    inputs = torch.randn(2, 8, 1)
    assert torch.equal(model(inputs), model.base(inputs))


@pytest.mark.parametrize("channels", [[4.0, 8.0], ["4", "8"]])
def test_validated_deeponet_channel_sizes_are_constructible(channels):
    config = {
        "model": {
            "name": "deeponet_2d",
            "branch_channels": channels,
            "grid_size": 8,
            "branch_fc_hidden": 8,
            "trunk_hidden": 8,
            "trunk_depth": 1,
            "latent_dim": 4,
            "coordinate_modes": 2,
        }
    }
    original = copy.deepcopy(config)
    validate_config(config, require_dataset=False)
    model = build_model(config)
    assert model(torch.randn(2, 8, 8, 1)).shape == (2, 8, 8, 1)
    assert model.branch_channels == (4, 8)
    assert config == original


@pytest.mark.parametrize("key", ["max_angle", "max_translation"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_invalid_molecular_action_bounds_are_rejected(key, value):
    config = {"symmetry": {"name": "se3_molecular", key: value}}
    with pytest.raises(ValueError, match=f"symmetry.{key}"):
        validate_config(config, require_dataset=False)


def test_zero_molecular_action_bounds_are_valid_identity_actions():
    config = {"symmetry": {"name": "molecular_rigid_motion", "max_angle": 0, "max_translation": 0}}
    validate_config(config, require_dataset=False)
    transform = build_transform(config)
    inputs = torch.randn(2, 4, 4)
    sample = transform.sample(2, torch.device("cpu"))
    assert torch.allclose(transform.apply_input(inputs, sample), inputs)


def test_zero_based_canonical_boost_indices_validate_and_run():
    config = {
        "model": {
            "name": "pace_style_fno2d",
            "width": 4,
            "depth": 1,
            "modes": 2,
            "boost_x_channel": 0,
            "boost_y_channel": 1,
        }
    }
    original = copy.deepcopy(config)
    validate_config(config, require_dataset=False)
    model = build_model(config)
    assert model(torch.randn(2, 8, 8, 3)).shape == (2, 8, 8, 1)
    assert config == original


@pytest.mark.parametrize(
    "symmetry",
    [
        {"name": "d4_scalar_2d"},
        {"name": "d4_vorticity2d"},
        {"name": "dirichlet_translation1d"},
        {"name": "rigid_motion3d"},
        {"transforms": [{"name": "translation_1d"}, {"name": "d4_scalar2d"}]},
    ],
)
@pytest.mark.parametrize("method", ["tangent_prop", "tangent_aug"])
def test_unsupported_tangent_actions_fail_before_dataset_preparation(monkeypatch, symmetry, method):
    def unexpected_preparation(_):
        pytest.fail("Unsupported tangent objectives must fail before dataset preparation")

    monkeypatch.setattr("otno.training.trainer._prepare_dataset", unexpected_preparation)
    config = {
        "dataset": {"kind": "advection1d", "path": "unused.pt"},
        "training": {"method": method},
        "symmetry": symmetry,
    }
    with pytest.raises(ValueError, match="requires infinitesimal actions"):
        train_from_config(config)


def test_supported_tangent_mixture_preserves_aliases_and_config():
    config = {
        "training": {"method": "tangent_propagation"},
        "symmetry": {
            "transforms": [{"name": "translation_1d"}, {"name": "galilean1d"}],
            "probabilities": [0.4, 0.6],
        },
    }
    original = copy.deepcopy(config)
    validate_config(config, require_dataset=False)
    transform = build_transform(config)
    assert [action.name for action in transform.transforms] == [
        "translation1d",
        "burgers1d_galilean",
    ]
    assert config == original


@pytest.mark.parametrize("section", ["dataset", "model", "training", "symmetry"])
def test_non_mapping_sections_report_configuration_errors(section):
    config = {"dataset": {"kind": "advection1d", "path": "unused.pt"}, section: "typo"}
    with pytest.raises(ValueError, match=f"{section} must be a mapping"):
        validate_config(config)


def test_non_mapping_yaml_is_rejected_at_load_time(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("- dataset\n- model\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_config(path)
