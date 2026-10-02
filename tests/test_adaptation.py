import copy
import json

import pytest
import torch

from otno.data.generators import generate_dataset_from_config
from otno.models import build_model
from otno.symmetry.transforms import Translation1D
from otno.training.adaptation import (
    _adaptation_epochs,
    _build_adaptation_state,
    _parameter_l2_to_initial,
    _prediction_preservation_loss,
    _random_orbit_consistency_loss,
    adapt_from_config,
)
from otno.utils import set_seed


def test_parameter_l2_to_initial_is_real_for_complex_parameters():
    param = torch.nn.Parameter(torch.tensor([1.0 + 2.0j], dtype=torch.complex64))
    initial = [torch.zeros_like(param)]

    reg = _parameter_l2_to_initial([param], initial)

    assert not torch.is_complex(reg)
    assert torch.isclose(reg, torch.tensor(5.0))
    reg.backward()
    assert param.grad is not None


def test_prediction_preservation_loss_is_zero_for_matching_predictions():
    pred = torch.randn(4, 8, 1)
    loss = _prediction_preservation_loss(pred, pred.detach())

    assert torch.isclose(loss, torch.tensor(0.0))


def test_prediction_preservation_loss_is_relative_squared_error():
    pred = torch.tensor([[[2.0], [0.0]]])
    reference = torch.tensor([[[1.0], [1.0]]])

    loss = _prediction_preservation_loss(pred, reference)

    assert torch.isclose(loss, torch.tensor(1.0))


def test_random_orbit_log_reports_absolute_defect():
    inputs = torch.tensor([1.0, 3.0]).view(2, 1, 1).expand(2, 4, 1)
    model = torch.nn.Identity()
    transform = Translation1D(max_shift=0.1)

    _, original = _random_orbit_consistency_loss(
        model, inputs, transform, normalize_by_epsilon=False
    )
    _, scaled = _random_orbit_consistency_loss(
        model, 3 * inputs, transform, normalize_by_epsilon=False
    )

    # Shuffled constant fields differ by 2 at four positions, giving L2 norm 4.
    assert original["orbit_defect_absolute"] == 4.0
    assert scaled["orbit_defect_absolute"] == 12.0
    assert "orbit_defect_relative" not in original


def test_adaptation_respects_overwrite_false(tmp_path):
    run_dir = tmp_path / "adaptation"
    run_dir.mkdir()
    saved_config = run_dir / "config.yaml"
    saved_config.write_text("existing experiment\n")
    config = {
        "adaptation": {"checkpoint": str(tmp_path / "missing.pt")},
        "runtime": {"run_dir": str(run_dir), "overwrite": False, "device": "cpu"},
    }

    with pytest.raises(FileExistsError):
        adapt_from_config(config)

    assert saved_config.read_text() == "existing experiment\n"


def test_invalid_source_preserves_previous_adaptation_outputs(tmp_path):
    run_dir = tmp_path / "adaptation"
    run_dir.mkdir()
    saved_config = run_dir / "config.yaml"
    saved_config.write_text("existing experiment\n")
    config = {
        "adaptation": {"checkpoint": str(tmp_path / "missing.pt")},
        "runtime": {"run_dir": str(run_dir), "overwrite": True, "device": "cpu"},
    }

    with pytest.raises(FileNotFoundError):
        adapt_from_config(config)

    assert saved_config.read_text() == "existing experiment\n"


def test_adaptation_rejects_unsupported_resume_before_writing(tmp_path):
    run_dir = tmp_path / "adaptation"
    config = {"runtime": {"run_dir": str(run_dir), "resume": True, "device": "cpu"}}

    with pytest.raises(ValueError, match="resume"):
        adapt_from_config(config)

    assert not run_dir.exists()


def test_fresh_adaptation_overwrite_reproduces_weights_without_duplicate_logs(tmp_path):
    base_config = {
        "dataset": {
            "kind": "advection1d",
            "path": str(tmp_path / "data.pt"),
            "n": 8,
            "modes": 2,
            "num_train": 4,
            "num_val": 2,
            "num_test": 2,
            "final_time": 0.1,
            "seed": 3,
        },
        "model": {"name": "fno1d", "width": 2, "modes": 2, "depth": 1},
        "symmetry": {"name": "translation1d", "max_shift": 0.1},
    }
    generate_dataset_from_config(base_config)
    checkpoint = tmp_path / "base.pt"
    torch.save({"model": build_model(base_config).state_dict(), "config": base_config}, checkpoint)
    run_dir = tmp_path / "adaptation"
    config = {
        "seed": 17,
        "adaptation": {
            "checkpoint": str(checkpoint),
            "epochs": 1,
            "batch_size": 2,
            "eval_orbit_samples": 1,
            "latency_repeats": 1,
            "latency_warmup": 0,
        },
        "runtime": {"run_dir": str(run_dir), "overwrite": True, "device": "cpu"},
    }
    expected = adapt_from_config(config)
    expected_model = torch.load(run_dir / "adapted.pt", weights_only=False)["model"]
    actual = adapt_from_config(config)
    actual_model = torch.load(run_dir / "adapted.pt", weights_only=False)["model"]

    assert actual["after"]["relative_l2"] == expected["after"]["relative_l2"]
    for key, value in expected_model.items():
        assert torch.equal(value, actual_model[key])
    records = [
        json.loads(line) for line in (run_dir / "adapt_metrics.jsonl").read_text().splitlines()
    ]
    assert [record["epoch"] for record in records] == [1]


@pytest.mark.parametrize("loss_mode", ["orbit", "random_orbit", "preserve"])
@pytest.mark.parametrize("trainable", ["projector", "last_block", "all"])
def test_adaptation_optimization_ignores_labels_and_preserves_frozen_source(loss_mode, trainable):
    set_seed(9)
    base = build_model({"model": {"name": "fno1d", "width": 2, "modes": 2, "depth": 2}})
    source = copy.deepcopy(base).eval().requires_grad_(False)
    source_weights = {key: value.clone() for key, value in source.state_dict().items()}
    inputs = torch.randn(2, 8, 1)
    cfg = {
        "trainable": trainable,
        "loss_mode": loss_mode,
        "epochs": 2,
        "gamma_prediction_preservation": 0.3,
        "beta_l2_initial": 0.01,
        "grad_clip": 0.5,
    }
    models, records = [], []
    # Both an input-only batch and a batch with unusable labels must optimize identically.
    for batch in ({"a": inputs}, {"a": inputs, "u": torch.full_like(inputs, float("nan"))}):
        set_seed(11)
        model = copy.deepcopy(base)
        state = _build_adaptation_state(model, cfg)
        selected = {name for name, param in model.named_parameters() if param.requires_grad}
        if trainable == "projector":
            assert selected == {"fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"}
        elif trainable == "last_block":
            assert "spectral.1.weights" in selected
            assert "pointwise.1.weight" in selected
            assert "spectral.0.weights" not in selected
        else:
            assert selected == set(dict(model.named_parameters()))
        records.append(
            list(
                _adaptation_epochs(
                    model,
                    source,
                    [batch],
                    device=torch.device("cpu"),
                    transform=Translation1D(max_shift=0.1),
                    target_input_transform=Translation1D(max_shift=0.2),
                    state=state,
                    adaptation_cfg=cfg,
                )
            )
        )
        models.append(model.state_dict())
        for name, param in model.named_parameters():
            if not param.requires_grad:
                assert param.grad is None
                assert torch.equal(param, base.state_dict()[name])
    assert records[0] == records[1]
    assert all(torch.equal(models[0][key], models[1][key]) for key in models[0])
    assert all(param.grad is None for param in source.parameters())
    assert all(
        torch.equal(value, source_weights[key]) for key, value in source.state_dict().items()
    )
