import pytest
import torch
from torch.utils.data import DataLoader

from otno.data.datasets import TensorDictDataset
from otno.symmetry.transforms import Translation1D
from otno.training.objectives import build_training_objective, compute_batch_loss
from otno.training.trainer import _train_epoch


@pytest.mark.parametrize("invalid_target", [float("nan"), float("inf")])
def test_nonfinite_loss_stops_before_parameter_or_optimizer_updates(invalid_target):
    model = torch.nn.Linear(1, 1)
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer = torch.optim.AdamW(model.parameters())
    inputs = torch.ones(2, 8, 1)
    dataset = TensorDictDataset(inputs, torch.full_like(inputs, invalid_target))
    loaders = {"train": DataLoader(dataset, batch_size=2)}

    with pytest.raises(FloatingPointError, match="epoch 1, batch 1"):
        _train_epoch(
            model,
            optimizer,
            loaders,
            build_training_objective({}),
            device=torch.device("cpu"),
            transform=None,
            epoch=1,
            training={"epochs": 1},
        )

    assert not optimizer.state
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, initial[name])
        assert parameter.grad is None


@pytest.mark.parametrize(
    ("method", "alias"),
    [
        ("aug_orbit_shuffle", "aug_orbit_shuffled"),
        ("aug_orbit_no_output", "aug_orbit_input_only"),
        ("aug_tangent", "tangent_aug"),
    ],
)
def test_method_aliases_preserve_loss_and_sampler_stream(method, alias):
    model = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.GELU(), torch.nn.Linear(2, 1))
    inputs, targets = torch.randn(3, 16, 1), torch.randn(3, 16, 1)
    transform = Translation1D(max_shift=0.1)
    settings = {"lambda_aug": 0.5, "lambda_orbit": 0.1, "lambda_tangent": 0.2}
    rng_state = torch.get_rng_state()
    loss, logs = compute_batch_loss(
        model,
        inputs,
        targets,
        build_training_objective({**settings, "method": method}),
        transform=transform,
    )
    final_rng_state = torch.get_rng_state()

    torch.set_rng_state(rng_state)
    alias_loss, alias_logs = compute_batch_loss(
        model,
        inputs,
        targets,
        build_training_objective({**settings, "method": alias}),
        transform=transform,
    )

    assert torch.equal(loss, alias_loss)
    assert logs == alias_logs
    assert torch.equal(torch.get_rng_state(), final_rng_state)
