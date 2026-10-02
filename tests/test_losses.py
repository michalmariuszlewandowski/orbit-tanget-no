import torch
from torch import nn

from otno.data.solvers2d import random_fourier_field_2d
from otno.symmetry.transforms import (
    BaseTransform,
    NavierStokes2DGalilean,
    TransformSample,
    Translation1D,
)
from otno.training.losses import (
    mean_squared_per_sample,
    orbit_consistency_loss,
    relative_l2_loss,
    tangent_propagation_loss,
    trajectory_nmse_loss,
)


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


class PositionWeightedModel(nn.Module):
    def forward(self, x):
        weights = torch.linspace(0.5, 1.5, x.shape[1], device=x.device, dtype=x.dtype)
        return x * weights.view(1, -1, 1)


def test_relative_l2_loss_zero_on_match():
    x = torch.randn(4, 16, 1)
    assert relative_l2_loss(x, x).item() < 1e-8


def test_relative_l2_reduces_each_example_before_batch_average():
    target = torch.tensor([[[1.0], [0.0]], [[10.0], [0.0]]])
    pred = torch.tensor([[[2.0], [0.0]], [[11.0], [0.0]]])
    # Equal absolute errors but different target norms: average (1, 0.1).
    assert torch.allclose(relative_l2_loss(pred, target), torch.tensor(0.55))


def test_trajectory_nmse_reduces_space_then_time_then_batch():
    target = torch.tensor([[[1.0, 10.0], [1.0, 10.0]], [[2.0, 1.0], [2.0, 1.0]]])
    pred = torch.tensor([[[2.0, 11.0], [2.0, 11.0]], [[3.0, 3.0], [3.0, 3.0]]])
    # Per-time squared ratios are (1, .01) and (.25, 4).
    assert torch.allclose(trajectory_nmse_loss(pred, target), torch.tensor(1.315))


class MaskedOffsetTransform(BaseTransform):
    def apply_input(self, a, sample):
        return a + sample.params["offset"]

    def apply_output(self, u, sample):
        return u

    def output_mask(self, u, sample):
        return sample.params["mask"]


def test_masked_orbit_loss_reduces_valid_entries_before_normalizing():
    sample = TransformSample(
        params={
            "offset": torch.tensor(
                [
                    [[1.0, 3.0], [100.0, 100.0], [100.0, 100.0]],
                    [[2.0, 2.0], [4.0, 4.0], [100.0, 100.0]],
                ]
            ),
            "mask": torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        },
        epsilon=torch.tensor([1.0, 2.0]),
        name="masked_offset",
    )
    # The valid entries give mean squares 5 and 10, counting both channels.
    # Normalize those examples separately, then average: (5/2 + 10/5)/2.
    loss, _ = orbit_consistency_loss(
        IdentityModel(), torch.zeros(2, 3, 2), MaskedOffsetTransform(), sample=sample, eta=1.0
    )
    assert torch.allclose(loss, torch.tensor(2.25))
    empty = mean_squared_per_sample(torch.ones(2, 3, 2), mask=torch.zeros(2, 3))
    assert torch.equal(empty, torch.zeros(2))


def test_orbit_loss_backpropagates_through_both_predictions():
    model = nn.Sequential(nn.Linear(1, 2), nn.Tanh(), nn.Linear(2, 1))
    inputs = torch.tensor([[[0.0], [1.0], [2.0], [-1.0]]])
    transform = Translation1D()
    sample = TransformSample(
        {"shift": torch.tensor([0.125])}, torch.tensor([0.125]), transform.name
    )
    branches = []

    def retain_output(module, args, output):
        output.retain_grad()
        branches.append(output)

    handle = model.register_forward_hook(retain_output)
    try:
        base_pred = model(inputs)
        loss, _ = orbit_consistency_loss(
            model, inputs, transform, sample=sample, base_pred=base_pred
        )
        loss.backward()
    finally:
        handle.remove()
    assert len(branches) == 2  # Original prediction is reused.
    assert loss.item() > 0
    for branch in branches:
        assert branch.grad is not None
        assert branch.grad.abs().sum().item() > 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_tangent_loss_retains_parameter_gradients():
    # Squaring mixes modes, giving a nonzero discrete spectral tangent defect.
    class WeightedSquare(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.7))

        def forward(self, x):
            return self.weight * x.square()

    model = WeightedSquare()
    inputs = torch.tensor([[[0.0], [1.0], [2.0], [-1.0]]])
    sample = TransformSample({"shift": torch.tensor([0.1])}, torch.tensor([0.1]), "translation1d")
    loss, _ = tangent_propagation_loss(model, inputs, Translation1D(), sample=sample)
    loss.backward()
    assert loss.item() > 0
    assert model.weight.grad is not None
    assert model.weight.grad.abs().item() > 0


def test_orbit_loss_zero_for_identity_translation():
    model = IdentityModel()
    transform = Translation1D(max_shift=0.1)
    x = torch.randn(4, 32, 1)
    sample = TransformSample(
        params={"shift": torch.tensor([0.1, -0.1, 0.05, 0.0])},
        epsilon=torch.ones(4),
        name="translation1d",
    )
    loss, _ = orbit_consistency_loss(model, x, transform, sample=sample)
    assert loss.item() < 1e-10


def test_orbit_loss_shuffle_output_control_breaks_identity_translation():
    model = IdentityModel()
    transform = Translation1D(max_shift=0.1)
    x = torch.randn(4, 32, 1)
    sample = TransformSample(
        params={"shift": torch.tensor([0.1, -0.1, 0.05, 0.0])},
        epsilon=torch.ones(4),
        name="translation1d",
    )
    loss, _ = orbit_consistency_loss(
        model, x, transform, sample=sample, target_mode="shuffle_output"
    )
    assert loss.item() > 1e-4


def test_orbit_loss_no_output_transform_control_breaks_identity_translation():
    model = IdentityModel()
    transform = Translation1D(max_shift=0.1)
    x = torch.randn(4, 32, 1)
    sample = TransformSample(
        params={"shift": torch.tensor([0.1, -0.1, 0.05, 0.02])},
        epsilon=torch.ones(4),
        name="translation1d",
    )
    loss, _ = orbit_consistency_loss(
        model, x, transform, sample=sample, target_mode="no_output_transform"
    )
    assert loss.item() > 1e-4


def test_orbit_loss_normalization_only_divides_by_recorded_step_size():
    model = PositionWeightedModel()
    transform = Translation1D(max_shift=0.5)
    x = torch.arange(16, dtype=torch.float32).view(2, 8, 1)
    sample = TransformSample(
        params={"shift": torch.tensor([0.125, -0.25])},
        epsilon=torch.tensor([0.125, 0.25]),
        name="translation1d",
    )
    base_pred = model(x)
    transformed_input_pred = model(transform.apply_input(x, sample))
    transformed_base_pred = transform.apply_output(base_pred, sample)
    raw_per_sample = (transformed_input_pred - transformed_base_pred).flatten(1).pow(2).mean(dim=1)
    eta = 1e-6

    raw_loss, _ = orbit_consistency_loss(
        model,
        x,
        transform,
        sample=sample,
        base_pred=base_pred,
        normalize_by_epsilon=False,
        eta=eta,
    )
    normalized_loss, _ = orbit_consistency_loss(
        model,
        x,
        transform,
        sample=sample,
        base_pred=base_pred,
        normalize_by_epsilon=True,
        eta=eta,
    )

    assert torch.allclose(raw_loss, raw_per_sample.mean())
    assert torch.allclose(
        normalized_loss,
        (raw_per_sample / (sample.epsilon.pow(2) + eta)).mean(),
    )


def test_tangent_loss_zero_for_identity_translation():
    model = IdentityModel()
    transform = Translation1D(max_shift=0.1)
    x = torch.randn(4, 32, 1)
    sample = TransformSample(
        params={"shift": torch.tensor([0.1, -0.1, 0.05, 0.02])},
        epsilon=torch.tensor([0.1, 0.1, 0.05, 0.02]),
        name="translation1d",
    )
    loss, _ = tangent_propagation_loss(model, x, transform, sample=sample)
    assert loss.item() < 1e-10


def test_galilean_output_tangent_matches_finite_difference():
    transform = NavierStokes2DGalilean(max_boost=0.2, final_time=0.5)
    u = random_fourier_field_2d(2, 16, smoothness=2.0, amplitude=0.2, seed=3)[..., None]
    direction = torch.tensor([[0.04, -0.03], [-0.02, 0.01]])
    sample = TransformSample(
        params={"boost": direction},
        epsilon=torch.linalg.norm(direction, dim=-1),
        name="navier_stokes2d_galilean",
    )
    h = 1e-3
    small_sample = TransformSample(
        params={"boost": h * direction},
        epsilon=h * torch.linalg.norm(direction, dim=-1),
        name="navier_stokes2d_galilean",
    )
    finite_difference = (transform.apply_output(u, small_sample) - u) / h
    tangent = transform.output_tangent(u, sample)
    assert torch.allclose(finite_difference, tangent, atol=2e-3, rtol=2e-3)
