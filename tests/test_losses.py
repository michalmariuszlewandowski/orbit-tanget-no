import torch
from torch import nn

from otno.data.solvers2d import random_fourier_field_2d
from otno.symmetry.transforms import NavierStokes2DGalilean, Translation1D, TransformSample
from otno.training.losses import orbit_consistency_loss, relative_l2_loss, tangent_propagation_loss


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
    loss, _ = orbit_consistency_loss(model, x, transform, sample=sample, target_mode="shuffle_output")
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
    loss, _ = orbit_consistency_loss(model, x, transform, sample=sample, target_mode="no_output_transform")
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
