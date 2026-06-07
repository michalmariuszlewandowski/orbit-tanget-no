import torch
from torch import nn

from otno.symmetry.transforms import Translation1D, TransformSample
from otno.training.losses import orbit_consistency_loss, relative_l2_loss


class IdentityModel(nn.Module):
    def forward(self, x):
        return x


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
