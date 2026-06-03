import torch
from torch import nn

from otno.data.solvers1d import random_fourier_field_1d

from otno.models.canonical import CanonicalFNO1d, estimate_first_mode_shift_1d
from otno.symmetry.transforms import periodic_shift_1d


class IdentityBase(nn.Module):
    def forward(self, x):
        return x


def test_first_mode_shift_transforms_affinely():
    n = 64
    x_grid = torch.arange(n, dtype=torch.float32) / n
    x = torch.cos(2 * torch.pi * (x_grid - 0.13))[None, :, None]
    shifted = periodic_shift_1d(x, torch.tensor([0.21]))
    s_x = estimate_first_mode_shift_1d(x)
    s_shifted = estimate_first_mode_shift_1d(shifted)
    diff = ((s_shifted - (s_x - 0.21) + 0.5) % 1.0) - 0.5
    assert diff.abs().item() < 1e-5


def test_translation_canonical_wrapper_identity_base_is_identity():
    model = CanonicalFNO1d(width=4, modes=2, depth=1, canonicalizers=["translation_first_mode"])
    model.base = IdentityBase()
    x = random_fourier_field_1d(3, 32, modes=5, amplitude=1.0, seed=44)[..., None]
    y = model(x)
    assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def test_canonical_model_shape():
    model = CanonicalFNO1d(width=8, modes=4, depth=2, canonicalizers=["galilean_mean", "translation_first_mode"])
    x = torch.randn(2, 32, 1)
    y = model(x)
    assert y.shape == x.shape
