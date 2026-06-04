import torch

from otno.data.solvers1d import random_fourier_field_1d
from otno.data.solvers2d import random_fourier_field_2d
from otno.symmetry.transforms import (
    Burgers1DGalilean,
    D4Pseudoscalar2D,
    D4Scalar2D,
    NavierStokes2DGalilean,
    Translation1D,
    TransformSample,
    periodic_shift_1d,
    periodic_shift_2d,
)


def test_periodic_shift_1d_roundtrip():
    base = random_fourier_field_1d(8, 64, modes=6, amplitude=1.0, seed=10)
    x = base.reshape(4, 64, 2)
    s = torch.tensor([0.1, -0.2, 0.05, 0.0])
    y = periodic_shift_1d(periodic_shift_1d(x, s), -s)
    assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def test_periodic_shift_2d_roundtrip():
    x = random_fourier_field_2d(3, 32, smoothness=2.0, seed=11)[..., None]
    s = torch.tensor([[0.1, -0.2], [0.05, 0.03], [0.0, 0.0]])
    y = periodic_shift_2d(periodic_shift_2d(x, s), -s)
    assert torch.allclose(x, y, atol=1e-5, rtol=1e-5)


def test_translation1d_shapes():
    transform = Translation1D(max_shift=0.1)
    x = torch.randn(5, 32, 1)
    sample = transform.sample(x.shape[0], x.device, x.dtype)
    y = transform.apply_input(x, sample)
    assert y.shape == x.shape
    assert sample.epsilon.shape == (5,)


def test_burgers_galilean_shapes():
    transform = Burgers1DGalilean(max_boost=0.2, final_time=0.5)
    x = torch.randn(5, 32, 1)
    sample = transform.sample(x.shape[0], x.device, x.dtype)
    y_in = transform.apply_input(x, sample)
    y_out = transform.apply_output(x, sample)
    assert y_in.shape == x.shape
    assert y_out.shape == x.shape


def test_navier_stokes_2d_galilean_boost_channels_and_output_shape():
    transform = NavierStokes2DGalilean(max_boost=0.2, final_time=0.5)
    x = torch.zeros(2, 8, 8, 3)
    sample = TransformSample(
        params={"boost": torch.tensor([[0.1, -0.2], [-0.05, 0.03]])},
        epsilon=torch.ones(2),
        name="navier_stokes2d_galilean",
    )
    y_in = transform.apply_input(x, sample)
    y_out = transform.apply_output(x[..., :1], sample)
    assert y_in.shape == x.shape
    assert y_out.shape == x[..., :1].shape
    assert torch.allclose(y_in[:, :, :, 0], x[:, :, :, 0])
    assert torch.allclose(y_in[0, :, :, 1], torch.full((8, 8), 0.1))
    assert torch.allclose(y_in[0, :, :, 2], torch.full((8, 8), -0.2))


def test_d4_scalar2d_shape_and_manual_sample():
    transform = D4Scalar2D()
    x = torch.randn(2, 16, 16, 1)
    sample = TransformSample(
        params={"k": torch.tensor([1, 2]), "flip": torch.tensor([False, True])},
        epsilon=torch.ones(2),
        name="d4_scalar2d",
    )
    y = transform.apply_input(x, sample)
    assert y.shape == x.shape


def test_periodic_shift_2d_integer_axis_convention():
    x = torch.arange(4 * 5, dtype=torch.float32).reshape(1, 4, 5, 1)
    y_x = periodic_shift_2d(x, torch.tensor([[1.0 / 5.0, 0.0]]))
    y_y = periodic_shift_2d(x, torch.tensor([[0.0, 1.0 / 4.0]]))
    assert torch.allclose(y_x, torch.roll(x, shifts=1, dims=2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(y_y, torch.roll(x, shifts=1, dims=1), atol=1e-5, rtol=1e-5)


def test_d4_pseudoscalar_reflection_flips_sign():
    transform = D4Pseudoscalar2D()
    x = torch.ones(2, 4, 4, 1)
    sample = TransformSample(
        params={"k": torch.tensor([0, 0]), "flip": torch.tensor([False, True])},
        epsilon=torch.ones(2),
        name="d4_pseudoscalar2d",
    )
    y = transform.apply_input(x, sample)
    assert torch.allclose(y[0], torch.ones_like(y[0]))
    assert torch.allclose(y[1], -torch.ones_like(y[1]))
