import torch

from otno.data.solvers1d import (
    random_fourier_field_1d,
    solve_advection_1d,
    solve_burgers_1d,
)
from otno.symmetry.transforms import Burgers1DGalilean, TransformSample, Translation1D


def test_advection_translation_commutes_with_solver():
    u0 = random_fourier_field_1d(3, 64, modes=4, amplitude=0.5, seed=1)[..., None]
    transform = Translation1D(max_shift=0.2)
    sample = TransformSample(
        params={"shift": torch.tensor([0.1, -0.07, 0.03])},
        epsilon=torch.ones(3),
        name="translation1d",
    )
    left = solve_advection_1d(transform.apply_input(u0, sample)[..., 0], velocity=0.8, final_time=0.25)[..., None]
    right = transform.apply_output(
        solve_advection_1d(u0[..., 0], velocity=0.8, final_time=0.25)[..., None], sample
    )
    assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


def test_burgers_galilean_identity_small_time():
    u0 = random_fourier_field_1d(2, 64, modes=3, amplitude=0.25, seed=2)[..., None]
    final_time = 0.02
    transform = Burgers1DGalilean(max_boost=0.1, final_time=final_time)
    sample = TransformSample(
        params={"boost": torch.tensor([0.08, -0.05])},
        epsilon=torch.ones(2),
        name="burgers1d_galilean",
    )
    left = solve_burgers_1d(
        transform.apply_input(u0, sample)[..., 0], viscosity=0.02, final_time=final_time, dt=0.0005
    )[..., None]
    right = transform.apply_output(
        solve_burgers_1d(u0[..., 0], viscosity=0.02, final_time=final_time, dt=0.0005)[..., None],
        sample,
    )
    max_err = (left - right).abs().max().item()
    assert max_err < 2e-3


def test_navier_stokes_translation_commutes_with_solver_small_time():
    from otno.data.solvers2d import random_fourier_field_2d, solve_navier_stokes_vorticity_2d
    from otno.symmetry.transforms import Translation2D

    omega0 = random_fourier_field_2d(2, 16, smoothness=4.0, amplitude=0.2, seed=5)[..., None]
    transform = Translation2D(max_shift=0.1)
    sample = TransformSample(
        params={"shift": torch.tensor([[1.0 / 16.0, -2.0 / 16.0], [0.0, 1.0 / 16.0]])},
        epsilon=torch.ones(2),
        name="translation2d",
    )
    left = solve_navier_stokes_vorticity_2d(
        transform.apply_input(omega0, sample)[..., 0], viscosity=1e-2, final_time=0.01, dt=0.005
    )[..., None]
    right = transform.apply_output(
        solve_navier_stokes_vorticity_2d(omega0[..., 0], viscosity=1e-2, final_time=0.01, dt=0.005)[..., None],
        sample,
    )
    assert torch.allclose(left, right, atol=2e-4, rtol=2e-4)


def test_navier_stokes_galilean_boost_identity_small_time():
    from otno.data.solvers2d import random_fourier_field_2d, solve_navier_stokes_vorticity_2d
    from otno.symmetry.transforms import NavierStokes2DGalilean

    final_time = 0.02
    omega0 = random_fourier_field_2d(2, 16, smoothness=4.0, amplitude=0.1, seed=7)
    base_boost = torch.tensor([[0.03, -0.02], [-0.01, 0.04]])
    a = torch.cat(
        [
            omega0[..., None],
            base_boost[:, None, None, :].expand(2, 16, 16, 2),
        ],
        dim=-1,
    )
    transform = NavierStokes2DGalilean(max_boost=0.1, final_time=final_time)
    sample = TransformSample(
        params={"boost": torch.tensor([[0.04, -0.03], [-0.02, 0.01]])},
        epsilon=torch.ones(2),
        name="navier_stokes2d_galilean",
    )
    transformed = transform.apply_input(a, sample)
    left = solve_navier_stokes_vorticity_2d(
        transformed[..., 0],
        viscosity=1e-2,
        final_time=final_time,
        dt=0.002,
        ambient_velocity=transformed[:, 0, 0, 1:3],
    )[..., None]
    right = transform.apply_output(
        solve_navier_stokes_vorticity_2d(
            a[..., 0],
            viscosity=1e-2,
            final_time=final_time,
            dt=0.002,
            ambient_velocity=base_boost,
        )[..., None],
        sample,
    )
    assert torch.allclose(left, right, atol=2e-4, rtol=2e-4)
