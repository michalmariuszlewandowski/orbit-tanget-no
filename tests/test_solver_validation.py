import torch

from otno.data.solvers1d import random_fourier_field_1d, solve_burgers_1d
from otno.data.solvers2d import random_fourier_field_2d, solve_navier_stokes_vorticity_2d
from otno.symmetry.transforms import D4Pseudoscalar2D, TransformSample


def _rel(x: torch.Tensor, y: torch.Tensor) -> float:
    return float(torch.linalg.norm((x - y).reshape(-1)) / torch.linalg.norm(y.reshape(-1)).clamp_min(1e-12))


def test_burgers_dt_halving_smoke_converges():
    u0 = random_fourier_field_1d(2, 64, modes=4, amplitude=0.2, seed=12)
    coarse = solve_burgers_1d(u0, viscosity=0.02, final_time=0.02, dt=0.002)
    fine = solve_burgers_1d(u0, viscosity=0.02, final_time=0.02, dt=0.001)
    assert _rel(coarse, fine) < 2e-4


def test_navier_stokes_dt_halving_smoke_converges():
    omega0 = random_fourier_field_2d(2, 16, smoothness=4.0, amplitude=0.1, seed=13)
    coarse = solve_navier_stokes_vorticity_2d(omega0, viscosity=1e-2, final_time=0.01, dt=0.002)
    fine = solve_navier_stokes_vorticity_2d(omega0, viscosity=1e-2, final_time=0.01, dt=0.001)
    assert _rel(coarse, fine) < 2e-4


def test_navier_stokes_d4_pseudoscalar_commutes_with_solver_small_time():
    omega0 = random_fourier_field_2d(3, 16, smoothness=4.0, amplitude=0.1, seed=14)[..., None]
    transform = D4Pseudoscalar2D()
    sample = TransformSample(
        params={"k": torch.tensor([1, 2, 0]), "flip": torch.tensor([False, False, True])},
        epsilon=torch.ones(3),
        name="d4_pseudoscalar2d",
    )
    left = solve_navier_stokes_vorticity_2d(
        transform.apply_input(omega0, sample)[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002
    )[..., None]
    right = transform.apply_output(
        solve_navier_stokes_vorticity_2d(omega0[..., 0], viscosity=1e-2, final_time=0.01, dt=0.002)[..., None],
        sample,
    )
    assert _rel(left, right) < 1e-6
