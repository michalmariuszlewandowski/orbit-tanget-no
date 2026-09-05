from __future__ import annotations

import math

import torch


def make_grid_2d(n: int, *, device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    x = torch.arange(n, device=device, dtype=dtype) / n
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def random_fourier_field_2d(
    num: int,
    n: int,
    *,
    smoothness: float = 8.0,
    amplitude: float = 1.0,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate smooth periodic scalar fields on [0, 1)^2. Returns [num, n, n]."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn(num, n, n, generator=generator, device=device)
    kx = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    ky = torch.fft.fftfreq(n, d=1.0 / n, device=device)
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    filt = torch.exp(-k2 / smoothness).to(noise.dtype)
    field = torch.fft.ifft2(torch.fft.fft2(noise) * filt[None, :, :]).real
    field = field - field.mean(dim=(-2, -1), keepdim=True)
    field = field / (field.std(dim=(-2, -1), keepdim=True) + 1e-6)
    return amplitude * field


def _wavenumbers_2d(n: int, length: float, device: torch.device, dtype: torch.dtype):
    kx = 2 * math.pi * torch.fft.fftfreq(n, d=length / n, device=device).to(dtype)
    ky = 2 * math.pi * torch.fft.fftfreq(n, d=length / n, device=device).to(dtype)
    return kx, ky


def dealias_2d(u: torch.Tensor, *, keep_fraction: float = 2.0 / 3.0) -> torch.Tensor:
    """Apply a tensor-product 2/3-rule spectral filter to real fields [..., h, w]."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    h, w = u.shape[-2:]
    ky = torch.fft.fftfreq(h, d=1.0 / h, device=u.device)
    kx = torch.fft.fftfreq(w, d=1.0 / w, device=u.device)
    mask_y = ky.abs() <= keep_fraction * (h // 2)
    mask_x = kx.abs() <= keep_fraction * (w // 2)
    mask = mask_y[:, None] & mask_x[None, :]
    u_hat = torch.fft.fft2(u, dim=(-2, -1))
    return torch.fft.ifft2(u_hat * mask, dim=(-2, -1)).real


def navier_stokes_vorticity_rhs_2d(
    omega: torch.Tensor,
    *,
    viscosity: float,
    force: torch.Tensor | None = None,
    ambient_velocity: torch.Tensor | tuple[float, float] | None = None,
    length: float = 1.0,
    dealias: bool = True,
) -> torch.Tensor:
    """Vorticity-form 2D incompressible Navier--Stokes RHS on a periodic square.

    Equation: omega_t + v · grad omega = nu Delta omega + force,
    with v = (psi_y, -psi_x) and Delta psi = omega. Tensors use shape
    [batch, y, x], so x-derivatives act on the last axis and y-derivatives on
    the penultimate axis.
    """
    if omega.ndim != 3:
        raise ValueError(f"Expected omega [batch, n, n], got {tuple(omega.shape)}")
    n = omega.shape[-1]
    if omega.shape[-2] != n:
        raise ValueError("This pilot solver expects square periodic grids")
    kx, ky = _wavenumbers_2d(n, length, omega.device, omega.dtype)
    ky_grid = ky[None, :, None]
    kx_grid = kx[None, None, :]
    k2 = kx_grid**2 + ky_grid**2

    omega_hat = torch.fft.fft2(omega, dim=(-2, -1))
    denominator = torch.where(k2 == 0, torch.ones_like(k2), k2)
    psi_hat = torch.where(k2 == 0, torch.zeros_like(omega_hat), -omega_hat / denominator)

    vel_x = torch.fft.ifft2(1j * ky_grid * psi_hat, dim=(-2, -1)).real
    vel_y = torch.fft.ifft2(-1j * kx_grid * psi_hat, dim=(-2, -1)).real
    if ambient_velocity is not None:
        boost = torch.as_tensor(ambient_velocity, device=omega.device, dtype=omega.dtype)
        if boost.ndim == 1 and boost.numel() == 2:
            boost = boost[None, :].expand(omega.shape[0], 2)
        if boost.shape != (omega.shape[0], 2):
            raise ValueError(f"ambient_velocity must be [2] or [batch, 2], got {tuple(boost.shape)}")
        vel_x = vel_x + boost[:, 0].view(-1, 1, 1)
        vel_y = vel_y + boost[:, 1].view(-1, 1, 1)
    omega_x = torch.fft.ifft2(1j * kx_grid * omega_hat, dim=(-2, -1)).real
    omega_y = torch.fft.ifft2(1j * ky_grid * omega_hat, dim=(-2, -1)).real
    lap = torch.fft.ifft2(-k2 * omega_hat, dim=(-2, -1)).real
    advective = vel_x * omega_x + vel_y * omega_y
    if dealias:
        advective = dealias_2d(advective)
    rhs = -advective + viscosity * lap
    if force is not None:
        rhs = rhs + force
    return rhs


def _rk4_step_ns(
    omega: torch.Tensor,
    dt: float,
    *,
    viscosity: float,
    force,
    ambient_velocity,
    length: float,
    dealias: bool,
) -> torch.Tensor:
    k1 = navier_stokes_vorticity_rhs_2d(
        omega,
        viscosity=viscosity,
        force=force,
        ambient_velocity=ambient_velocity,
        length=length,
        dealias=dealias,
    )
    k2 = navier_stokes_vorticity_rhs_2d(
        omega + 0.5 * dt * k1,
        viscosity=viscosity,
        force=force,
        ambient_velocity=ambient_velocity,
        length=length,
        dealias=dealias,
    )
    k3 = navier_stokes_vorticity_rhs_2d(
        omega + 0.5 * dt * k2,
        viscosity=viscosity,
        force=force,
        ambient_velocity=ambient_velocity,
        length=length,
        dealias=dealias,
    )
    k4 = navier_stokes_vorticity_rhs_2d(
        omega + dt * k3,
        viscosity=viscosity,
        force=force,
        ambient_velocity=ambient_velocity,
        length=length,
        dealias=dealias,
    )
    return omega + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_navier_stokes_vorticity_2d(
    omega0: torch.Tensor,
    *,
    viscosity: float = 1e-3,
    final_time: float = 0.5,
    dt: float = 1e-3,
    force: torch.Tensor | None = None,
    ambient_velocity: torch.Tensor | tuple[float, float] | None = None,
    length: float = 1.0,
    dealias: bool = True,
) -> torch.Tensor:
    """Pseudo-spectral RK4 solver for 2D periodic vorticity dynamics.

    The nonlinear advective term is de-aliased by default. Check time-step
    convergence by halving ``dt`` on a held-out subset.
    """
    steps = max(1, int(math.ceil(final_time / dt)))
    h = final_time / steps
    omega = omega0.clone()
    for _ in range(steps):
        omega = _rk4_step_ns(
            omega,
            h,
            viscosity=viscosity,
            force=force,
            ambient_velocity=ambient_velocity,
            length=length,
            dealias=dealias,
        )
    return omega
