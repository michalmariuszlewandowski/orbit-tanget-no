from __future__ import annotations

import math

import torch


def make_grid_1d(n: int, *, device: torch.device | None = None, dtype=torch.float32) -> torch.Tensor:
    return torch.arange(n, device=device, dtype=dtype) / n


def random_fourier_field_1d(
    num: int,
    n: int,
    *,
    modes: int = 8,
    decay: float = 1.5,
    amplitude: float = 1.0,
    mean: float = 0.0,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Generate smooth periodic scalar fields on [0, 1). Returns [num, n]."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    x = make_grid_1d(n, device=device)
    k = torch.arange(1, modes + 1, device=device, dtype=x.dtype)
    basis_sin = torch.sin(2 * math.pi * k[:, None] * x[None, :])
    basis_cos = torch.cos(2 * math.pi * k[:, None] * x[None, :])
    scale = (k ** (-decay))[None, :, None]
    a = torch.randn(num, modes, 1, generator=generator, device=device, dtype=x.dtype)
    b = torch.randn(num, modes, 1, generator=generator, device=device, dtype=x.dtype)
    fields = ((a * basis_sin[None, :, :] + b * basis_cos[None, :, :]) * scale).sum(dim=1)
    fields = fields - fields.mean(dim=-1, keepdim=True)
    fields = fields / (fields.std(dim=-1, keepdim=True) + 1e-6)
    return amplitude * fields + mean


def _fft_frequencies_1d(n: int, length: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return 2 * math.pi * torch.fft.fftfreq(n, d=length / n, device=device).to(dtype)


def dealias_1d(u: torch.Tensor, *, keep_fraction: float = 2.0 / 3.0) -> torch.Tensor:
    """Apply a 2/3-rule spectral filter to real periodic fields u[..., n]."""
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    n = u.shape[-1]
    freqs = torch.fft.fftfreq(n, d=1.0 / n, device=u.device)
    cutoff = keep_fraction * (n // 2)
    mask = freqs.abs() <= cutoff
    u_hat = torch.fft.fft(u, dim=-1)
    return torch.fft.ifft(u_hat * mask, dim=-1).real


def spectral_derivatives_1d(u: torch.Tensor, *, length: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first and second periodic spectral derivatives for u[..., n]."""
    n = u.shape[-1]
    k = _fft_frequencies_1d(n, length, u.device, u.dtype)
    u_hat = torch.fft.fft(u, dim=-1)
    ux = torch.fft.ifft(1j * k * u_hat, dim=-1).real
    uxx = torch.fft.ifft(-(k**2) * u_hat, dim=-1).real
    return ux, uxx


def periodic_shift_1d_spectral(u: torch.Tensor, shift: torch.Tensor | float, *, length: float = 1.0) -> torch.Tensor:
    """Return f(x - shift) for fields u with shape [batch, n]."""
    if u.ndim != 2:
        raise ValueError(f"Expected [batch, n], got {tuple(u.shape)}")
    batch, n = u.shape
    shift_t = torch.as_tensor(shift, device=u.device, dtype=u.dtype)
    if shift_t.ndim == 0:
        shift_t = shift_t.expand(batch)
    if shift_t.shape != (batch,):
        raise ValueError(f"shift must be scalar or [batch], got {tuple(shift_t.shape)}")
    freqs = torch.fft.rfftfreq(n, d=length / n, device=u.device).to(u.dtype)
    phase = torch.exp(-2j * math.pi * shift_t[:, None] * freqs[None, :])
    u_hat = torch.fft.rfft(u, dim=-1)
    return torch.fft.irfft(u_hat * phase, n=n, dim=-1)


def solve_advection_1d(
    u0: torch.Tensor,
    *,
    velocity: float | torch.Tensor = 1.0,
    final_time: float = 1.0,
    length: float = 1.0,
) -> torch.Tensor:
    """Exact solution of u_t + c u_x = 0 on a periodic line."""
    c = torch.as_tensor(velocity, device=u0.device, dtype=u0.dtype)
    shift = c * final_time
    return periodic_shift_1d_spectral(u0, shift, length=length)


def burgers_rhs_1d(
    u: torch.Tensor,
    *,
    viscosity: float,
    length: float = 1.0,
    dealias: bool = True,
) -> torch.Tensor:
    ux, uxx = spectral_derivatives_1d(u, length=length)
    nonlinear = u * ux
    if dealias:
        nonlinear = dealias_1d(nonlinear)
    return -nonlinear + viscosity * uxx


def _rk4_step_burgers(u: torch.Tensor, dt: float, *, viscosity: float, length: float, dealias: bool) -> torch.Tensor:
    k1 = burgers_rhs_1d(u, viscosity=viscosity, length=length, dealias=dealias)
    k2 = burgers_rhs_1d(u + 0.5 * dt * k1, viscosity=viscosity, length=length, dealias=dealias)
    k3 = burgers_rhs_1d(u + 0.5 * dt * k2, viscosity=viscosity, length=length, dealias=dealias)
    k4 = burgers_rhs_1d(u + dt * k3, viscosity=viscosity, length=length, dealias=dealias)
    return u + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_burgers_1d(
    u0: torch.Tensor,
    *,
    viscosity: float = 0.01,
    final_time: float = 0.5,
    dt: float = 1e-3,
    length: float = 1.0,
    dealias: bool = True,
) -> torch.Tensor:
    """Pseudo-spectral RK4 solver for viscous Burgers on a periodic line.

    The nonlinear term is de-aliased by default using a 2/3-rule spectral filter.
    Check time-step convergence by halving ``dt`` on a held-out subset.
    """
    steps = max(1, int(math.ceil(final_time / dt)))
    h = final_time / steps
    u = u0.clone()
    for _ in range(steps):
        u = _rk4_step_burgers(u, h, viscosity=viscosity, length=length, dealias=dealias)
    return u
