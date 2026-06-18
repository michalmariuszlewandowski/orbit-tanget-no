from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from otno.models.fno import FNO1d, FNO2d
from otno.symmetry.transforms import periodic_shift_1d, periodic_shift_2d


@dataclass
class CanonicalState1D:
    translation_shift: torch.Tensor | None = None
    galilean_mean: torch.Tensor | None = None


@dataclass
class CanonicalState2D:
    galilean_boost: torch.Tensor | None = None


def estimate_first_mode_shift_1d(
    x: torch.Tensor,
    *,
    channel: int = 0,
    mode: int = 1,
    length: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate the shift that makes the selected Fourier mode real-positive.

    For the action T_s f(x)=f(x-s), the Fourier phase of mode k changes by
    -2*pi*k*s/length. The returned shift s_hat satisfies T_{s_hat} f has zero
    phase for the selected mode when that mode has non-negligible amplitude.
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, n, channels], got {tuple(x.shape)}")
    n = x.shape[1]
    if not 1 <= mode <= n // 2:
        raise ValueError(f"mode must be in [1, {n // 2}], got {mode}")
    coeff = torch.fft.rfft(x[..., channel], dim=-1)[:, mode]
    phase = torch.angle(coeff)
    shift = phase * length / (2.0 * math.pi * mode)
    amplitude = coeff.abs()
    # Near-zero modes produce unstable phases. Fall back to zero shift for those cases.
    return torch.where(amplitude > eps, shift, torch.zeros_like(shift))


def observed_galilean_boost_2d(
    x: torch.Tensor,
    *,
    boost_x_channel: int = 1,
    boost_y_channel: int = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """Read a constant Galilean frame coordinate from 2D boost channels."""
    if x.ndim != 4:
        raise ValueError(f"Expected [batch, h, w, channels], got {tuple(x.shape)}")
    channels = max(boost_x_channel, boost_y_channel) + 1
    if x.shape[-1] < channels:
        raise ValueError(f"Expected at least {channels} channels, got {tuple(x.shape)}")
    if reduction == "corner":
        boost_x = x[:, 0, 0, boost_x_channel]
        boost_y = x[:, 0, 0, boost_y_channel]
    elif reduction == "mean":
        boost_x = x[..., boost_x_channel].mean(dim=(1, 2))
        boost_y = x[..., boost_y_channel].mean(dim=(1, 2))
    else:
        raise ValueError(f"Unknown boost reduction: {reduction!r}")
    return torch.stack([boost_x, boost_y], dim=-1)


class CanonicalFNO1d(nn.Module):
    """FNO1d wrapped by lightweight analytic canonicalization.

    Supported canonicalizers:
    - ``translation_first_mode``: phase-align the selected Fourier mode;
    - ``galilean_mean``: subtract the input-channel spatial mean and restore the
      Burgers Galilean action on the output.

    This is intended as a computational baseline against orbit-consistency training.
    It is deliberately simple and deterministic; no frame estimator is learned.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes: int = 16,
        depth: int = 4,
        add_grid: bool = True,
        canonicalizers: list[str] | tuple[str, ...] = ("translation_first_mode",),
        canonical_channel: int = 0,
        translation_mode: int = 1,
        length: float = 1.0,
        final_time: float = 0.5,
    ):
        super().__init__()
        self.base = FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes=modes,
            depth=depth,
            add_grid=add_grid,
        )
        self.canonicalizers = tuple(canonicalizers)
        self.canonical_channel = int(canonical_channel)
        self.translation_mode = int(translation_mode)
        self.length = float(length)
        self.final_time = float(final_time)

    def _canonicalize(self, x: torch.Tensor) -> tuple[torch.Tensor, CanonicalState1D]:
        state = CanonicalState1D()
        z = x
        if "galilean_mean" in self.canonicalizers:
            mean = z[..., self.canonical_channel].mean(dim=1)
            z = z.clone()
            z[..., self.canonical_channel] = z[..., self.canonical_channel] - mean[:, None]
            state.galilean_mean = mean
        if "translation_first_mode" in self.canonicalizers:
            shift = estimate_first_mode_shift_1d(
                z,
                channel=self.canonical_channel,
                mode=self.translation_mode,
                length=self.length,
            )
            z = periodic_shift_1d(z, shift, length=self.length)
            state.translation_shift = shift
        return z, state

    def _decanonicalize(self, y: torch.Tensor, state: CanonicalState1D) -> torch.Tensor:
        z = y
        if state.translation_shift is not None:
            z = periodic_shift_1d(z, -state.translation_shift, length=self.length)
        if state.galilean_mean is not None:
            mean = state.galilean_mean
            z = periodic_shift_1d(z, mean * self.final_time, length=self.length)
            z = z.clone()
            z[..., self.canonical_channel] = z[..., self.canonical_channel] + mean[:, None]
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, state = self._canonicalize(x)
        y = self.base(z)
        return self._decanonicalize(y, state)


class ObservableGalileanCanonicalFNO2d(nn.Module):
    """FNO2d wrapped by observed-frame Galilean canonicalization.

    This is a PACE-style comparison baseline for the boosted vorticity dataset:
    it reads the constant ambient velocity channels from a single input, sets the
    input frame to zero, applies a standard FNO2d, then restores the terminal
    frame by the analytically induced shift ``boost * final_time``.
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        out_channels: int = 1,
        width: int = 64,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        add_grid: bool = True,
        boost_x_channel: int = 1,
        boost_y_channel: int = 2,
        length: float = 1.0,
        final_time: float = 0.5,
        boost_reduction: str = "mean",
    ):
        super().__init__()
        self.base = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            modes1=modes1,
            modes2=modes2,
            depth=depth,
            add_grid=add_grid,
        )
        self.boost_x_channel = int(boost_x_channel)
        self.boost_y_channel = int(boost_y_channel)
        self.length = float(length)
        self.final_time = float(final_time)
        self.boost_reduction = str(boost_reduction)

    def _canonicalize(self, x: torch.Tensor) -> tuple[torch.Tensor, CanonicalState2D]:
        boost = observed_galilean_boost_2d(
            x,
            boost_x_channel=self.boost_x_channel,
            boost_y_channel=self.boost_y_channel,
            reduction=self.boost_reduction,
        )
        z = x.clone()
        z[..., self.boost_x_channel] = 0.0
        z[..., self.boost_y_channel] = 0.0
        return z, CanonicalState2D(galilean_boost=boost)

    def _decanonicalize(self, y: torch.Tensor, state: CanonicalState2D) -> torch.Tensor:
        if state.galilean_boost is None:
            return y
        return periodic_shift_2d(y, state.galilean_boost * self.final_time, length=self.length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, state = self._canonicalize(x)
        y = self.base(z)
        return self._decanonicalize(y, state)
