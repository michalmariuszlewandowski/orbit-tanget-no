from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


def _as_batch_vector(value: torch.Tensor | float, batch: int, device, dtype) -> torch.Tensor:
    value_t = torch.as_tensor(value, device=device, dtype=dtype)
    if value_t.ndim == 0:
        return value_t.expand(batch)
    if value_t.shape == (batch,):
        return value_t
    raise ValueError(f"Expected scalar or [{batch}], got {tuple(value_t.shape)}")


def periodic_shift_1d(x: torch.Tensor, shift: torch.Tensor | float, *, length: float = 1.0) -> torch.Tensor:
    """Return f(x - shift) for tensors with shape [batch, n, channels]."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, n, channels], got {tuple(x.shape)}")
    batch, n, channels = x.shape
    shift_t = _as_batch_vector(shift, batch, x.device, x.dtype)
    x_ch = x.permute(0, 2, 1)
    x_ft = torch.fft.rfft(x_ch, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=length / n, device=x.device).to(x.dtype)
    phase = torch.exp(-2j * math.pi * shift_t[:, None, None] * freqs[None, None, :])
    y = torch.fft.irfft(x_ft * phase, n=n, dim=-1)
    return y.permute(0, 2, 1).contiguous()


def periodic_shift_2d(
    x: torch.Tensor,
    shift: torch.Tensor | tuple[float, float],
    *,
    length: float = 1.0,
) -> torch.Tensor:
    """Return f(x - sx, y - sy) for tensors [batch, h, w, channels].

    The shift tensor has shape [batch, 2] with columns [sx, sy].
    """
    if x.ndim != 4:
        raise ValueError(f"Expected [batch, h, w, channels], got {tuple(x.shape)}")
    batch, h, w, channels = x.shape
    shift_t = torch.as_tensor(shift, device=x.device, dtype=x.dtype)
    if shift_t.ndim == 1 and shift_t.numel() == 2:
        shift_t = shift_t[None, :].expand(batch, 2)
    if shift_t.ndim == 0:
        shift_t = shift_t.expand(batch, 2)
    if shift_t.shape != (batch, 2):
        raise ValueError(f"Expected shift scalar, [2], or [batch, 2], got {tuple(shift_t.shape)}")
    sx = shift_t[:, 0]
    sy = shift_t[:, 1]
    x_ch = x.permute(0, 3, 1, 2)
    x_ft = torch.fft.rfft2(x_ch, dim=(-2, -1))
    freq_y = torch.fft.fftfreq(h, d=length / h, device=x.device).to(x.dtype)
    freq_x = torch.fft.rfftfreq(w, d=length / w, device=x.device).to(x.dtype)
    phase = torch.exp(
        -2j
        * math.pi
        * (
            sy[:, None, None, None] * freq_y[None, None, :, None]
            + sx[:, None, None, None] * freq_x[None, None, None, :]
        )
    )
    y = torch.fft.irfft2(x_ft * phase, s=(h, w), dim=(-2, -1))
    return y.permute(0, 2, 3, 1).contiguous()


@dataclass
class TransformSample:
    params: dict[str, Any]
    epsilon: torch.Tensor
    name: str


class BaseTransform:
    name = "base"

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        raise NotImplementedError

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        raise NotImplementedError

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        raise NotImplementedError

    def output_mask(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor | None:
        return None


class Translation1D(BaseTransform):
    name = "translation1d"

    def __init__(self, max_shift: float = 0.25, length: float = 1.0):
        self.max_shift = float(max_shift)
        self.length = float(length)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        shift = (2 * torch.rand(batch_size, device=device, dtype=dtype) - 1) * self.max_shift
        eps = shift.abs().clamp_min(1e-6)
        return TransformSample({"shift": shift}, eps, self.name)

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return periodic_shift_1d(a, sample.params["shift"], length=self.length)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return periodic_shift_1d(u, sample.params["shift"], length=self.length)


class Translation2D(BaseTransform):
    name = "translation2d"

    def __init__(self, max_shift: float = 0.25, length: float = 1.0):
        self.max_shift = float(max_shift)
        self.length = float(length)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        shift = (2 * torch.rand(batch_size, 2, device=device, dtype=dtype) - 1) * self.max_shift
        eps = torch.linalg.norm(shift, dim=-1).clamp_min(1e-6)
        return TransformSample({"shift": shift}, eps, self.name)

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return periodic_shift_2d(a, sample.params["shift"], length=self.length)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return periodic_shift_2d(u, sample.params["shift"], length=self.length)


class Burgers1DGalilean(BaseTransform):
    name = "burgers1d_galilean"

    def __init__(self, max_boost: float = 0.5, final_time: float = 0.5, channel: int = 0, length: float = 1.0):
        self.max_boost = float(max_boost)
        self.final_time = float(final_time)
        self.channel = int(channel)
        self.length = float(length)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        boost = (2 * torch.rand(batch_size, device=device, dtype=dtype) - 1) * self.max_boost
        eps = boost.abs().clamp_min(1e-6)
        return TransformSample({"boost": boost}, eps, self.name)

    def _add_boost(self, x: torch.Tensor, boost: torch.Tensor) -> torch.Tensor:
        y = x.clone()
        view_shape = [boost.shape[0]] + [1] * (x.ndim - 2) + [1]
        y[..., self.channel] = y[..., self.channel] + boost.view(*view_shape).squeeze(-1)
        return y

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._add_boost(a, sample.params["boost"])

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        boost = sample.params["boost"]
        shifted = periodic_shift_1d(u, boost * self.final_time, length=self.length)
        return self._add_boost(shifted, boost)


class D4Scalar2D(BaseTransform):
    """Discrete dihedral transforms on a square for scalar 2D fields.

    The transform samples rotations by 0, 90, 180, 270 degrees and optional horizontal reflection.
    This is a finite-group consistency baseline, not a Lie-tangent loss.
    """

    name = "d4_scalar2d"

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        k = torch.randint(0, 4, (batch_size,), device=device)
        flip = torch.randint(0, 2, (batch_size,), device=device).bool()
        eps = torch.ones(batch_size, device=device, dtype=dtype)
        return TransformSample({"k": k, "flip": flip}, eps, self.name)

    def _apply(self, x: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [batch, h, w, channels], got {tuple(x.shape)}")
        out = []
        for b in range(x.shape[0]):
            y = torch.rot90(x[b], int(sample.params["k"][b].item()), dims=(0, 1))
            if bool(sample.params["flip"][b].item()):
                y = torch.flip(y, dims=(1,))
            out.append(y)
        return torch.stack(out, dim=0).contiguous()

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._apply(a, sample)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._apply(u, sample)




class D4Pseudoscalar2D(D4Scalar2D):
    """Dihedral transforms for orientation-dependent 2D pseudoscalars such as vorticity.

    Proper rotations act as scalar rotations. Reflections additionally multiply the field by -1.
    This is the correct action for scalar vorticity under orientation-reversing maps.
    """

    name = "d4_pseudoscalar2d"

    def _apply(self, x: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        y = super()._apply(x, sample)
        flip = sample.params["flip"].to(device=x.device)
        sign = torch.where(
            flip,
            torch.full_like(flip, -1, dtype=x.dtype),
            torch.ones_like(flip, dtype=x.dtype),
        )
        return y * sign.view(-1, 1, 1, 1)


class CompositeTransform(BaseTransform):
    """Sample one transform per minibatch from a list."""

    name = "composite"

    def __init__(self, transforms: list[BaseTransform], probabilities: list[float] | None = None):
        if not transforms:
            raise ValueError("CompositeTransform requires at least one transform")
        self.transforms = transforms
        if probabilities is None:
            self.probabilities = torch.ones(len(transforms)) / len(transforms)
        else:
            probs = torch.tensor(probabilities, dtype=torch.float32)
            self.probabilities = probs / probs.sum()
        self._last_transform: BaseTransform | None = None

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        idx = int(torch.multinomial(self.probabilities, num_samples=1).item())
        transform = self.transforms[idx]
        self._last_transform = transform
        sample = transform.sample(batch_size, device, dtype)
        sample.name = transform.name
        sample.params["_composite_idx"] = idx
        return sample

    def _select(self, sample: TransformSample) -> BaseTransform:
        if "_composite_idx" in sample.params:
            return self.transforms[int(sample.params["_composite_idx"])]
        if self._last_transform is None:
            raise RuntimeError("CompositeTransform has no sampled transform")
        return self._last_transform

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._select(sample).apply_input(a, sample)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._select(sample).apply_output(u, sample)

    def output_mask(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor | None:
        return self._select(sample).output_mask(u, sample)
