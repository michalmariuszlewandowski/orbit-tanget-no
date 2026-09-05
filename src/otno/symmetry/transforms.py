from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F


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


def periodic_derivative_1d(x: torch.Tensor, *, length: float = 1.0) -> torch.Tensor:
    """Return the spatial derivative of periodic 1D fields."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, n, channels], got {tuple(x.shape)}")
    _, n, _ = x.shape
    x_ch = x.permute(0, 2, 1)
    x_ft = torch.fft.rfft(x_ch, dim=-1)
    freqs = torch.fft.rfftfreq(n, d=length / n, device=x.device).to(x.dtype)
    deriv_ft = (2j * math.pi * freqs[None, None, :]) * x_ft
    y = torch.fft.irfft(deriv_ft, n=n, dim=-1)
    return y.permute(0, 2, 1).contiguous()


def nonperiodic_shift_1d(x: torch.Tensor, shift: torch.Tensor | float, *, length: float = 1.0) -> torch.Tensor:
    """Return f(x - shift) on [0, length] with zero padding outside the interval."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, n, channels], got {tuple(x.shape)}")
    batch, n, _ = x.shape
    shift_t = _as_batch_vector(shift, batch, x.device, x.dtype)
    grid_x = torch.linspace(0.0, length, n, device=x.device, dtype=x.dtype)
    source = grid_x[None, :] - shift_t[:, None]
    source_norm = 2.0 * source / length - 1.0
    grid = torch.stack([source_norm, torch.zeros_like(source_norm)], dim=-1).unsqueeze(1)
    x_ch = x.permute(0, 2, 1).unsqueeze(2)
    y = F.grid_sample(x_ch, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    return y.squeeze(2).permute(0, 2, 1).contiguous()


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


def periodic_gradient_2d(x: torch.Tensor, *, length: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return x- and y-derivatives of periodic 2D fields."""
    if x.ndim != 4:
        raise ValueError(f"Expected [batch, h, w, channels], got {tuple(x.shape)}")
    _, h, w, _ = x.shape
    x_ch = x.permute(0, 3, 1, 2)
    x_ft = torch.fft.fft2(x_ch, dim=(-2, -1))
    freq_y = torch.fft.fftfreq(h, d=length / h, device=x.device).to(x.dtype)
    freq_x = torch.fft.fftfreq(w, d=length / w, device=x.device).to(x.dtype)
    grad_x_ft = (2j * math.pi * freq_x[None, None, None, :]) * x_ft
    grad_y_ft = (2j * math.pi * freq_y[None, None, :, None]) * x_ft
    grad_x = torch.fft.ifft2(grad_x_ft, dim=(-2, -1)).real
    grad_y = torch.fft.ifft2(grad_y_ft, dim=(-2, -1)).real
    return (
        grad_x.permute(0, 2, 3, 1).contiguous(),
        grad_y.permute(0, 2, 3, 1).contiguous(),
    )


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

    def sample_tangent(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> TransformSample:
        return self.sample(batch_size, device, dtype)

    def input_tangent(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not define an infinitesimal input action")

    def output_tangent(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        raise NotImplementedError(f"{self.name} does not define an infinitesimal output action")


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

    def input_tangent(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        shift = sample.params["shift"].view(-1, 1, 1)
        return -shift * periodic_derivative_1d(a, length=self.length)

    def output_tangent(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        shift = sample.params["shift"].view(-1, 1, 1)
        return -shift * periodic_derivative_1d(u, length=self.length)


class NonPeriodicTranslation1D(BaseTransform):
    name = "nonperiodic_translation1d"

    def __init__(
        self,
        max_shift: float = 0.1,
        length: float = 1.0,
        use_mask: bool = True,
        mask_margin: float = 0.0,
    ):
        self.max_shift = float(max_shift)
        self.length = float(length)
        self.use_mask = bool(use_mask)
        self.mask_margin = float(mask_margin)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        shift = (2 * torch.rand(batch_size, device=device, dtype=dtype) - 1) * self.max_shift
        eps = shift.abs().clamp_min(1e-6)
        return TransformSample({"shift": shift}, eps, self.name)

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return nonperiodic_shift_1d(a, sample.params["shift"], length=self.length)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return nonperiodic_shift_1d(u, sample.params["shift"], length=self.length)

    def output_mask(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor | None:
        if not self.use_mask:
            return None
        if u.ndim != 3:
            raise ValueError(f"{self.name} expects [batch, n, channels], got {tuple(u.shape)}")
        batch, n, _ = u.shape
        shift = sample.params["shift"].to(device=u.device, dtype=u.dtype)
        grid_x = torch.linspace(0.0, self.length, n, device=u.device, dtype=u.dtype)
        source = grid_x[None, :] - shift[:, None]
        return ((source >= self.mask_margin) & (source <= self.length - self.mask_margin)).to(u.dtype)


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

    def input_tangent(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        shift = sample.params["shift"]
        grad_x, grad_y = periodic_gradient_2d(a, length=self.length)
        return -shift[:, 0].view(-1, 1, 1, 1) * grad_x - shift[:, 1].view(-1, 1, 1, 1) * grad_y

    def output_tangent(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        shift = sample.params["shift"]
        grad_x, grad_y = periodic_gradient_2d(u, length=self.length)
        return -shift[:, 0].view(-1, 1, 1, 1) * grad_x - shift[:, 1].view(-1, 1, 1, 1) * grad_y


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

    def input_tangent(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        tangent = torch.zeros_like(a)
        boost = sample.params["boost"]
        tangent[..., self.channel] = boost.view(-1, 1)
        return tangent

    def output_tangent(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        boost = sample.params["boost"]
        tangent = -boost.view(-1, 1, 1) * self.final_time * periodic_derivative_1d(
            u, length=self.length
        )
        tangent[..., self.channel] = tangent[..., self.channel] + boost.view(-1, 1)
        return tangent


class NavierStokes2DGalilean(BaseTransform):
    """Galilean boosts for 2D vorticity maps with explicit boost channels.

    Inputs are expected to contain ``[omega0, boost_x, boost_y]`` channels. A
    sampled boost delta leaves omega0 unchanged, adds the delta to the boost
    channels, and shifts the final vorticity by delta * final_time.
    """

    name = "navier_stokes2d_galilean"

    def __init__(
        self,
        max_boost: float = 0.5,
        final_time: float = 0.5,
        length: float = 1.0,
        boost_x_channel: int = 1,
        boost_y_channel: int = 2,
    ):
        self.max_boost = float(max_boost)
        self.final_time = float(final_time)
        self.length = float(length)
        self.boost_x_channel = int(boost_x_channel)
        self.boost_y_channel = int(boost_y_channel)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        boost = (2 * torch.rand(batch_size, 2, device=device, dtype=dtype) - 1) * self.max_boost
        eps = torch.linalg.norm(boost, dim=-1).clamp_min(1e-6)
        return TransformSample({"boost": boost}, eps, self.name)

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        channels = max(self.boost_x_channel, self.boost_y_channel) + 1
        if a.ndim != 4 or a.shape[-1] < channels:
            raise ValueError(
                "NavierStokes2DGalilean expects inputs [batch, h, w, channels] "
                f"with at least {channels} channels, got {tuple(a.shape)}"
            )
        boost = sample.params["boost"]
        y = a.clone()
        y[..., self.boost_x_channel] = y[..., self.boost_x_channel] + boost[:, 0].view(-1, 1, 1)
        y[..., self.boost_y_channel] = y[..., self.boost_y_channel] + boost[:, 1].view(-1, 1, 1)
        return y

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        shift = sample.params["boost"] * self.final_time
        return periodic_shift_2d(u, shift, length=self.length)

    def input_tangent(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        channels = max(self.boost_x_channel, self.boost_y_channel) + 1
        if a.ndim != 4 or a.shape[-1] < channels:
            raise ValueError(
                "NavierStokes2DGalilean expects inputs [batch, h, w, channels] "
                f"with at least {channels} channels, got {tuple(a.shape)}"
            )
        boost = sample.params["boost"]
        tangent = torch.zeros_like(a)
        tangent[..., self.boost_x_channel] = boost[:, 0].view(-1, 1, 1)
        tangent[..., self.boost_y_channel] = boost[:, 1].view(-1, 1, 1)
        return tangent

    def output_tangent(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        boost = sample.params["boost"]
        grad_x, grad_y = periodic_gradient_2d(u, length=self.length)
        return -self.final_time * (
            boost[:, 0].view(-1, 1, 1, 1) * grad_x
            + boost[:, 1].view(-1, 1, 1, 1) * grad_y
        )


class D4Scalar2D(BaseTransform):
    name = "d4_scalar2d"

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        k = torch.randint(0, 4, (batch_size,), device=device)
        flip = torch.rand(batch_size, device=device) < 0.5
        return TransformSample({"k": k, "flip": flip}, torch.ones(batch_size, device=device, dtype=dtype), self.name)

    def _apply_spatial(self, x: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"{self.name} expects [batch, h, w, channels], got {tuple(x.shape)}")
        k = sample.params["k"].to(device=x.device)
        flip = sample.params["flip"].to(device=x.device)
        out = []
        for item, rotations, do_flip in zip(x, k.tolist(), flip.tolist()):
            y = torch.rot90(item, int(rotations), dims=(0, 1))
            if bool(do_flip):
                y = torch.flip(y, dims=(1,))
            out.append(y)
        return torch.stack(out, dim=0).contiguous()

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._apply_spatial(a, sample)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._apply_spatial(u, sample)


class D4Pseudoscalar2D(D4Scalar2D):
    name = "d4_pseudoscalar2d"

    def _apply_pseudoscalar(self, x: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        y = self._apply_spatial(x, sample)
        flip = sample.params["flip"].to(device=x.device)
        sign = torch.where(flip, -torch.ones_like(flip, dtype=x.dtype), torch.ones_like(flip, dtype=x.dtype))
        return y * sign.view(-1, 1, 1, 1)

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._apply_pseudoscalar(a, sample)

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._apply_pseudoscalar(u, sample)


class MolecularRigidMotion(BaseTransform):
    name = "molecular_rigid_motion"

    def __init__(self, max_angle: float = math.pi, max_translation: float = 1.0):
        self.max_angle = float(max_angle)
        self.max_translation = float(max_translation)

    def sample(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> TransformSample:
        axis = torch.randn(batch_size, 3, device=device, dtype=dtype)
        axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        angle = (2 * torch.rand(batch_size, device=device, dtype=dtype) - 1) * self.max_angle
        rotation = _axis_angle_rotation(axis, angle)
        translation = (2 * torch.rand(batch_size, 3, device=device, dtype=dtype) - 1) * self.max_translation
        epsilon = (angle.abs() + translation.norm(dim=-1)).clamp_min(1e-6)
        return TransformSample({"rotation": rotation, "translation": translation}, epsilon, self.name)

    def apply_input(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        if a.ndim != 3 or a.shape[-1] < 3:
            raise ValueError(f"{self.name} expects [batch, atoms, channels>=3], got {tuple(a.shape)}")
        rotation = sample.params["rotation"].to(device=a.device, dtype=a.dtype)
        translation = sample.params["translation"].to(device=a.device, dtype=a.dtype)
        y = a.clone()
        y[..., :3] = torch.bmm(a[..., :3], rotation.transpose(1, 2)) + translation[:, None, :]
        return y

    def apply_output(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        if u.ndim != 3 or u.shape[-1] != 3:
            raise ValueError(f"{self.name} expects force outputs [batch, atoms, 3], got {tuple(u.shape)}")
        rotation = sample.params["rotation"].to(device=u.device, dtype=u.dtype)
        return torch.bmm(u, rotation.transpose(1, 2))


def _axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    x, y, z = axis.unbind(dim=-1)
    c = torch.cos(angle)
    s = torch.sin(angle)
    one_c = 1 - c
    return torch.stack(
        [
            c + x * x * one_c,
            x * y * one_c - z * s,
            x * z * one_c + y * s,
            y * x * one_c + z * s,
            c + y * y * one_c,
            y * z * one_c - x * s,
            z * x * one_c - y * s,
            z * y * one_c + x * s,
            c + z * z * one_c,
        ],
        dim=-1,
    ).reshape(axis.shape[0], 3, 3)


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

    def sample_tangent(
        self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> TransformSample:
        return self.sample(batch_size, device, dtype)

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

    def input_tangent(self, a: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._select(sample).input_tangent(a, sample)

    def output_tangent(self, u: torch.Tensor, sample: TransformSample) -> torch.Tensor:
        return self._select(sample).output_tangent(u, sample)
