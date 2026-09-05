from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _pair(size: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError(f"Expected a two-dimensional size, got {size!r}")
    return int(size[0]), int(size[1])


def _periodic_pad2d(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Periodically extend a channel-first tensor, including for large halos."""
    height, width = x.shape[-2:]
    if pad_h <= height and pad_w <= width:
        return F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="circular")

    rows = torch.arange(-pad_h, height + pad_h, device=x.device).remainder(height)
    columns = torch.arange(-pad_w, width + pad_w, device=x.device).remainder(width)
    return x.index_select(-2, rows).index_select(-1, columns)


def _compatible_halo(minimum_halo: int, input_size: int, output_size: int) -> int:
    """Choose a halo whose resized width is an integer number of samples."""
    period = input_size // math.gcd(input_size, output_size)
    return ((minimum_halo + period - 1) // period) * period


def periodic_resize2d(
    x: torch.Tensor,
    output_size: int | Sequence[int],
    *,
    halo: int = 8,
) -> torch.Tensor:
    """Bicubically resize a periodic field without introducing boundary edges.

    PyTorch's antialiased interpolation is applied to a circularly extended
    field.  Cropping the corresponding resized halo makes every retained
    output sample see the correct periodic neighbours.  The halo is enlarged
    when necessary so that its resized width is integral; consequently the
    padded interpolation has exactly the same sampling ratio as the requested
    unpadded interpolation.

    Args:
        x: Tensor with shape ``[batch, channels, height, width]``.
        output_size: Requested output ``(height, width)`` or a square size.
        halo: Minimum circular halo in input-grid samples.
    """
    if x.ndim != 4:
        raise ValueError(
            "periodic_resize2d expects [batch, channels, height, width], "
            f"got {tuple(x.shape)}"
        )
    output_height, output_width = _pair(output_size)
    if output_height < 1 or output_width < 1:
        raise ValueError("periodic_resize2d output dimensions must be positive")
    if halo < 1:
        raise ValueError("periodic_resize2d halo must be positive")

    input_height, input_width = x.shape[-2:]
    if (input_height, input_width) == (output_height, output_width):
        return x

    pad_h = _compatible_halo(halo, input_height, output_height)
    pad_w = _compatible_halo(halo, input_width, output_width)
    crop_h = pad_h * output_height // input_height
    crop_w = pad_w * output_width // input_width
    padded = _periodic_pad2d(x, pad_h, pad_w)
    resized = F.interpolate(
        padded,
        size=(output_height + 2 * crop_h, output_width + 2 * crop_w),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized[..., crop_h : crop_h + output_height, crop_w : crop_w + output_width]


class PeriodicFilteredLeakyReLU2d(nn.Module):
    """CNO filtered activation specialized to a two-dimensional torus."""

    def __init__(
        self,
        *,
        output_scale: float = 1.0,
        upsample_factor: int = 2,
        negative_slope: float = 0.01,
        resample_halo: int = 8,
    ):
        super().__init__()
        if output_scale not in {0.5, 1.0, 2.0}:
            raise ValueError("output_scale must be one of 0.5, 1.0, or 2.0")
        if upsample_factor < 1:
            raise ValueError("upsample_factor must be positive")
        self.output_scale = output_scale
        self.upsample_factor = upsample_factor
        self.negative_slope = negative_slope
        self.resample_halo = resample_halo

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        upsampled = periodic_resize2d(
            x,
            (self.upsample_factor * height, self.upsample_factor * width),
            halo=self.resample_halo,
        )
        activated = F.leaky_relu(upsampled, negative_slope=self.negative_slope)
        output_size = (round(self.output_scale * height), round(self.output_scale * width))
        return periodic_resize2d(activated, output_size, halo=self.resample_halo)


def _normalization(channels: int, use_batch_norm: bool) -> nn.Module:
    return nn.BatchNorm2d(channels) if use_batch_norm else nn.Identity()


def _periodic_conv2d(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        padding=1,
        padding_mode="circular",
    )


class _CNOBlock2d(nn.Module):
    """Periodic convolution, optional batch normalization, and CNO activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        output_scale: float,
        use_batch_norm: bool,
        negative_slope: float,
        resample_halo: int,
    ):
        super().__init__()
        self.convolution = _periodic_conv2d(in_channels, out_channels)
        self.batch_norm = _normalization(out_channels, use_batch_norm)
        self.activation = PeriodicFilteredLeakyReLU2d(
            output_scale=output_scale,
            negative_slope=negative_slope,
            resample_halo=resample_halo,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.batch_norm(self.convolution(x)))


class _LiftProjectBlock2d(nn.Module):
    """Two-convolution lifting or projection block used by the reference CNO."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        latent_channels: int,
        negative_slope: float,
        resample_halo: int,
    ):
        super().__init__()
        self.intermediate = _CNOBlock2d(
            in_channels,
            latent_channels,
            output_scale=1.0,
            use_batch_norm=False,
            negative_slope=negative_slope,
            resample_halo=resample_halo,
        )
        self.convolution = _periodic_conv2d(latent_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolution(self.intermediate(x))


class _ResidualBlock2d(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        use_batch_norm: bool,
        negative_slope: float,
        resample_halo: int,
    ):
        super().__init__()
        self.convolution1 = _periodic_conv2d(channels, channels)
        self.batch_norm1 = _normalization(channels, use_batch_norm)
        self.activation = PeriodicFilteredLeakyReLU2d(
            negative_slope=negative_slope,
            resample_halo=resample_halo,
        )
        self.convolution2 = _periodic_conv2d(channels, channels)
        self.batch_norm2 = _normalization(channels, use_batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.convolution1(x)
        residual = self.activation(self.batch_norm1(residual))
        residual = self.batch_norm2(self.convolution2(residual))
        return x + residual


def _residual_stack(
    channels: int,
    blocks: int,
    *,
    use_batch_norm: bool,
    negative_slope: float,
    resample_halo: int,
) -> nn.Sequential:
    return nn.Sequential(
        *[
            _ResidualBlock2d(
                channels,
                use_batch_norm=use_batch_norm,
                negative_slope=negative_slope,
                resample_halo=resample_halo,
            )
            for _ in range(blocks)
        ]
    )


class CNO2d(nn.Module):
    """Periodic two-dimensional Convolutional Neural Operator.

    The encoder/decoder, residual paths, filtered LeakyReLU, and channel
    schedule follow the official simplified CNO implementation accompanying
    Raonic et al., *Convolutional Neural Operators for Robust and Accurate
    Learning of PDEs* (NeurIPS 2023):
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator

    This compact reimplementation changes the tensor interface to the
    repository's ``[batch, height, width, channels]`` convention and makes all
    convolutions and filtered resampling explicitly periodic.  Batch
    normalization is disabled by default so spatially constant Galilean boost
    channels retain their sample-specific values.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4,
        n_res: int = 4,
        n_res_neck: int = 3,
        channel_multiplier: int = 20,
        lift_project_channels: int = 64,
        use_batch_norm: bool = False,
        negative_slope: float = 0.01,
        resample_halo: int = 8,
    ):
        super().__init__()
        integer_arguments = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "n_layers": n_layers,
            "n_res": n_res,
            "n_res_neck": n_res_neck,
            "channel_multiplier": channel_multiplier,
            "lift_project_channels": lift_project_channels,
            "resample_halo": resample_halo,
        }
        for name, value in integer_arguments.items():
            if value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.n_res = n_res
        self.n_res_neck = n_res_neck
        self.channel_multiplier = channel_multiplier
        self.lift_project_channels = lift_project_channels
        self.use_batch_norm = use_batch_norm
        self.negative_slope = negative_slope
        self.resample_halo = resample_halo
        self.lift_channels = channel_multiplier // 2

        self.encoder_features = [self.lift_channels]
        self.encoder_features.extend(
            (2**level) * channel_multiplier for level in range(n_layers)
        )
        decoder_features_in = list(reversed(self.encoder_features[1:]))
        self.decoder_features_out = list(reversed(self.encoder_features[:-1]))
        for level in range(1, n_layers):
            decoder_features_in[level] *= 2
        self.decoder_features_in = decoder_features_in

        self.lift = _LiftProjectBlock2d(
            in_channels,
            self.encoder_features[0],
            latent_channels=lift_project_channels,
            negative_slope=negative_slope,
            resample_halo=resample_halo,
        )
        self.project = _LiftProjectBlock2d(
            self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels,
            latent_channels=lift_project_channels,
            negative_slope=negative_slope,
            resample_halo=resample_halo,
        )

        self.encoder = nn.ModuleList(
            [
                _CNOBlock2d(
                    self.encoder_features[level],
                    self.encoder_features[level + 1],
                    output_scale=0.5,
                    use_batch_norm=use_batch_norm,
                    negative_slope=negative_slope,
                    resample_halo=resample_halo,
                )
                for level in range(n_layers)
            ]
        )
        self.encoder_decoder_expansion = nn.ModuleList(
            [
                _CNOBlock2d(
                    channels,
                    channels,
                    output_scale=1.0,
                    use_batch_norm=use_batch_norm,
                    negative_slope=negative_slope,
                    resample_halo=resample_halo,
                )
                for channels in self.encoder_features
            ]
        )
        self.decoder = nn.ModuleList(
            [
                _CNOBlock2d(
                    self.decoder_features_in[level],
                    self.decoder_features_out[level],
                    output_scale=2.0,
                    use_batch_norm=use_batch_norm,
                    negative_slope=negative_slope,
                    resample_halo=resample_halo,
                )
                for level in range(n_layers)
            ]
        )
        self.residual_stacks = nn.ModuleList(
            [
                _residual_stack(
                    self.encoder_features[level],
                    n_res,
                    use_batch_norm=use_batch_norm,
                    negative_slope=negative_slope,
                    resample_halo=resample_halo,
                )
                for level in range(n_layers)
            ]
        )
        self.neck = _residual_stack(
            self.encoder_features[-1],
            n_res_neck,
            use_batch_norm=use_batch_norm,
            negative_slope=negative_slope,
            resample_halo=resample_halo,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"CNO2d expects [batch, height, width, channels], got {tuple(x.shape)}")
        _, height, width, channels = x.shape
        if channels != self.in_channels:
            raise ValueError(
                f"CNO2d expects {self.in_channels} input channels, got {channels}"
            )
        divisor = 2**self.n_layers
        if height % divisor or width % divisor:
            raise ValueError(
                "CNO2d spatial dimensions must be divisible by "
                f"2**n_layers={divisor}, got {(height, width)}"
            )

        x = self.lift(x.permute(0, 3, 1, 2))
        skip: list[torch.Tensor] = []
        for residual_stack, encoder in zip(self.residual_stacks, self.encoder):
            skip.append(residual_stack(x))
            # The reference CNO sends the pre-residual stream through its D block;
            # the residual stream is retained as the corresponding decoder skip.
            x = encoder(x)

        x = self.neck(x)
        for level, decoder in enumerate(self.decoder):
            expansion_index = self.n_layers - level
            if level == 0:
                x = self.encoder_decoder_expansion[expansion_index](x)
            else:
                expanded_skip = self.encoder_decoder_expansion[expansion_index](skip[-level])
                x = torch.cat((x, expanded_skip), dim=1)
            x = decoder(x)

        x = torch.cat((x, self.encoder_decoder_expansion[0](skip[0])), dim=1)
        x = self.project(x)
        return x.permute(0, 2, 3, 1)


__all__ = ["CNO2d", "PeriodicFilteredLeakyReLU2d", "periodic_resize2d"]
