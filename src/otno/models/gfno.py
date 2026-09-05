from __future__ import annotations

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


_D4_ELEMENTS: tuple[tuple[int, bool], ...] = tuple(
    (k, flip) for flip in (False, True) for k in range(4)
)
_D4_IDENTITY = _D4_ELEMENTS.index((0, False))


def _d4_apply(x: torch.Tensor, index: int) -> torch.Tensor:
    k, flip = _D4_ELEMENTS[index]
    y = torch.rot90(x, k, dims=(-2, -1))
    if flip:
        y = torch.flip(y, dims=(-1,))
    return y


def _d4_parity(index: int) -> float:
    return -1.0 if _D4_ELEMENTS[index][1] else 1.0


@lru_cache(maxsize=1)
def _d4_tables() -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    marker = torch.arange(25).reshape(5, 5)
    transformed = [_d4_apply(marker, i) for i in range(8)]
    compose: list[list[int]] = [[0 for _ in range(8)] for _ in range(8)]
    for a in range(8):
        for b in range(8):
            image = _d4_apply(_d4_apply(marker, b), a)
            for c, candidate in enumerate(transformed):
                if torch.equal(image, candidate):
                    compose[a][b] = c
                    break
            else:  # pragma: no cover - impossible unless the D4 action is edited incorrectly.
                raise RuntimeError("Failed to build D4 composition table")

    inverse: list[int] = [0 for _ in range(8)]
    for a in range(8):
        for b in range(8):
            if compose[a][b] == _D4_IDENTITY and compose[b][a] == _D4_IDENTITY:
                inverse[a] = b
                break
        else:  # pragma: no cover
            raise RuntimeError("Failed to build D4 inverse table")
    return tuple(tuple(row) for row in compose), tuple(inverse)


def _center_slices(size: int, modes: int) -> tuple[slice, slice]:
    retained = min(modes, size // 2)
    full = slice(size // 2 - retained, size // 2 + retained + 1)
    param = slice(modes - retained, modes + retained + 1)
    return full, param


class D4GroupPointwiseConv2d(nn.Module):
    """Pointwise convolution over channels and the D4 regular representation."""

    def __init__(self, in_channels: int, out_channels: int, *, bias: bool = True):
        super().__init__()
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(8, in_channels, out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def _weight_bank(self) -> torch.Tensor:
        compose, inverse = _d4_tables()
        rows = []
        for s in range(8):
            rows.append(torch.stack([self.weight[compose[inverse[s]][t]] for t in range(8)], dim=0))
        return torch.stack(rows, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 8:
            raise ValueError(f"D4 group features must have shape [b, c, 8, h, w], got {tuple(x.shape)}")
        out = torch.einsum("bithw,stio->bothw", x, self._weight_bank())
        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)
        return out


class D4GroupSpectralConv2d(nn.Module):
    """Fourier-domain D4 group convolution for p4m-style regular features."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        if int(modes1) != int(modes2):
            raise ValueError("D4GroupSpectralConv2d requires modes1 == modes2 for D4 rotations")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, in_channels * out_channels * 8)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                8,
                in_channels,
                out_channels,
                2 * self.modes1 + 1,
                2 * self.modes2 + 1,
                dtype=torch.cfloat,
            )
        )

    def _weight_bank(self, h_param: slice, w_param: slice) -> torch.Tensor:
        weights = self.weights[..., h_param, w_param]
        compose, inverse = _d4_tables()
        bank = []
        for s in range(8):
            row = []
            for t in range(8):
                relative = compose[inverse[s]][t]
                row.append(_d4_apply(weights[relative], s))
            bank.append(torch.stack(row, dim=0))
        return torch.stack(bank, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 8:
            raise ValueError(f"D4 group features must have shape [b, c, 8, h, w], got {tuple(x.shape)}")
        _, _, _, h, w = x.shape
        h_full, h_param = _center_slices(h, self.modes1)
        w_full, w_param = _center_slices(w, self.modes2)
        x_ft = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))
        x_low = x_ft[..., h_full, w_full]
        weights = self._weight_bank(h_param, w_param)
        out_low = torch.einsum("bithw,stiohw->bothw", x_low, weights)
        out_ft = x_ft.new_zeros(x.shape[0], self.out_channels, 8, h, w)
        out_ft[..., h_full, w_full] = out_low
        out_ft = torch.fft.ifftshift(out_ft, dim=(-2, -1))
        return torch.fft.ifft2(out_ft, dim=(-2, -1)).real


class D4Lift(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, pseudoscalar: bool):
        super().__init__()
        self.pseudoscalar = bool(pseudoscalar)
        self.linear = nn.Linear(in_channels, out_channels, bias=not self.pseudoscalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x).permute(0, 3, 1, 2)
        channels = []
        for s in range(8):
            sign = _d4_parity(s) if self.pseudoscalar else 1.0
            channels.append(sign * x)
        return torch.stack(channels, dim=2)


class D4Project(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, pseudoscalar: bool):
        super().__init__()
        self.pseudoscalar = bool(pseudoscalar)
        self.linear = nn.Linear(in_channels, out_channels, bias=not self.pseudoscalar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 8:
            raise ValueError(f"D4 group features must have shape [b, c, 8, h, w], got {tuple(x.shape)}")
        pieces = []
        for s in range(8):
            sign = _d4_parity(s) if self.pseudoscalar else 1.0
            y = x[:, :, s].permute(0, 2, 3, 1)
            pieces.append(sign * self.linear(y))
        return torch.stack(pieces, dim=0).mean(dim=0)


class D4GFNO2d(nn.Module):
    """D4/p4m group-equivariant FNO for scalar or pseudoscalar fields.

    The hidden representation is a feature field over the D4 stabilizer. Each
    Fourier layer performs a regular-representation group convolution in the
    frequency domain, following the p4m G-FNO construction.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 12,
        modes1: int = 8,
        modes2: int = 8,
        depth: int = 4,
        add_grid: bool = False,
        input_pseudoscalar: bool = True,
        output_pseudoscalar: bool = True,
    ):
        super().__init__()
        if add_grid:
            raise ValueError("D4GFNO2d requires add_grid=false; coordinate grids are not D4 scalars.")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        self.depth = int(depth)
        self.add_grid = False
        self.input_pseudoscalar = bool(input_pseudoscalar)
        self.output_pseudoscalar = bool(output_pseudoscalar)

        self.lift = D4Lift(
            self.in_channels,
            self.width,
            pseudoscalar=self.input_pseudoscalar,
        )
        self.spectral = nn.ModuleList(
            [
                D4GroupSpectralConv2d(self.width, self.width, self.modes1, self.modes2)
                for _ in range(self.depth)
            ]
        )
        self.pointwise = nn.ModuleList(
            [D4GroupPointwiseConv2d(self.width, self.width) for _ in range(self.depth)]
        )
        self.post1 = D4GroupPointwiseConv2d(self.width, 2 * self.width)
        self.post2 = D4GroupPointwiseConv2d(2 * self.width, self.width)
        self.project = D4Project(
            self.width,
            self.out_channels,
            pseudoscalar=self.output_pseudoscalar,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"D4GFNO2d expects [batch, h, w, channels], got {tuple(x.shape)}")
        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = F.gelu(spectral(x) + pointwise(x))
        x = F.gelu(self.post1(x))
        x = F.gelu(self.post2(x))
        return self.project(x)


__all__ = ["D4GFNO2d"]
