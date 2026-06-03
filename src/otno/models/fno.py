from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, n]
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )
        modes = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :modes], self.weights[:, :, :modes]
        )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes: int = 16,
        depth: int = 4,
        add_grid: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.depth = depth
        self.add_grid = add_grid
        lifted_channels = in_channels + (1 if add_grid else 0)
        self.fc0 = nn.Linear(lifted_channels, width)
        self.spectral = nn.ModuleList([SpectralConv1d(width, width, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv1d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, out_channels)

    def _grid(self, batch: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        grid = torch.arange(n, device=device, dtype=dtype) / n
        return grid[None, :, None].expand(batch, n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"FNO1d expects [batch, n, channels], got {tuple(x.shape)}")
        batch, n, _ = x.shape
        if self.add_grid:
            x = torch.cat([x, self._grid(batch, n, x.device, x.dtype)], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = F.gelu(spectral(x) + pointwise(x))
        x = x.permute(0, 2, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        self.weights_pos = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights_neg = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, channels, h, w]
        batch, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            h,
            w // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights_pos[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -m1:, :m2], self.weights_neg[:, :, :m1, :m2]
        )
        return torch.fft.irfft2(out_ft, s=(h, w), dim=(-2, -1))


class FNO2d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,
        modes1: int = 12,
        modes2: int = 12,
        depth: int = 4,
        add_grid: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.depth = depth
        self.add_grid = add_grid
        lifted_channels = in_channels + (2 if add_grid else 0)
        self.fc0 = nn.Linear(lifted_channels, width)
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes1, modes2) for _ in range(depth)]
        )
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.fc1 = nn.Linear(width, 2 * width)
        self.fc2 = nn.Linear(2 * width, out_channels)

    def _grid(self, batch: int, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        y = torch.arange(h, device=device, dtype=dtype) / h
        x = torch.arange(w, device=device, dtype=dtype) / w
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)
        return grid[None, :, :, :].expand(batch, h, w, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FNO2d expects [batch, h, w, channels], got {tuple(x.shape)}")
        batch, h, w, _ = x.shape
        if self.add_grid:
            x = torch.cat([x, self._grid(batch, h, w, x.device, x.dtype)], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = F.gelu(spectral(x) + pointwise(x))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        return self.fc2(x)
