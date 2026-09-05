from __future__ import annotations

import math

import torch
from torch import nn


def _mlp(
    in_features: int,
    hidden_features: int,
    out_features: int,
    hidden_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    features = in_features
    for _ in range(hidden_layers):
        layers.extend([nn.Linear(features, hidden_features), nn.GELU()])
        features = hidden_features
    layers.append(nn.Linear(features, out_features))
    return nn.Sequential(*layers)


def _periodic_coordinate_features(height: int, width: int, modes: int) -> torch.Tensor:
    """Return deterministic axial Fourier features on a periodic rectangular grid."""
    y = torch.arange(height, dtype=torch.float32) / height
    x = torch.arange(width, dtype=torch.float32) / width
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    frequencies = torch.arange(1, modes + 1, dtype=torch.float32)
    phase_x = 2.0 * math.pi * xx[..., None] * frequencies
    phase_y = 2.0 * math.pi * yy[..., None] * frequencies
    features = torch.cat(
        [phase_x.sin(), phase_x.cos(), phase_y.sin(), phase_y.cos()], dim=-1
    )
    return features.reshape(height * width, 4 * modes)


class DeepONet2d(nn.Module):
    """Sensorized 2D DeepONet with a periodic coordinate trunk.

    The convolutional branch network observes every input-grid sensor.  The
    trunk network embeds each output coordinate using fixed axial Fourier
    features.  Their latent vectors are combined by the standard DeepONet
    inner product.  Circular branch padding reflects the periodic domain but
    does not encode Galilean equivariance.
    """

    def __init__(
        self,
        *,
        in_channels: int = 1,
        out_channels: int = 1,
        grid_height: int = 64,
        grid_width: int = 64,
        branch_channels: tuple[int, ...] = (32, 64, 128, 256),
        branch_fc_hidden: int = 480,
        trunk_hidden: int = 256,
        trunk_depth: int = 3,
        latent_dim: int = 256,
        coordinate_modes: int = 12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_height = grid_height
        self.grid_width = grid_width
        if not branch_channels or any(channel < 1 for channel in branch_channels):
            raise ValueError("branch_channels must contain positive integers")
        self.branch_channels = tuple(branch_channels)
        self.branch_fc_hidden = branch_fc_hidden
        self.trunk_hidden = trunk_hidden
        self.trunk_depth = trunk_depth
        self.latent_dim = latent_dim
        self.coordinate_modes = coordinate_modes

        joint_output = out_channels * latent_dim
        branch_layers: list[nn.Module] = []
        channels = in_channels
        branch_height, branch_width = grid_height, grid_width
        for output_channels in self.branch_channels:
            branch_layers.extend(
                [
                    nn.Conv2d(
                        channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        padding_mode="circular",
                    ),
                    nn.GELU(),
                ]
            )
            channels = output_channels
            branch_height = (branch_height + 1) // 2
            branch_width = (branch_width + 1) // 2
        branch_flattened = channels * branch_height * branch_width
        branch_layers.extend(
            [
                nn.Flatten(),
                nn.Linear(branch_flattened, branch_fc_hidden),
                nn.GELU(),
                nn.Linear(branch_fc_hidden, joint_output),
            ]
        )
        self.branch = nn.Sequential(*branch_layers)
        self.trunk = _mlp(4 * coordinate_modes, trunk_hidden, joint_output, trunk_depth)
        self.output_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer(
            "coordinate_features",
            _periodic_coordinate_features(grid_height, grid_width, coordinate_modes),
            persistent=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"DeepONet2d expects [batch, height, width, channels], got {tuple(x.shape)}"
            )
        batch, height, width, channels = x.shape
        expected = (self.grid_height, self.grid_width, self.in_channels)
        if (height, width, channels) != expected:
            raise ValueError(
                "DeepONet2d input shape does not match its fixed sensor grid: "
                f"expected [batch, {expected[0]}, {expected[1]}, {expected[2]}], "
                f"got {tuple(x.shape)}"
            )

        branch = self.branch(x.permute(0, 3, 1, 2)).reshape(
            batch, self.out_channels, self.latent_dim
        )
        trunk = self.trunk(self.coordinate_features.to(dtype=x.dtype)).reshape(
            height * width, self.out_channels, self.latent_dim
        )
        output = torch.einsum("bor,por->bpo", branch, trunk) / math.sqrt(self.latent_dim)
        output = output + self.output_bias
        return output.reshape(batch, height, width, self.out_channels)
