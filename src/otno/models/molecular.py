from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MoleculeMLP(nn.Module):
    """A deliberately non-equivariant fixed-molecule force predictor.

    Inputs have shape ``[batch, atoms, in_channels]`` and outputs have shape
    ``[batch, atoms, out_channels]``. This baseline is intentionally simple:
    it gives the orbit objective a portable vector-output model without
    introducing a specialized molecular architecture.
    """

    def __init__(
        self,
        *,
        n_atoms: int,
        in_channels: int = 4,
        out_channels: int = 3,
        hidden: int = 128,
        depth: int = 4,
    ):
        super().__init__()
        if n_atoms < 1:
            raise ValueError("n_atoms must be positive")
        if depth < 1:
            raise ValueError("depth must be positive")
        self.n_atoms = int(n_atoms)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden = int(hidden)
        self.depth = int(depth)
        layers: list[nn.Module] = []
        in_features = self.n_atoms * self.in_channels
        for _ in range(self.depth):
            layers.append(nn.Linear(in_features, self.hidden))
            in_features = self.hidden
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(in_features, self.n_atoms * self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"MoleculeMLP expects [batch, atoms, channels], got {tuple(x.shape)}")
        if x.shape[1] != self.n_atoms or x.shape[2] != self.in_channels:
            raise ValueError(
                "MoleculeMLP input shape mismatch: "
                f"expected atoms={self.n_atoms}, channels={self.in_channels}, got {tuple(x.shape)}"
            )
        y = x.reshape(x.shape[0], self.n_atoms * self.in_channels)
        for layer in self.layers:
            y = F.silu(layer(y))
        y = self.head(y)
        return y.reshape(x.shape[0], self.n_atoms, self.out_channels)
