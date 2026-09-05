from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class TensorDictDataset(Dataset):
    def __init__(self, a: torch.Tensor, u: torch.Tensor):
        if a.shape[0] != u.shape[0]:
            raise ValueError("Input and output tensors must have the same first dimension")
        self.a = a.float().contiguous()
        self.u = u.float().contiguous()

    def __len__(self) -> int:
        return self.a.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"a": self.a[idx], "u": self.u[idx]}


def load_tensor_dataset(
    path: str | Path,
    split: str,
    *,
    fraction: float = 1.0,
    seed: int = 0,
) -> tuple[TensorDictDataset, dict[str, Any]]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "splits" not in payload or split not in payload["splits"]:
        raise KeyError(f"Dataset {path} does not contain split {split}")
    data = payload["splits"][split]
    a = data["a"]
    u = data["u"]
    if fraction < 1.0:
        n = max(1, int(round(a.shape[0] * fraction)))
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(a.shape[0], generator=g)[:n]
        a = a[idx]
        u = u[idx]
    return TensorDictDataset(a, u), payload.get("metadata", {})
