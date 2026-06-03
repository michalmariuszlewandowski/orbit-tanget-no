from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch.

    Deterministic kernels are opt-in because some PyTorch deterministic paths are slower or
    unavailable on specific accelerators. The flag is intended for final paper runs and
    reproducibility audits, not necessarily for rapid sweeps.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed))
    return gen


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def get_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def append_jsonl(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def stable_json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(root: str | Path | None = None) -> str | None:
    root = Path(root or Path.cwd())
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def environment_fingerprint(root: str | Path | None = None) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "git_commit": get_git_commit(root),
    }



def environment_manifest(config: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = environment_fingerprint(Path.cwd())
    manifest["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS")
    manifest["mkl_num_threads"] = os.environ.get("MKL_NUM_THREADS")
    if config is not None:
        manifest["config_hash"] = stable_json_hash(config)[:12]
    return manifest


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


def repo_root_from_env() -> Path:
    return Path(os.environ.get("OTNO_ROOT", Path.cwd()))
