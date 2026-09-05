#!/usr/bin/env python
"""Remove disposable validation caches, preserving all experiment evidence."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    targets = [ROOT / name for name in ("runs/smoke", "runs/quality", ".pytest_cache", ".ruff_cache")]
    for folder in ("src", "scripts", "tests"):
        targets.extend((ROOT / folder).rglob("__pycache__"))
    for target in targets:
        resolved = target.resolve()
        if not resolved.is_relative_to(ROOT) or resolved == ROOT:
            raise ValueError(f"Cleanup target escapes the repository: {target}")
        if target.is_symlink():
            continue
        if target.is_dir():
            shutil.rmtree(target)
            print(f"removed {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
