#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the preserved rMD17 tensor against raw NPZ old_indices."
    )
    parser.add_argument(
        "--raw",
        default="data/rmd17/raw/rmd17/npz_data/rmd17_ethanol.npz",
    )
    parser.add_argument(
        "--processed",
        default="data/rmd17/rmd17_ethanol_force_500.pt",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    processed_path = Path(args.processed)
    raw = np.load(raw_path)
    payload = torch.load(processed_path, map_location="cpu", weights_only=False)

    raw_ids = [int(value) for value in raw["old_indices"]]
    if len(raw_ids) != len(set(raw_ids)):
        raise SystemExit("Raw old_indices are not unique.")
    lookup = {value: index for index, value in enumerate(raw_ids)}

    charges = torch.as_tensor(raw["nuclear_charges"], dtype=torch.float32)
    charge_scale = float(payload["metadata"].get("charge_scale", 10.0))
    charge_channel = (charges / charge_scale).view(1, -1, 1)
    max_input_diff = 0.0
    max_force_diff = 0.0
    matched = 0

    for split in ("train", "val", "test"):
        cached = payload["splits"][split]
        selected_ids = [int(value) for value in cached["old_indices"].tolist()]
        missing = [value for value in selected_ids if value not in lookup]
        if missing:
            raise SystemExit(f"{split}: {len(missing)} old_indices are absent from raw NPZ.")
        rows = [lookup[value] for value in selected_ids]
        coords = torch.as_tensor(raw["coords"][rows], dtype=torch.float32)
        coords = coords - coords.mean(dim=1, keepdim=True)
        inputs = torch.cat(
            [coords, charge_channel.expand(coords.shape[0], -1, -1)], dim=-1
        )
        forces = torch.as_tensor(raw["forces"][rows], dtype=torch.float32)
        max_input_diff = max(max_input_diff, float((inputs - cached["a"]).abs().max()))
        max_force_diff = max(max_force_diff, float((forces - cached["u"]).abs().max()))
        matched += len(rows)

    result = {
        "matched_samples": matched,
        "max_input_abs_diff": max_input_diff,
        "max_force_abs_diff": max_force_diff,
        "raw_sha256": _sha256(raw_path),
        "processed_sha256": _sha256(processed_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if max_input_diff != 0.0 or max_force_diff != 0.0:
        raise SystemExit("Cached rMD17 tensors do not exactly match the raw indexed samples.")


if __name__ == "__main__":
    main()
