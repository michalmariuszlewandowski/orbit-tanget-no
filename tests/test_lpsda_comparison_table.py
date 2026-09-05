from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_table_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "make_lpsda_kdv_faithful_comparison.py"
    )
    spec = importlib.util.spec_from_file_location("make_lpsda_kdv_faithful_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_published_reference_uses_matching_kdv_40s_ar_rows():
    module = _load_table_module()
    assert module.PUBLISHED_ROWS == (
        ("FNO (AR), no augmentation", "none", 0.1248, 0.0108),
        ("FNO (AR) + LPSDA (g1g2g3g4)", "g1g2g3g4", 0.0606, 0.0040),
    )


def test_paper_nmse_supports_legacy_cumulative_and_normalized_metrics():
    module = _load_table_module()
    assert module._paper_nmse({"trajectory_nmse": 12.0}, rollout_steps=100) == 0.12
    assert (
        module._paper_nmse(
            {"trajectory_nmse": 0.12, "trajectory_nmse_cumulative": 12.0},
            rollout_steps=100,
        )
        == 0.12
    )
