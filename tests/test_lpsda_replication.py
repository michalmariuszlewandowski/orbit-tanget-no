from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_replication_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_lpsda_kdv_replication.py"
    spec = importlib.util.spec_from_file_location("run_lpsda_kdv_replication", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fourier_shift_accepts_scalar_and_time_dependent_shifts():
    module = _load_replication_module()
    u = torch.randn(5, 16)
    shifted_scalar = module.fourier_shift(u, 0.1, dim=-1)
    shifted_time = module.fourier_shift(u, torch.linspace(-0.1, 0.1, 5)[:, None], dim=-1)
    assert shifted_scalar.shape == u.shape
    assert shifted_time.shape == u.shape


def test_lpsda_fno_forward_shape():
    module = _load_replication_module()
    model = module.LPSDAFNO1d(
        nx=16,
        time_history=4,
        time_future=3,
        modes=4,
        width=8,
        num_layers=1,
    )
    x = torch.randn(2, 16, 4)
    y = model(x, torch.ones(2), torch.ones(2))
    assert y.shape == (2, 16, 3)


def test_faithful_kdv_galilean_window_zero_boost_is_identity():
    module = _load_replication_module()
    x = torch.randn(2, 16, 5)
    y = module._apply_kdv_galilean_window(
        x,
        boost=torch.zeros(2),
        dx=torch.ones(2),
        dt=torch.ones(2) * 0.1,
        offset_steps=3,
    )
    assert torch.allclose(x, y, atol=1e-6)


def test_evaluate_reports_paper_normalized_rollout_nmse():
    module = _load_replication_module()

    class ZeroModel(torch.nn.Module):
        def forward(self, u, dx, dt):
            return torch.zeros(u.shape[0], u.shape[1], 2, device=u.device)

    trajectories = torch.ones(2, 8, 4)
    dx = torch.ones(2)
    dt = torch.ones(2)
    metrics = module._evaluate(
        ZeroModel(),
        [(trajectories, dx, dt)],
        batch_size=2,
        nx=4,
        nt_effective=8,
        time_history=2,
        time_future=2,
        device=torch.device("cpu"),
    )
    assert metrics["rollout_time_steps"] == 4
    assert metrics["trajectory_nmse"] == 1.0
    assert metrics["trajectory_nmse_cumulative"] == 4.0


def test_orbit_evaluate_reports_kdv_galilean_metrics():
    module = _load_replication_module()

    class ZeroModel(torch.nn.Module):
        def forward(self, u, dx, dt):
            return torch.zeros(u.shape[0], u.shape[1], 2, device=u.device)

    trajectories = torch.ones(2, 8, 4)
    dx = torch.ones(2)
    dt = torch.ones(2) * 0.1
    metrics = module._evaluate_orbit(
        ZeroModel(),
        [(trajectories, dx, dt)],
        nx=4,
        nt_effective=8,
        time_history=2,
        time_future=2,
        device=torch.device("cpu"),
        max_boost=0.05,
        orbit_samples=2,
        seed=123,
    )
    assert metrics["orbit_rollout_time_steps"] == 4
    assert metrics["orbit_eval_samples"] == 2
    assert metrics["orbit_ood_trajectory_nmse"] > 0.0
    assert metrics["equivariance_defect_relative"] >= 0.0
