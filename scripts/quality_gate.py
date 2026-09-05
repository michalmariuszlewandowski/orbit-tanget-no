#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the repo quality gate used before paper experiments.")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-solver-validation", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[variable] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    _run([sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"], env=env)
    _run([sys.executable, "scripts/check_configs.py"], env=env)
    _run([sys.executable, "scripts/run_matrix.py", "--matrix", "configs/experiments_core.yaml", "--dry-run"], env=env)
    if not args.skip_tests:
        _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], env=env)
    if not args.skip_solver_validation:
        _run([sys.executable, "scripts/validate_solvers.py"], env=env)
    if not args.skip_smoke:
        _run([sys.executable, "scripts/smoke_test.py", "--work-dir", "runs/quality/smoke", "--device", "cpu"], env=env)
    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
