# Validation status

Validated in the build environment on CPU with `PYTHONPATH=src`, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`.

```text
python -m compileall -q src scripts tests
COMPILE_OK

python scripts/check_configs.py
CONFIG CHECK PASSED

python -m pytest -q -p no:ddtrace -p no:cacheprovider
25 passed in 3.68s

python scripts/validate_solvers.py --out /tmp/otno_solver_validation_vfinal.json
advection_translation_rel:      1.8148470815049222e-07
burgers_galilean_rel:           1.7959644083020976e-07
burgers_dt_halving_rel:         8.527271688762994e-08
ns2d_translation_rel:           1.6780525413651048e-07
ns2d_d4_pseudoscalar_rel:       2.2777593144951425e-08
ns2d_dt_halving_rel:            8.384914451653458e-08
SOLVER VALIDATION PASSED

python scripts/smoke_test.py --work-dir /tmp/otno_smoke_pg_final --device cpu
completed one-epoch CPU train/eval smoke run and wrote full run artifacts
```

`ruff` is listed as a development dependency in `pyproject.toml`; it was not installed in the build environment used for this validation.
