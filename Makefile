PYTHON ?= python
TEST_ENV = PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

.PHONY: install test smoke format lint clean check-configs validate-solvers quality dry-run tables compile

install:
	$(PYTHON) -m pip install -e ".[dev]"

compile:
	PYTHONPATH=src $(PYTHON) -m compileall -q src scripts tests

test:
	$(TEST_ENV) $(PYTHON) -m pytest -q -p no:ddtrace -p no:cacheprovider

smoke:
	$(TEST_ENV) $(PYTHON) scripts/smoke_test.py --work-dir runs/smoke --device cpu

check-configs:
	PYTHONPATH=src $(PYTHON) scripts/check_configs.py

validate-solvers:
	$(TEST_ENV) $(PYTHON) scripts/validate_solvers.py --out runs/quality/solver_validation.json

dry-run:
	PYTHONPATH=src $(PYTHON) scripts/run_matrix.py --matrix configs/experiments_june_september.yaml --dry-run

tables:
	PYTHONPATH=src $(PYTHON) scripts/make_paper_tables.py --runs runs --out-prefix runs/paper_tables/main

quality:
	$(TEST_ENV) $(PYTHON) scripts/quality_gate.py

lint:
	$(PYTHON) -m ruff check src scripts tests

format:
	$(PYTHON) -m ruff format src scripts tests

clean:
	rm -rf runs/smoke runs/quality runs/paper_tables .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
