# LOCO: Local Orbit Consistency for Symmetry-Robust Neural Operators

LOCO trains neural operators to give consistent predictions under symmetry
transformations. It adds a training loss without changing the inference model.

At 2% labels on the N64 Galilean benchmark, the five-seed FNO results are:

| Method | Orbit OOD relative L2 | Equivariance defect |
| --- | ---: | ---: |
| Supervised augmentation | 0.5276 ± 0.0525 | 0.3816 ± 0.0322 |
| Augmentation + LOCO, weight 0.10 | 0.3950 ± 0.0334 | 0.2677 ± 0.0174 |
| Augmentation + LOCO, weight 0.30 | 0.3416 ± 0.0289 | 0.2044 ± 0.0160 |

Values are means ± sample standard deviations. Weight 0.30 gives the main
result; label-efficiency, severity, and backbone comparisons use weight 0.10.

## Install

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

For locked dependencies, use `uv sync --locked --python 3.11`.

## Reproduce

From the repository root:

```bash
python scripts/verify_paper_results.py
python scripts/reproduce_release.py --dry-run
python scripts/reproduce_release.py --preflight
python scripts/reproduce_release.py --suite main --execute
```

Omit `--suite main` to run all experiments. Use `--stage data`, `train`,
`evaluate`, `reports`, or `figures` to select a stage. Experiment definitions,
seeds, and dataset hashes are listed in [configs/release.yaml](configs/release.yaml).

PDE datasets are generated when needed. The rMD17 reference split is included
under `data/rmd17/`. See [Reproduction](docs/reproducibility.md) for experiment
stages and checkpoint evaluation.
