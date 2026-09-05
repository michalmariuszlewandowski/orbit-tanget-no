# LOCO: Local Orbit Consistency for Symmetry-Robust Neural Operators

Code and reproducibility artifacts for the **TMLR rebuttal v11** manuscript.
LOCO compares predictions on transformed inputs with transformed predictions
during training, preserving the base model's inference architecture.

At 2% labels on the N64 Galilean benchmark, the five-seed FNO comparison gives:

| Method | Orbit OOD relative L2 | Equivariance defect |
| --- | ---: | ---: |
| Supervised augmentation | 0.5276 ± 0.0525 | 0.3816 ± 0.0322 |
| Augmentation + LOCO, weight 0.10 | 0.3950 ± 0.0334 | 0.2677 ± 0.0174 |
| Augmentation + LOCO, weight 0.30 | 0.3416 ± 0.0289 | 0.2044 ± 0.0160 |

Weight 0.30 is the stronger v11 headline; weight 0.10 is the fixed setting for
label-efficiency, severity, and backbone comparisons. The release also retains
the CNO result where predictive improvement is unresolved.

## Install and validate

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/quality_gate.py
python scripts/verify_paper_results.py
```

For a locked installation, use `uv sync --locked --extra dev --python 3.11`.
`pyproject.toml` owns the dependency list; `requirements.txt` delegates to it.
The quality gate runs lint, config validation, tests, solver identities, and a
CPU training smoke test. `make quality` invokes the same gate where Make is available.

## Reproduce

The cross-platform workflow in [configs/release.yaml](configs/release.yaml) maps
the PDF's tables and figures to existing training, evaluation, and reporting code.

```bash
# Read-only: inspect all commands and check prerequisites.
python scripts/reproduce_release.py --dry-run
python scripts/reproduce_release.py --preflight

# Execute the main FNO suite, including data, training, evaluation, and figures.
python scripts/reproduce_release.py --suite main --execute

# Execute all manuscript suites. Full CPU training can take days.
python scripts/reproduce_release.py --execute
```

Use `--stage data`, `train`, `evaluate`, `reports`, or `figures` to select a stage.
Completed matching training runs are reused; incomplete protected runs fail before
work begins. Use a fresh extracted checkout for new training experiments.

**Reproducibility has two levels.** Cached per-seed records reproduce the listed
PDF statistics, and archived checkpoints can reproduce their evaluation metrics.
Fresh training follows the published scientific recipe, but historical chunked
training used an incorrect resume RNG path, so it cannot be promised to recreate
those weights bit for bit. See [the reproduction guide](docs/reproducibility.md)
for exact-data requirements, suite coverage, and validation limits.

## Release contents

| Path | Purpose |
| --- | --- |
| `src/otno/` | Models, group actions, losses, solvers, and training |
| `configs/release.yaml` | One index of the v11 experiment suites |
| `scripts/` | Training, evaluation, reporting, and release entry points |
| `figure_specs/` | Qualitative selection rule and printed numerical references |
| `tests/` | Numerical identities and implementation regression tests |
| `runs/paper_tables/`, `runs/figures/` | Curated evidence records and figures |

Build and verify the curated code archive and separate exact-data archive:

```bash
python scripts/make_anonymous_submission.py --out dist/local_orbit_consistency_code.zip --data-out dist/local_orbit_consistency_data.zip
python scripts/make_anonymous_submission.py --verify dist/local_orbit_consistency_code.zip
python scripts/make_anonymous_submission.py --verify dist/local_orbit_consistency_data.zip
```

Extract both archives into the same empty directory. The data archive preserves
the exact five dataset files and their SHA-256 hashes. The small historical rMD17
split is also included in the code archive. Full trained checkpoints are separate
run artifacts. The release omits duplicated rebuttal trees, manuscript drafts,
local environments, external source checkouts, and unrelated pilot results.

Code license: [MIT](LICENSE). Dataset and dependency licenses remain with their
respective sources. Current verification evidence is recorded in [VALIDATION.md](VALIDATION.md).
