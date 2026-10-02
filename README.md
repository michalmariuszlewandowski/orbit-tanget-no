# LOCO: Local Orbit Consistency for Symmetry-Robust Neural Operators

LOCO trains neural operators to give consistent predictions under symmetry
transformations. It adds a training loss without changing the inference model.

The finite-orbit loss compares the prediction on a transformed input with the
transformed prediction on the original input. The `aug_orbit` method combines
this loss with ordinary and symmetry-augmented supervision. The repository also
includes tangent propagation, canonicalization, alternative backbones, and
controls for the transformation and loss normalization.

## Implementation

The numerical method lives in `src/otno/`; command-line entry points and
experiment-specific reports live in `scripts/`.

| Location | Contents |
| --- | --- |
| [`training/losses.py`](src/otno/training/losses.py) | Supervised, finite-orbit, and tangent losses |
| [`training/objectives.py`](src/otno/training/objectives.py) | Method definitions and joint minibatch objectives |
| [`symmetry/`](src/otno/symmetry/) | Input/output actions, sampling, and infinitesimal generators |
| [`models/`](src/otno/models/) | FNO, DeepONet, CNO, D4-equivariant, and canonicalized models |
| [`data/`](src/otno/data/) | PDE solvers, dataset generation, and tensor loading |
| [`training/trainer.py`](src/otno/training/trainer.py) | Optimization, validation, checkpoints, and resume |
| [`reporting.py`](src/otno/reporting.py), [`plotting/`](src/otno/plotting/) | Protocol checks, seed statistics, and field figures |
| [`configs/release.yaml`](configs/release.yaml) | Experiment suites, dependencies, and report commands |

See the [code guide](docs/code-guide.md) for tensor conventions, configuration
fields, and the path from a single experiment to a reported result.

## Install

Use Python 3.11. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

For locked dependencies, use `uv sync --locked --python 3.11`.

For a small CPU run that generates its own data:

```bash
python scripts/smoke_test.py --work-dir runs/smoke --device cpu
```

For an annotated experiment you can modify, run:

```bash
python scripts/train.py --config configs/examples/advection_quickstart.yaml
```

The [advection YAML](configs/examples/advection_quickstart.yaml) explains every
setting and generates its own small dataset. The [walkthrough](docs/code-guide.md#runnable-advection-example)
lists every output file. Reusable settings live under `configs/problems/`;
manuscript sweeps remain under `configs/ablations/`.
See the [loss equations](docs/code-guide.md#equations-and-loss-implementation)
and [numerical assumptions and verified cases](docs/numerics.md) before extending
the experiments.

## Reproduce figures from saved results

The saved per-seed CSV results regenerate the statistical plots without model
checkpoints:

```bash
python scripts/verify_paper_results.py
python scripts/plot_galilean_n64_paper_figures.py
python scripts/make_paper_diagnostics.py --cached-compute
```

The qualitative fields, fixed-stress diagnostics, and equivariance residuals
use the two existing local seed-23 checkpoints at the original run paths listed
in [figure_specs/2d_galilean_n64_id_ood_seed23.yaml](figure_specs/2d_galilean_n64_id_ood_seed23.yaml).
With those checkpoints and their Galilean N64 dataset present, regenerate all
figures with:

```bash
python scripts/reproduce_release.py --suite main --stage figures --preflight
python scripts/reproduce_release.py --suite main --stage figures --execute
```

Figures are written as PNG and PDF files under `runs/figures/`. Preflight checks
the input hashes recorded in the figure spec.
Preserve those local checkpoints for figure edits; retrain other models as needed.

## Retrain the experiments

From the repository root:

```bash
python scripts/reproduce_release.py --dry-run
python scripts/reproduce_release.py --preflight
python scripts/reproduce_release.py --suite main --execute
```

The default workflow runs data generation, training, evaluation, and reports.
Regenerate the frozen paper figures separately using the commands above;
retrained checkpoints have new provenance and different file hashes.

Omit `--suite main` to run all experiments. Use `--stage data`, `train`,
`evaluate`, or `reports` to select a stage. Experiment definitions,
seeds, and dataset hashes are listed in [configs/release.yaml](configs/release.yaml).

Use a fresh checkout for retraining; completed local archives may have pruned
training checkpoints. The workflow generates missing PDE datasets. The rMD17
reference split is included under `data/rmd17/`. Reports are CSV files.
See [Reproduction](docs/reproducibility.md) for experiment stages, checkpoint
evaluation, and the distinction between regenerating figures and retraining.

To reclaim space after completed experiments, preview removable final-training
checkpoints with `python scripts/prune_release_checkpoints.py`. See
[checkpoint cleanup](docs/reproducibility.md#reclaim-checkpoint-space-after-experiments)
for the removal command and retained artifacts.

## Development

Install the development dependencies, then run formatting, configuration,
numerical, and training checks:

```bash
python -m pip install -e ".[dev]"
python scripts/quality_gate.py
```

Use `python -m ruff format src scripts tests` to apply the repository's formatting.
The tests check solver convergence and symmetry identities, loss behavior,
checkpoint reproduction, and report protocols. Saved paper values can be checked
independently with `python scripts/verify_paper_results.py`.
