# Local Orbit Consistency for Symmetry-Robust Neural Operators

This anonymous artifact contains the code, final N64 Galilean configs, cached
tables, and figures for the submitted result on stochastic local orbit
consistency. Manuscript source files and LaTeX build products are intentionally
excluded.

The central claim tested by this package is:

```text
A neural operator trained with local orbit consistency has lower
symmetry-induced OOD error and lower equivariance defect than supervised FNO
baselines, with unchanged inference architecture.
```

The included evidence is the periodic `64 x 64` 2D Navier-Stokes vorticity
setting with Galilean boosts. The final headline method is FNO plus supervised
augmentation plus orbit consistency.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
```

## Artifact Map

```text
src/otno/
  Core implementation for FNO training, Galilean transforms, metrics, and data.

configs/pilot/2d_navier_stokes_galilean.yaml
  Base N64 Galilean experiment config.

configs/ablations/2d_galilean_n64_*.yaml
  Final N64 label-efficiency, lambda, headline, and OOD-severity sweeps.

runs/paper_tables/
  Cached CSV/TeX tables used for the submitted N64 result.

runs/figures/
  Cached paper-facing N64 figures.

scripts/reproduce_paper_artifacts.py
  Regenerates the final cached tables and figures from completed local runs.

scripts/make_anonymous_submission.py
  Builds the curated anonymous ZIP with an explicit allowlist.
```

Large generated datasets, checkpoints, full run directories, manuscript sources,
and pivoted experiment caches are not included in the ZIP.

## Included Results

The cached artifact contains:

```text
runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.*
runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.*
runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.*
runs/paper_tables/2d_galilean_n64_2pct_lambda_extended.*
runs/paper_tables/2d_galilean_n64_2pct_lambda_robustness.*
runs/paper_tables/2d_galilean_n64_training_compute.*
runs/paper_tables/2d_galilean_n64_defect_ood_correlation.*
runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization*
runs/paper_tables/2d_galilean_n64_final_manifest.csv
runs/paper_tables/2d_galilean_n64_claim_summary.csv
runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.{png,pdf}
runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.{png,pdf}
runs/figures/2d_galilean_n64_defect_ood_correlation.{png,pdf}
```

The final artifact inventory is documented in
`docs/final_paper_artifacts.md`.

## Reproducing Cached Artifacts

If the completed training and severity-evaluation run directories are present,
regenerate the final cached tables and figures with:

```bash
python scripts/reproduce_paper_artifacts.py
```

To redraw the figures from the included cached CSV tables:

```bash
python scripts/plot_galilean_n64_paper_figures.py
```

To build the anonymous ZIP:

```bash
python scripts/check_anonymity.py
python scripts/make_anonymous_submission.py
```

## Running Experiments From Configs

Generate the N64 Galilean dataset:

```bash
python scripts/generate_data.py --config configs/pilot/2d_navier_stokes_galilean.yaml
```

Run the headline five-seed matrix:

```bash
python scripts/run_matrix.py --matrix configs/ablations/2d_galilean_n64_2pct_headline_5seed.yaml
```

Run the OOD-severity evaluations:

```bash
python scripts/run_ood_severity_matrix.py --matrix configs/ablations/2d_galilean_n64_2pct_lambda_0p1_ood_severity.yaml
```

Each complete training run writes `config.yaml`, `meta.json`, RNG snapshots,
metric JSON/JSONL files, and `checkpoints/best.pt`. The paper-facing metrics
are `relative_l2`, `orbit_ood_relative_l2`,
`equivariance_defect_relative`, `latency_ms_per_sample`,
`best_val_relative_l2`, `parameters`, and `method`.
