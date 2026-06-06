# Orbit-Tangent Neural Operators

This repository is a paper-grade experimental scaffold for testing whether stochastic local orbit consistency is a viable ICLR 2027 contribution for symmetry-aware neural operators. It contains runnable pilots, baselines, validation scripts, reproducibility metadata, solver checks, table-generation utilities, and cached N64 Galilean paper artifacts.

The core hypothesis is that a pretrained or jointly trained neural operator can be made more robust along known PDE symmetry orbits by adding a cheap consistency objective

\[
\mathcal{L}_{\mathrm{orb}}
=
\mathbb{E}_{a,i,\epsilon}
\frac{
\left\|
\mathcal{G}_\theta(T_{\exp(\epsilon X_i)}^{\mathcal A}a)
-
T_{\exp(\epsilon X_i)}^{\mathcal U}\mathcal{G}_\theta(a)
\right\|^2
}{\epsilon^2+\eta},
\]

on top of a standard neural-operator backbone. The implementation uses finite group perturbations rather than explicit Jacobian-vector products. With one sampled transform per minibatch, training adds one extra forward pass and inference uses the unchanged base model.

## Installation

```bash
cd orbit-tangent-no
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
make install
```

Run the paper-grade local validation gate first:

```bash
make quality
```

This validates configs and matrix expansion, runs the unit tests, checks numerical solver/symmetry identities, and executes a CPU smoke training run. For a shorter check, run:

```bash
make test
make smoke
```

The smoke test generates a tiny 1D advection dataset, trains a small FNO for one epoch with augmentation plus orbit consistency, and writes outputs to `runs/smoke`.

## Repository map

```text
src/otno/
  data/          dataset generators and spectral PDE solvers
  models/        FNO1d/FNO2d backbones and lightweight analytic canonicalization baselines
  symmetry/      translation, Galilean, and D4 transform actions
  training/      losses, trainer, metrics, latency measurement
scripts/
  generate_data.py
  train.py
  evaluate.py
  run_matrix.py
  summarize_results.py
  make_paper_tables.py
  check_configs.py
  validate_solvers.py
  quality_gate.py
configs/
  pilot/         main pilot configurations
  baselines/     supervised, augmentation, orbit, and combined variants
  ablations/     lambda and label-fraction sweeps
  experiments_june_september.yaml
docs/
  theory.md
  roadmap_iclr2027.md
  experiment_registry.md
  pivot_criteria.md
  paper_grade_protocol.md
  known_limitations.md
paper/
  related_work.bib
  sections_draft.tex
  theory_section.tex
  method_section.tex
```

## Core implementation

A training run is specified by a YAML file with four sections: `dataset`, `model`, `symmetry`, and `training`.

The supported datasets are:

1. `advection1d`: exact periodic advection, used for transform-sign checks and the June 10 viability gate.
2. `burgers1d`: viscous periodic Burgers solved by pseudo-spectral Runge--Kutta 4, used for translation and Galilean symmetry experiments.
3. `navier_stokes_vorticity2d`: 2D incompressible Navier--Stokes in vorticity form on a periodic square, used for 2D translation and discrete square-symmetry experiments.

The supported symmetry actions are:

1. `translation1d`: periodic shift, implemented spectrally as `f(x - s)`.
2. `translation2d`: periodic 2D shift, implemented spectrally as `f(x - sx, y - sy)`.
3. `burgers1d_galilean`: for Burgers solution maps `u0 -> uT`, applies
   \[
   u_0'(x)=u_0(x)+c,
   \qquad
   u_T'(x)=u_T(x-cT)+c.
   \]
4. `d4_scalar2d`: rotations by multiples of 90 degrees and optional reflection for ordinary scalar 2D fields.
5. `d4_vorticity2d`: the corresponding pseudoscalar action for vorticity, with a sign flip under reflections. This is the correct action for 2D vorticity under orientation-reversing maps. Both D4 actions are finite-group baselines rather than Lie-tangent objectives.

The supported methods are:

```yaml
training:
  method: baseline    # supervised only
  method: aug         # supervised Lie data augmentation
  method: orbit       # supervised + orbit consistency
  method: aug_orbit   # augmentation + orbit consistency
```

## Minimal commands

Generate data explicitly:

```bash
python scripts/generate_data.py --config configs/pilot/1d_burgers_translation_galilean.yaml
```

Train the main 1D Burgers orbit model:

```bash
python scripts/train.py --config configs/pilot/1d_burgers_translation_galilean.yaml
```

Run a supervised baseline with the same config:

```bash
python scripts/train.py \
  --config configs/pilot/1d_burgers_translation_galilean.yaml \
  --override training.method=baseline \
  --override runtime.run_dir=runs/manual/1d_burgers/baseline
```

Run the June-through-September matrix:

```bash
python scripts/run_matrix.py --matrix configs/experiments_june_september.yaml --dry-run
python scripts/run_matrix.py --matrix configs/experiments_june_september.yaml
```

Collect metrics and draft paper tables:

```bash
python scripts/summarize_results.py --runs runs --out runs/summary.csv
python scripts/make_paper_tables.py --runs runs --out-prefix runs/paper_tables/main
```

Regenerate the frozen N64 Galilean paper artifacts from completed run
directories:

```bash
python scripts/reproduce_paper_artifacts.py
```

The final artifact map, claim summary, and scope decision are documented in
`docs/final_paper_artifacts.md`.

Evaluate a checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint runs/pilot/1d_burgers_translation_galilean/orbit/checkpoints/best.pt \
  --split test \
  --orbit-samples 8
```

## Metrics

Each run writes:

```text
<run_dir>/config.yaml
<run_dir>/meta.json
<run_dir>/train_metrics.jsonl
<run_dir>/val_metrics.jsonl
<run_dir>/test_metrics.json
<run_dir>/checkpoints/best.pt
<run_dir>/checkpoints/last.pt
```

The key metrics are:

1. `relative_l2`: sample-weighted ordinary supervised error on the test split.
2. `orbit_ood_relative_l2`: supervised error after transforming test inputs and labels.
3. `equivariance_defect_relative`: relative norm of
   \[
   \mathcal{G}_\theta(T_g a)-T_g\mathcal{G}_\theta(a).
   \]
4. `latency_ms_per_batch` and `latency_ms_per_sample`: inference timing for the unchanged base model.
5. `best_val_relative_l2`: model-selection metric.

Evaluation also writes standard deviations and standard errors for the main sample-wise metrics when the metric is defined per example.

The primary viability claim requires a lower `equivariance_defect_relative` and lower `orbit_ood_relative_l2` than the supervised baseline at the same inference latency. A method that reduces the equivariance defect but increases ordinary test error needs hyperparameter tuning or adapter-only training before being considered viable.

## Paper-grade protocol

Before any result is considered table-eligible, run:

```bash
make quality
python scripts/make_paper_tables.py --runs runs --out-prefix runs/paper_tables/main
```

For the current N64 Galilean paper result, run:

```bash
python scripts/reproduce_paper_artifacts.py
```

A table-eligible run must contain `config.yaml`, `meta.json`, `rng_state_initial.pt`, `rng_state_final.pt`, `train_metrics.jsonl`, `val_metrics.jsonl`, `test_metrics.json`, and `checkpoints/best.pt`. The metadata records the dataset SHA-256 hash, config hash, seed, parameter count, model name, dataset kind, and environment snapshot.

The main exact-symmetry configs set `model.add_grid: false` because absolute coordinate channels provide the FNO with a fixed origin and can break translation equivariance. The grid-channel condition is isolated in `configs/ablations/grid_ablation.yaml`.

The solver validation script checks both symmetry identities and dt-halving residuals for the pilot PDE solvers:

```bash
python scripts/validate_solvers.py --out runs/quality/solver_validation.json
```

Final paper runs should store this report together with the result tables.

## ICLR 2027 execution roadmap

The current date assumption for this repo is May 28, 2026. The working internal freeze is September 15, with the external ICLR deadline assumed around September 20.

### By June 10: exact-symmetry pilot

Run:

```bash
python scripts/run_matrix.py --matrix configs/experiments_june_september.yaml --dry-run
python scripts/train.py --config configs/pilot/1d_advection_translation.yaml --override training.method=baseline --override runtime.run_dir=runs/june10/1d_advection/baseline
python scripts/train.py --config configs/pilot/1d_advection_translation.yaml --override training.method=orbit --override runtime.run_dir=runs/june10/1d_advection/orbit
```

Decision gate:

```text
Pass: orbit consistency reduces equivariance_defect_relative and orbit_ood_relative_l2 without material inference overhead.
Fail: no reduction in equivariance defect after transform signs and loss scale are verified.
```

Use this stage to debug signs and scaling, not to claim novelty.

### By June 24: Burgers translation + Galilean pilot

Run the 1D Burgers baseline suite:

```bash
python scripts/train.py --config configs/baselines/1d_burgers_baseline.yaml
python scripts/train.py --config configs/baselines/1d_burgers_aug.yaml
python scripts/train.py --config configs/pilot/1d_burgers_translation_galilean.yaml
python scripts/train.py --config configs/baselines/1d_burgers_aug_orbit.yaml
python scripts/train.py --config configs/baselines/1d_burgers_canonical.yaml
```

Decision gate:

```text
Pass: orbit or aug_orbit beats baseline on symmetry-induced OOD error and is competitive with augmentation.
Strong pass: orbit improves over augmentation at low label fractions or enables unlabeled adaptation.
Fail: augmentation dominates all orbit variants after lambda and epsilon sweeps.
```

### By July 15: 2D periodic experiment

Run:

```bash
python scripts/train.py --config configs/pilot/2d_navier_stokes_translation_d4.yaml --override training.method=baseline --override runtime.run_dir=runs/july15/2d_ns/baseline
python scripts/train.py --config configs/pilot/2d_navier_stokes_translation_d4.yaml --override training.method=aug --override runtime.run_dir=runs/july15/2d_ns/aug
python scripts/train.py --config configs/pilot/2d_navier_stokes_translation_d4.yaml --override training.method=orbit --override runtime.run_dir=runs/july15/2d_ns/orbit
python scripts/train.py --config configs/pilot/2d_navier_stokes_translation_d4.yaml --override training.method=aug_orbit --override runtime.run_dir=runs/july15/2d_ns/aug_orbit
```

Decision gate:

```text
Pass: 1D conclusions survive at 2D resolution with a standard FNO2d backbone.
Fail: 2D gains vanish while training cost doubles. Pivot to unlabeled adaptation or label-efficiency framing.
```

### By August 1: ablations

Run:

```bash
python scripts/run_matrix.py --matrix configs/ablations/lambda_sweep.yaml
python scripts/run_matrix.py --matrix configs/ablations/data_fraction_sweep.yaml
```

Required plots:

1. error versus `lambda_orbit`;
2. equivariance defect versus `lambda_orbit`;
3. relative L2 error versus labeled data fraction;
4. orbit OOD error versus transformation magnitude;
5. wall-clock and inference latency table.

### By August 15: approximate and broken-symmetry stress tests

Add masks and partial-domain transformations only after the exact-symmetry story works. The current code has the `output_mask` hook in `BaseTransform`; implement a masked transform class in `src/otno/symmetry/transforms.py` and set `training.method: orbit` with `L_orb^w`.

Candidate stress tests:

1. non-periodic crop after periodic translation;
2. boundary-exclusion mask;
3. transformed coefficient field with untransformed observation window;
4. forced Navier--Stokes where the forcing channel is transformed consistently in one condition and held fixed in another.

### By August 25: theory section freeze

Use `docs/theory.md` as the source of the theory section. The minimal theory package should contain:

1. first-order expansion of local orbit consistency into Lie-algebraic equivariance defect;
2. orbit OOD error decomposition;
3. cost proposition: one transform adds one extra forward pass during training and no inference-time operation;
4. discussion of approximate symmetries via masked norms.

### By September 5: result freeze

Freeze all main tables. Run three seeds for the smallest set of experiments that supports the paper claim. Avoid adding new PDE families after this date.

### By September 15: paper freeze

Freeze the introduction, related work, method, and main results. Final work after this date should be limited to wording, plots, and consistency checks.

## Codex-agent task order

1. Run `make install`, `make test`, and `make smoke`.
2. Verify the transformation identities numerically. In particular, check that the Burgers Galilean action satisfies `solve(T_in u0) ≈ T_out solve(u0)` for small solver tolerance.
3. Run the June 10 advection pilot and inspect `equivariance_defect_relative`.
4. Run the four 1D Burgers baselines.
5. Add seed support to `configs/experiments_june_september.yaml` once one-seed trends are stable.
6. Run the unlabeled adaptation script after the supervised baseline checkpoint exists. Use `configs/pilot/1d_burgers_unlabeled_adapt.yaml` and projector-only updates first.
7. Run the lightweight analytic canonicalization baseline as a computational comparator.
8. Generate summary CSVs and paper-table drafts with `scripts/make_paper_tables.py`.


## Unlabeled adaptation command

After training a supervised or orbit model, adapt it using only target inputs and the orbit-consistency objective:

```bash
python scripts/adapt.py --config configs/pilot/1d_burgers_unlabeled_adapt.yaml
```

Common overrides:

```bash
python scripts/adapt.py \
  --config configs/pilot/1d_burgers_unlabeled_adapt.yaml \
  --override adaptation.checkpoint=runs/june10/1d_burgers/baseline/checkpoints/best.pt \
  --override adaptation.trainable=last_block \
  --override adaptation.epochs=10 \
  --override runtime.run_dir=runs/adaptation/1d_burgers/last_block
```

The script writes `adapt_metrics.jsonl`, `adapt_results.json`, and `adapted.pt`. Labels are used only for pre/post adaptation evaluation.

## Known failure modes

1. Coordinate-grid channels can break exact translation equivariance. This is acceptable for a standard-FNO stress test, but run `model.add_grid=false` diagnostics when debugging pure translation behavior.
2. Normalizing fields with dataset statistics can break Galilean equivariance. The current code leaves normalization out by default.
3. For Galilean Burgers, the output action uses `u_T'(x)=u_T(x-cT)+c`. A sign error here will invalidate the central experiment.
4. Continuous rotations on a square grid introduce interpolation artifacts. The current 2D finite-group baseline uses exact 90-degree rotations and flips.
5. PDE symmetry must preserve the full learning problem, including forcing, coefficients, boundary conditions, and observations.
