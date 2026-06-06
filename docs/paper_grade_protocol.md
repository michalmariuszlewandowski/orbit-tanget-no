# Paper-grade protocol

This repository is table-eligible only after the local quality gate passes and the relevant runs include full reproducibility artifacts.

## Local gate

Run from the repository root:

```bash
make quality
```

The gate performs config validation, experiment-matrix expansion, unit tests, numerical symmetry checks, solver dt-halving checks, and a CPU smoke training run. The solver report is written to `runs/quality/solver_validation.json`.

## Required files per completed training run

A completed run must contain:

```text
config.yaml
meta.json
rng_state_initial.pt
rng_state_final.pt
train_metrics.jsonl
val_metrics.jsonl
test_metrics.json
checkpoints/best.pt
checkpoints/last.pt
```

`meta.json` records the dataset path, dataset SHA-256 hash, config hash, seed, model name, dataset kind, parameter counts, and environment information.

## Table generation

After runs finish, generate paper-table drafts with:

```bash
python scripts/make_paper_tables.py --runs runs --out-prefix runs/paper_tables/main
```

The command writes raw run rows, grouped aggregates, and a LaTeX draft table.

For the frozen N64 Galilean paper artifacts, use the cached reproducer instead:

```bash
python scripts/reproduce_paper_artifacts.py
```

This regenerates the final lambda=0.10 tables, figures, reproducibility
manifest, and claim-summary CSV from completed local run directories. The
artifact map is maintained in `docs/final_paper_artifacts.md`.

## Exact-symmetry runs

Exact translation and Galilean experiments use `model.add_grid: false` in the main configs. Absolute coordinate channels are evaluated separately in `configs/ablations/grid_ablation.yaml` because they give the model an origin and can increase equivariance defect.

## Solver validation

The pseudo-spectral solvers are pilot solvers. Before final tables, store dt-halving and symmetry reports for every PDE family used in the paper. Increase resolution or decrease `dt` when the dt-halving residual is not comfortably below the reported model error scale.
