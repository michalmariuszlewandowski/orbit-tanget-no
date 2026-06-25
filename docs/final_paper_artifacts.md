# Final N64 Galilean Artifact Map

This file records the current paper-facing evidence for the narrow claim:

```text
A neural operator trained with local orbit consistency has lower symmetry-induced
OOD error and lower equivariance defect than a supervised FNO baseline, with
unchanged inference architecture and parameter count.
```

The main positive result is the N64 boosted-Galilean 2D Navier--Stokes setting
with low labels. The headline method is `aug_orbit` with `lambda_orbit=0.10`.

## Cached Reproduction Command

Regenerate the final cached tables, figures, manifest, and claim summary from
completed local run directories:

```bash
python scripts/reproduce_paper_artifacts.py
```

Outputs:

```text
runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.*
runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.*
runs/paper_tables/2d_galilean_n64_2pct_lambda_robustness.*
runs/paper_tables/2d_galilean_n64_2pct_lambda_extended.*
runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.*
runs/paper_tables/2d_galilean_n64_training_compute.*
runs/paper_tables/2d_galilean_n64_defect_ood_correlation.*
runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization.*
runs/paper_tables/2d_galilean_n64_2pct_oracle_canonicalization_summary.*
runs/paper_tables/2d_galilean_n64_final_manifest.csv
runs/paper_tables/2d_galilean_n64_claim_summary.csv
runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.{png,pdf}
runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.{png,pdf}
runs/figures/2d_galilean_n64_defect_ood_correlation.{png,pdf}
```

`2d_galilean_n64_final_manifest.csv` checks every final training run for:

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

It also checks that every final severity evaluation has its
`severity_metrics.json` and source checkpoint.

## Main Table

Use:

```text
runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.aggregate.csv
```

At 2% labels over five seeds, `aug_orbit` with `lambda_orbit=0.10` improves over
compute-matched supervised augmentation:

```text
orbit_ood_relative_l2:         0.3950 vs 0.5276  (25.1% reduction)
equivariance_defect_relative:  0.2677 vs 0.3816  (29.8% reduction)
parameters:                    identical, 2,668,609
```

The same method improves over the plain supervised FNO baseline by 58.7% on
orbit OOD error and 61.7% on equivariance defect.

## Label Efficiency

Use:

```text
runs/paper_tables/2d_galilean_n64_label_efficiency_lambda_0p1.aggregate.csv
runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.{png,pdf}
```

The low-label curve compares compute-matched augmentation against
`aug_orbit(lambda_orbit=0.10)` at 1%, 2%, and 5% labels.

OOD-error reductions:

```text
1% labels: 13.7%
2% labels: 25.1%
5% labels: 22.3%
```

Equivariance-defect reductions:

```text
1% labels: 21.3%
2% labels: 29.8%
5% labels: 33.2%
```

This supports the label-efficiency version of the claim without adding an
inference-time operation.

## OOD Severity

Use:

```text
runs/paper_tables/2d_galilean_n64_2pct_ood_severity_lambda_0p1.aggregate.csv
runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.{png,pdf}
```

At 2% labels over five seeds, the method keeps an advantage from in-radius
boosts through 2x the training boost radius:

```text
max boost 0.10: OOD -26.4%, defect -33.6%
max boost 0.20: OOD -25.8%, defect -31.3%
max boost 0.25: OOD -25.3%, defect -30.2%
max boost 0.35: OOD -23.6%, defect -27.8%
max boost 0.50: OOD -20.3%, defect -23.6%
```

This is the strongest direct evidence for lower symmetry-induced OOD error.

## Ablations

Use:

```text
runs/paper_tables/2d_galilean_n64_2pct_lambda_extended.aggregate.csv
```

The completed high-weight lambda sweep supports using `lambda_orbit=0.10` as
the broader-experiment setting while documenting that `lambda_orbit=0.30` is
the strongest tested value:

```text
lambda 0.10: OOD 0.3950, defect 0.2677
lambda 0.20: OOD 0.3564, defect 0.2268
lambda 0.30: OOD 0.3416, defect 0.2044
```

Earlier pilot branches and secondary variants are intentionally excluded from
the anonymous ZIP because they are not part of the final submitted result path.

## Reviewer-Facing Claim Map

```text
Claim: lower Galilean OOD error
Evidence: 2% headline table and 2% OOD severity curve.

Claim: lower equivariance defect
Evidence: defect columns in headline, label-efficiency, and severity tables.

Claim: unchanged inference model
Evidence: all headline methods use FNO2d with 2,668,609 parameters; orbit
         consistency is a training objective only.

Claim: label efficiency
Evidence: 1%, 2%, and 5% label-efficiency table and figure.

Claim: robustness to orbit weight
Evidence: 2% lambda robustness table, with lambda 0.10 improving over 0.05 and
         0.01 in the completed runs.
```

## Scope Decision

Do not add a 3D PDE for the main submission. A same-resolution N64 3D analogue
would multiply raw field size by 64 and likely multiply per-seed training time
by roughly 50-150x after batch-size reduction. It would also create a new
solver, transform, storage, and baseline-validation burden. The current result
is already enough for the narrow claim if the paper is framed around exact
periodic 1D validation plus the N64 2D Galilean low-label result.
