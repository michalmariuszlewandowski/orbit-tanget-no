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
runs/paper_tables/2d_galilean_n64_id_ood_fields.json
runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.{png,pdf}
runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.{png,pdf}
runs/figures/2d_galilean_n64_defect_ood_correlation.{png,pdf}
runs/figures/2d_galilean_n64_id_ood_fields.{png,pdf}
```

The qualitative figure is generated from the real held-out N64 fields and the
seed-23 batch-forward-evaluation-matched augmentation and fixed-weight Aug.+LOCO checkpoints.  Its
YAML specification fixes the checkpoints and boost; the adjacent JSON records the
selection rule, selected index, field/error scales, full-test fixed-boost means,
dataset/checkpoint hashes, and plotting-script hash:

```text
figure_specs/2d_galilean_n64_id_ood_seed23.yaml
scripts/plot_navier_stokes_qualitative.py
runs/figures/2d_galilean_n64_id_ood_fields.{png,pdf}
runs/paper_tables/2d_galilean_n64_id_ood_fields.json
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
forward-evaluation-matched supervised augmentation (the backward-pass counts
differ):

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

The low-label curve compares batch-forward-evaluation-matched augmentation against
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

These tested low-label settings support a low-label benefit without adding an
inference-time operation; they do not establish a general sample-complexity law.

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

The rebuilt anonymous artifact is:

```text
dist/local_orbit_consistency_anonymous.zip
```

Its final entry count, byte size, and SHA-256 are recorded in
`rebuttal/README.md` after packaging so this archive input does not contain a
self-invalidating checksum.

After extraction, the DeepONet model/reporting/config tests pass (`34 passed`).
The archive includes the DeepONet implementation, matrix, audited table family,
and their referenced core configs; `rebuttal/` and temporary QA files are excluded.

## Reviewer Follow-Up Controls

Regenerate the cached Reviewer 2/3 control tables, corrected boundary evaluation,
and solver-closure diagnostics with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_reviewer_followup_experiments.ps1
```

The reviewer-facing outputs are:

```text
runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.*
runs/paper_tables/2d_galilean_n64_2pct_tangent_baseline.*
runs/paper_tables/2d_galilean_n64_2pct_plain_finite_baseline.*
runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.*
runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.*
runs/paper_tables/2d_d4_rotation_with_gfno_5seed.*
runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.*
runs/paper_tables/1d_heat_dirichlet_common_mask_evaluation.*
runs/paper_tables/1d_burgers_solver_closure.*
runs/paper_tables/2d_galilean_n64_solver_closure*
```

The completed DeepONet control contains 20 direct runs (four methods by five
seeds), all on dataset SHA-256
`defc9703dbc41494fabf9fcfe9e3408eb2e1c3890814e47fe03d0f6b849e3f50` and the
same 2,688,033-parameter inference graph within the comparison. The predeclared
primary Aug.+LOCO-minus-augmentation results are:

```text
ID relative L2:       -0.06911, 95% CI [-0.08038, -0.05784]  (7.2% reduction)
Orbit OOD relative L2:-0.06799, 95% CI [-0.08087, -0.05510]  (7.0% reduction)
Equivariance defect:  -0.20674, 95% CI [-0.25557, -0.15791] (29.6% reduction)
```

The augmentation and Aug.+LOCO rows each use 1,800 batch-level model evaluations
over 150 epochs. They use 900 and 600 optimizer/backward steps, respectively.
The strict aggregator checks all five seeds, the dataset/config protocol, direct-
run metadata, thread settings, finite metrics, and a single Git commit before it
writes the manifest, aggregate, paired CSV, and LaTeX table.

The supporting Burgers four-method table is retained separately as
`runs/paper_tables/1d_final_four_method.*`; it is not regenerated by the reviewer
wrapper. The D4 controls use the base config
`configs/pilot/2d_navier_stokes_translation_d4.yaml`.

Generated PDE tensors are intentionally excluded from the anonymous ZIP because
they can be recreated from the checked-in base configs. Before running the wrapper
from a fresh archive, generate the Galilean N64, D4 N64, and Burgers N128 tensors:

```bash
python scripts/generate_data.py --config configs/pilot/2d_navier_stokes_galilean.yaml
python scripts/generate_data.py --config configs/pilot/2d_navier_stokes_translation_d4.yaml
python scripts/generate_data.py --config configs/pilot/1d_burgers_translation_galilean.yaml
```

The DeepONet matrix is the direct second-neural-operator control. Its four methods
use an identical fixed-grid branch--trunk inference graph, and its primary
augmentation versus Aug.+LOCO pair is matched at 12 batch-level model evaluations
per epoch but not at optimizer/backward counts. Reproduce the reported values with
uninterrupted direct `run_matrix.py` jobs; the current chunked-resume path is not
RNG- or timing-equivalent for this small-subset schedule.

Interpret the remaining controls narrowly. The normalized rMD17 MLP is a
non-Fourier mechanism check, not a second neural-operator family or a competitive
molecular model. The D4 G-FNO is a hard-equivariant saturation reference and was
not trained with LOCO. The boundary values are approximate-action pseudo-target scores: a mask
restricts the scoring region but does not restore translation equivariance of a
fixed Dirichlet problem. The raw finite-transform coefficient is an approximate
leading-order average-scale match, not an exhaustive weight sweep. Over the five
matched seeds, normalized finite LOCO has 2.7% lower ID error and 2.0% lower
orbit-OOD error than the raw finite control. The normalized-minus-raw paired
differences are -0.01025 (95% CI [-0.01948, -0.00103]) and -0.00813
([-0.01604, -0.00021]), respectively, so both predictive intervals exclude zero
narrowly. The equivariance-defect difference is -0.00103
([-0.00748, 0.00543]) and remains unresolved. Interpret this as a modest
predictive advantage at the fixed radius 0.25 and approximate scale match, not as
evidence that normalization is universally necessary or superior.

At fixed radius 0.25, the recorded `eta` values 1e-8, 1e-6, and 1e-4 give nearly
identical results. Interpret the radius rows cautiously: changing
`symmetry.max_boost` changes both the training-step distribution and the default
transformed-test draw, so this cache is not a pure training-radius ablation.

The rMD17 runs use the preserved processed tensor
`data/rmd17/rmd17_ethanol_force_500.pt` (SHA-256
`fb5515b3355651459c2ad9078e7cfb5bb63dce96e55e7c36956cbe9ff25e5472`).
It is force-included in the anonymous archive despite the general `data/`
exclusion. The historical cache records raw `old_indices`, but the current
`torch.randperm` generator does not recreate the same selection from seed 2027;
the preserved tensor is therefore the authoritative evidence input. With the raw
NPZ present, `python scripts/verify_rmd17_cached_dataset.py` matches all 2,500
cached conformations by `old_indices` with zero float32 input/force discrepancy.

## Reviewer-Facing Claim Map

```text
Claim: lower Galilean OOD error
Evidence: 2% headline table and 2% OOD severity curve.

Claim: lower equivariance defect
Evidence: defect columns in headline, label-efficiency, and severity tables.

Claim: unchanged inference model
Evidence: all headline methods use FNO2d with 2,668,609 parameters; orbit
         consistency is a training objective only.

Claim: portability to a tested second neural-operator family
Evidence: five-seed, four-method DeepONet table and paired CSV. Aug.+LOCO lowers
         OOD error by 7.0% and defect by 29.6%; the smaller predictive reduction
         than FNO is reported as backbone dependence, not uniform gain.

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

## External KdV Published-Reference Calibration

The independent KdV 40s FNO(AR) runs are a secondary calibration, not evidence
for the headline symmetry-OOD claim. Regenerate their scoped comparison with:

```bash
python scripts/make_lpsda_kdv_faithful_comparison.py
```

Authoritative outputs:

```text
runs/paper_tables/lpsda_kdv_faithful_comparison.{csv,tex}
runs/paper_tables/lpsda_kdv_faithful_comparison.runs.csv
runs/paper_tables/lpsda_kdv_faithful_comparison.paired.csv
runs/paper_tables/lpsda_kdv_faithful_comparison.paired_summary.csv
runs/paper_tables/lpsda_kdv_faithful_comparison.interpretation.md
```

The published rows are KdV 40s FNO(AR) values transcribed from Brandstetter et
al., Table 3; they are not local reruns. Local LOCO uses only the Galilean
generator, while published LPSDA uses four generators. Treat the blocks as a
protocol-aligned published-reference comparison, not a controlled
LOCO-versus-LPSDA experiment.
