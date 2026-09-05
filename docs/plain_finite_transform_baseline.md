# Plain Finite-Transform Equivariance Baseline

Status: complete on 2026-09-04 for matched seeds 23, 31, 47, 59, and 71.

## Question

This control isolates the step-size normalization in LOCO from the use of finite
Lie-group elements.  Define the finite equivariance defect

\[
\Delta_\theta(a,\delta)
=G_\theta(T_\delta^{\mathcal A}a)
-T_\delta^{\mathcal U}G_\theta(a).
\]

The raw baseline minimizes

\[
\mathcal L_{\rm raw}
=\mathbb E_{a,\delta}\!\left[\operatorname{MSE}
\bigl(\Delta_\theta(a,\delta)\bigr)\right],
\]

whereas normalized finite LOCO divides the same per-example numerator by
\(\epsilon^2+\eta\), with \(\epsilon=\max(\lVert\delta\rVert_2,10^{-6})\).

## Matched protocol

| Item | Setting |
|---|---|
| Dataset | Periodic N64 2D Navier--Stokes; SHA-256 `defc9703dbc41494fabf9fcfe9e3408eb2e1c3890814e47fe03d0f6b849e3f50` |
| Labels | 2% of 1,024 training examples = 20 labels |
| Backbone | FNO2d, width 48, 12x12 modes, depth 4, no coordinate grid; 2,668,609 parameters |
| Seeds | 23, 31, 47, 59, 71 |
| Transform | Componentwise Galilean boost `delta_i ~ Uniform[-0.25, 0.25]` |
| Training | 150 epochs, batch size 16, 4 updates/epoch, AdamW, learning rate 0.001, weight decay 0.0001 |
| Supervision | Both methods retain the same supervised augmentation branch with `lambda_aug=1` |
| Compute accounting | 12 batch-level model-forward evaluations per epoch for both finite objectives |
| Evaluation | Best-validation checkpoint; 256 test cases; four orbit draws per case |

The only objective changes are `normalize_by_epsilon: true -> false` and the
coefficient described below.  The raw and normalized logs have identical
`train_epsilon_mean` and `train_epsilon_max` at every epoch for all five seeds,
confirming matched transform draws.

## Coefficient match

For componentwise square sampling with radius \(r=0.25\),

\[
\mathbb E\lVert\delta\rVert_2^2
=\frac{2r^2}{3}=\frac{1}{24}.
\]

Therefore, normalized \(\lambda_{\rm orb}=0.10\) is compared with

\[
\lambda_{\rm raw}=\frac{0.10}{1/24}=2.4.
\]

This is an approximate leading-order average-scale match, not a tuned optimum or
an exact nonlinear equivalence.  Square sampling couples direction and norm, and
finite defects need not scale exactly quadratically at the sampled radius.

## Five-seed results

Entries are mean plus/minus sample standard deviation.

| Objective | ID relative L2 | Orbit-OOD relative L2 | Equivariance defect |
|---|---:|---:|---:|
| Augmentation + raw finite, `lambda_raw=2.4` | 0.3779 +/- 0.0421 | 0.4031 +/- 0.0385 | 0.2687 +/- 0.0219 |
| Augmentation + normalized finite, `lambda_orbit=0.10` | **0.3676 +/- 0.0360** | **0.3950 +/- 0.0334** | **0.2677 +/- 0.0174** |

Paired differences are normalized finite minus raw finite.  Confidence intervals
are two-sided 95% Student-t intervals over the five matched seeds.

| Metric | Paired difference | 95% CI | Relative change |
|---|---:|---:|---:|
| ID relative L2 | -0.01025 | [-0.01948, -0.00103] | -2.71% |
| Orbit-OOD relative L2 | -0.00813 | [-0.01604, -0.00021] | -2.02% |
| Equivariance defect | -0.00103 | [-0.00748, 0.00543] | -0.38% |

Per-seed values:

| Seed | Raw ID | Normalized ID | Raw OOD | Normalized OOD | Raw defect | Normalized defect |
|---:|---:|---:|---:|---:|---:|---:|
| 23 | 0.3559 | 0.3517 | 0.3775 | 0.3744 | 0.2463 | 0.2531 |
| 31 | 0.3910 | 0.3740 | 0.4166 | 0.4028 | 0.2738 | 0.2693 |
| 47 | 0.3922 | 0.3764 | 0.4102 | 0.3974 | 0.2694 | 0.2666 |
| 59 | 0.3195 | 0.3189 | 0.3554 | 0.3559 | 0.2520 | 0.2534 |
| 71 | 0.4309 | 0.4172 | 0.4558 | 0.4444 | 0.3020 | 0.2959 |

## Interpretation

The raw finite penalty already recovers most of the gain over compute-matched
augmentation, showing that finite equivariance consistency itself is the main
ingredient.  At the tested radius and approximate scale match, normalization adds
a modest predictive improvement: 2.7% for ID error and 2.0% for orbit-OOD error,
with paired intervals below zero.  The defect difference is unresolved.

This does not show that normalization is universally more accurate, nor does a
single average-scale coefficient establish equivalence between the objectives.
The structural contribution of the normalization is clearer: it removes the
leading quadratic coupling between sampled step magnitude and regularizer
strength, so the coefficient is less mechanically tied to the chosen local radius.
The separate tangent/JVP control addresses finite endpoint evaluations versus an
identity-tangent penalty.

## Artifacts and validation

- Matrix: `configs/ablations/2d_galilean_n64_2pct_plain_finite_baseline.yaml`
- Runs: `runs/ablations/2d_galilean_n64_2pct_plain_finite_baseline/`
- Combined LaTeX table: `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.tex`
- Aggregate CSV: `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.aggregate.csv`
- Paired CSV: `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.paired.csv`
- Run manifest: `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.runs.csv`

Every seed contains the resolved config, dataset and config hashes, initial and
final RNG snapshots, training and validation streams, best and last checkpoints,
and the complete paper-facing `test_metrics.json` fields.  The repository quality
gate passed before launch.  A focused test also verifies numerically that the raw
branch returns the finite per-example MSE and that the normalized branch divides
the same values by the recorded \(\epsilon^2+\eta\).

Seeds 59 and 71 were run concurrently on CPU.  Their logged wall times therefore
include resource contention and should not be used for a training-time comparison;
the forward-evaluation budgets and predictive metrics remain protocol-matched.

## Integration note for the attached v6 manuscript

The attached v6 source defines the raw finite loss but does not yet report this
five-seed comparison.  The updated repository text is in
`paper/results_section.tex`, the table is in `paper/results_tables.tex`, and the
point-by-point response is in `docs/reviewer_response_reviewers_2_and_3.md`.

Before porting those blocks, also correct the displayed supervised and augmented
losses in the attached v6 source: they are written as squared relative L2 losses,
whereas training uses the batch mean of the unsquared per-example relative L2.
The finite raw and normalized consistency numerators are per-example MSEs.  The
implemented equations are already reflected in `paper/method_section.tex`.
