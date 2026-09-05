# Backbone controls: training details and results

Last verified: 2026-09-03

This note is the paper-facing record for the two non-FNO neural-operator
backbone controls on the 2%-label, N64 Galilean Navier--Stokes task. It records
the frozen training protocol, complete five-seed results, paired statistics,
provenance, and the intended manuscript placement.

## Manuscript placement and bottom line

| Backbone control | Placement | Primary Aug.+LOCO versus augmentation result | Conclusion |
|---|---|---|---|
| DeepONet | Main body | ID -7.2%; orbit OOD -7.0%; equivariance defect -29.6%. All three paired 95% intervals exclude zero. | Positive second-neural-operator result: LOCO improves prediction and symmetry metrics without changing the inference graph. |
| CNO2d | Appendix | ID +7.0%; orbit OOD +6.2%; equivariance defect -6.1%. Only the full five-seed defect interval excludes zero. | Mixed backbone-dependence result: LOCO lowers mean equivariance defect, but this experiment does not establish a predictive improvement beyond augmentation. |

The main-body claim should therefore be that the method transfers beyond Fourier
layers to the tested DeepONet, while the effect size is backbone dependent. The
CNO2d result should be presented as a transparent negative/mixed stress test,
not as a second positive replication. There is no cross-backbone significance
test, and the tables below should not be interpreted as one.

The current paper source already imports the DeepONet table in
[`paper/results_tables.tex`](../paper/results_tables.tex). The CNO2d table is not
currently imported.

## Authoritative inputs and outputs

### DeepONet

- Implementation: [`src/otno/models/deeponet.py`](../src/otno/models/deeponet.py)
- Matrix: [`configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml`](../configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml)
- Base config: [`configs/pilot/2d_navier_stokes_galilean_deeponet.yaml`](../configs/pilot/2d_navier_stokes_galilean_deeponet.yaml)
- Reporter: [`scripts/make_deeponet_backbone_table.py`](../scripts/make_deeponet_backbone_table.py)
- Aggregate results: [`runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.aggregate.csv`](../runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.aggregate.csv)
- Paired results: [`runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.paired.csv`](../runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.paired.csv)
- Per-run results and provenance: [`runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.runs.csv`](../runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.runs.csv)
- Generated LaTeX table: [`runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.tex`](../runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.tex)

### CNO2d

- Implementation: [`src/otno/models/cno.py`](../src/otno/models/cno.py)
- Matrix: [`configs/ablations/2d_galilean_n64_2pct_cno2d_5seed.yaml`](../configs/ablations/2d_galilean_n64_2pct_cno2d_5seed.yaml)
- Base config: [`configs/pilot/2d_navier_stokes_galilean_cno2d.yaml`](../configs/pilot/2d_navier_stokes_galilean_cno2d.yaml)
- Reporter: [`scripts/make_cno2d_backbone_table.py`](../scripts/make_cno2d_backbone_table.py)
- Aggregate results: [`runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.aggregate.csv`](../runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.aggregate.csv)
- Paired results: [`runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.paired.csv`](../runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.paired.csv)
- Per-run results and provenance: [`runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.runs.csv`](../runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.runs.csv)
- Generated LaTeX table: [`runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.tex`](../runs/paper_tables/2d_galilean_n64_2pct_cno2d_5seed.tex)
- Completion/resume record: [`runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/PAUSED.md`](../runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/PAUSED.md)
- Frozen CNO2d source snapshot: [`runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/_provenance/source_snapshot.zip`](../runs/ablations/2d_galilean_n64_2pct_cno2d_5seed/_provenance/source_snapshot.zip)

## Shared experimental protocol

### PDE, data, and labeled subsets

The task learns the terminal-vorticity solution operator for the unforced 2D
Navier--Stokes vorticity equation

\[
\partial_t\omega + v\cdot\nabla\omega=\nu\Delta\omega,\qquad
v=(\partial_y\psi,-\partial_x\psi)+c,\qquad \Delta\psi=\omega
\]

on the periodic square \([0,1)^2\).

| Item | Value |
|---|---|
| Grid | \(64\times64\) uniform periodic grid |
| Train / validation / test | 1,024 / 128 / 256 independently generated examples |
| Stored input | \((\omega_0,c_x,c_y)\), shape \(64\times64\times3\); scalar velocity components are broadcast across the grid |
| Stored target | \(\omega(T)\), shape \(64\times64\times1\) |
| Ambient frame | \(c_x,c_y\overset{\mathrm{iid}}{\sim}\mathcal U[-0.5,0.5]\) |
| Final time / time step | \(T=0.5\), \(\Delta t=10^{-3}\), 500 RK4 steps |
| Viscosity / forcing | \(\nu=10^{-3}\) / none |
| Solver | Pseudo-spectral classical RK4 with tensor-product 2/3 de-aliasing; generation batch size 8 |
| Initial vorticity | Gaussian grid noise filtered by \(\exp(-\lVert k\rVert_2^2/8)\), spatially centered and divided by sample spatial SD plus \(10^{-6}\), amplitude 1 |
| Precision and preprocessing | Stored and trained in `float32`; no model-side input/target normalization, clipping, cleaning, or discarded examples |
| Dataset seed | 31, with split offsets 0 / 100,000 / 200,000 |
| Dataset SHA-256 | `defc9703dbc41494fabf9fcfe9e3408eb2e1c3890814e47fe03d0f6b849e3f50` |
| Dataset config fingerprint | `b2b5dec2dd3049ff491071674c34042c89f8c1a855f86cbb199614bfafed0cdd` |

Each training seed retains the first
\(\max(1,\operatorname{round}(1024\times0.02))=20\) indices from a
seeded `torch.randperm`. Thus the methods at a given seed use the same 20 labeled
examples and initialization seed. Validation and test always use their full
splits. The five training seeds are **23, 31, 47, 59, and 71**. Across-seed
variation therefore includes initialization and labeled-subset variation, while
the generated dataset remains fixed.

### Galilean action and objectives

For a relative boost \(\delta=(\delta_x,\delta_y)\), the input action adds
\(\delta\) to the two ambient-velocity channels. The output action is the exact
periodic spectral shift
\(\omega_T(x)\mapsto\omega_T(x-\delta T)\). During training and the standard
transformed evaluation,
\(\delta_x,\delta_y\overset{\mathrm{iid}}{\sim}\mathcal U[-0.25,0.25]\); this
is a componentwise square, not a Euclidean ball.

The supervised loss is the mean unsquared per-example relative \(L^2\) error.
Supervised augmentation applies the same loss to a separately transformed
input/target pair and uses \(\lambda_{\mathrm{aug}}=1\). Normalized LOCO uses

\[
\mathcal L_{\mathrm{orb}}=
\mathbb E_{a,\delta}\left[
\frac{\operatorname{MSE}(G_\theta(T_\delta^{\mathcal A}a)
-T_\delta^{\mathcal U}G_\theta(a))}
{\epsilon^2+\eta}
\right],
\]

where \(\epsilon=\max(\lVert\delta\rVert_2,10^{-6})\) and
\(\eta=10^{-6}\). The MSE is taken over spatial and output entries per example
before the batch mean. Gradients pass through both prediction branches. When
augmentation and LOCO are combined, their transforms are sampled independently,
with independent transform parameters per example.

The four methods are:

- **Baseline:** \(\mathcal L_{\mathrm{sup}}\).
- **Normalized LOCO:** \(\mathcal L_{\mathrm{sup}}+0.05\mathcal L_{\mathrm{orb}}\).
- **Augmentation:** \(\mathcal L_{\mathrm{sup}}+\mathcal L_{\mathrm{aug}}\).
- **Augmentation + normalized LOCO:**
  \(\mathcal L_{\mathrm{sup}}+\mathcal L_{\mathrm{aug}}+0.10\mathcal L_{\mathrm{orb}}\).

DeepONet used the copied FNO coefficients without a DeepONet-specific test-set or
\(\lambda\) sweep. CNO2d used the same fixed four-method protocol.

### Optimization, checkpointing, and compute matching

Both backbones use the following optimization protocol:

| Setting | Value |
|---|---|
| Optimizer | AdamW, \(\beta=(0.9,0.999)\), optimizer \(\epsilon=10^{-8}\), no AMSGrad |
| Epochs | 150; no early stopping |
| Learning rate / weight decay | \(10^{-3}\) / \(10^{-4}\) |
| Schedule | Cosine annealing over all 150 epochs to zero; no warm-up |
| Batch size | 16, with the final 4-example partial batch retained |
| Data loader | Shuffled training loader, zero workers; validation/test not shuffled |
| Gradient clipping | Global norm 1.0 |
| Validation | All 128 examples every 10 epochs |
| Selection | Lowest validation ID mean per-example relative \(L^2\); final metrics use that best checkpoint |
| Numerical mode | `float32`, no automatic mixed precision; deterministic kernels not forced |

The primary comparison is **forward-evaluation matched**, not update- or
backward-pass matched:

| Method | Steps/epoch | Optimizer steps | Model forwards/step | Model forwards/epoch | Model forwards total | Backward passes total | Transformed-example presentations |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 4 | 600 | 1 | 4 | 600 | 600 | 0 |
| Normalized LOCO | 4 | 600 | 2 | 8 | 1,200 | 600 | 6,000 orbit |
| Augmentation | 6 | 900 | 2 | 12 | 1,800 | 900 | 9,000 augmented |
| Augmentation + normalized LOCO | 4 | 600 | 3 | 12 | 1,800 | 600 | 6,000 augmented + 6,000 orbit |

At 20 labels, the loader cycles through batch sizes 16 and 4. Consequently the
example counts above use the actual partial batches. Augmentation and
Aug.+LOCO both perform 12 batch-level model evaluations per epoch and 1,800 over
training, while Aug.+LOCO uses fewer optimizer updates and backward passes.

Matching seeds aligns the labeled subset, initialization seed, and evaluation
draws. It does **not** common-random-number couple the full training trajectories:
the schedules restart the short loader a different number of times, and LOCO
consumes an additional transform draw.

### Evaluation and uncertainty

- **ID relative \(L^2\):** the mean of one relative error per example over all
  256 held-out examples.
- **Orbit-OOD relative \(L^2\):** the mean prediction error after four independent
  Galilean transforms per example, for 1,024 transformed pairs. These are
  symmetry-shifted versions of the held-out examples, not a separately simulated
  OOD split.
- **Relative equivariance defect:**
  \(\lVert G(Ta)-T G(a)\rVert_2/\max(\lVert T G(a)\rVert_2,10^{-8})\), averaged
  over those same 1,024 pairs.
- The standard test transform RNG seed is training seed \(+200{,}000\), so
  evaluation draws are matched by seed across methods.
- Aggregate entries are arithmetic mean \(\pm\) sample standard deviation across
  the five seeds (`ddof=1`).
- Primary paired results use Aug.+LOCO minus augmentation differences and a
  two-sided 95% Student-\(t\) interval with four degrees of freedom. Negative
  deltas are favorable. The intervals are not multiplicity-adjusted.
- Latency uses the first 16-example test batch, five untimed warm-up forwards,
  then 20 timed forwards; elapsed time is divided by \(20\times16\).

## Backbone architectures

### DeepONet

The fixed-grid DeepONet has 2,688,033 trainable parameters, 0.73% more than the
2,668,609-parameter FNO2d headline model.

- The branch sees every N64 input sensor through four circularly padded
  \(3\times3\), stride-2 convolutions with channels
  \(3\to32\to64\to128\to256\), each followed by GELU.
- The resulting \(256\times4\times4=4096\) features pass through a
  \(4096\to480\to256\) projection with GELU after the first linear map.
- The trunk receives 48 fixed periodic coordinate features:
  \(\{\sin(2\pi kx),\cos(2\pi kx),\sin(2\pi ky),\cos(2\pi ky)\}_{k=1}^{12}\).
  Its MLP is \(48\to256\to256\to256\to256\), with GELU after the first three
  maps.
- A rank-256 branch--trunk inner product divided by \(\sqrt{256}\), plus one
  learned scalar bias, produces the output at every query point.
- There is no normalization, spectral convolution, or hard-coded Galilean
  input--output action. Circular padding and periodic query features encode only
  the domain topology.
- The sensor and query grids are fixed at N64; this control does not test
  resolution transfer.

### CNO2d

The CNO2d has 2,667,743 trainable parameters, 866 fewer than the FNO2d
(approximately 0.03%). It is a compact, reference-style periodic
reimplementation, not an invocation of the official external CNO codebase.

- It has four encoder/decoder scales, four residual blocks per encoder scale,
  and three residual blocks at the neck.
- The lifted/encoder channel schedule is \(10,20,40,80,160\), from a channel
  multiplier of 20. Lift/project blocks use 64 latent channels.
- Convolutions are circularly padded \(3\times3\) operations.
- Down/up-sampling uses periodic, bicubic, antialiased resizing with a minimum
  halo of 8 samples. The filtered activation upsamples, applies LeakyReLU with
  slope 0.01, and resamples to the requested scale.
- Batch normalization is disabled so the spatially constant ambient-velocity
  channels retain their sample-specific information.
- The model has no hard-coded Galilean action. This N64 experiment does not test
  resolution transfer.

## Main-body result: DeepONet

The campaign contains 20/20 complete runs: four methods by five seeds, all
uninterrupted direct runs.

### Five-seed aggregate

| Method | ID relative \(L^2\) | Orbit-OOD relative \(L^2\) | Equivariance defect | Best validation relative \(L^2\) | Latency (ms/sample) | Recorded wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| DeepONet | 0.9949 +/- 0.0081 | 1.0013 +/- 0.0072 | 0.5821 +/- 0.0429 | 1.0006 +/- 0.0131 | 5.153 +/- 0.376 | 256.7 +/- 13.0 |
| DeepONet + normalized LOCO | 0.9150 +/- 0.0318 | 0.9236 +/- 0.0300 | 0.5545 +/- 0.0229 | 0.9146 +/- 0.0326 | 5.146 +/- 0.401 | 397.1 +/- 2.4 |
| DeepONet + augmentation | 0.9553 +/- 0.0335 | 0.9652 +/- 0.0308 | 0.6988 +/- 0.0219 | 0.9597 +/- 0.0384 | 4.903 +/- 0.674 | 526.5 +/- 18.1 |
| **DeepONet + augmentation + normalized LOCO** | **0.8862 +/- 0.0362** | **0.8972 +/- 0.0340** | **0.4921 +/- 0.0295** | **0.8887 +/- 0.0354** | 5.903 +/- 1.053 | 575.6 +/- 17.4 |

The prediction and defect columns above reproduce the generated LaTeX table.
Best-validation, latency, and recorded wall-time fields are included here for
audit completeness.

### Primary paired comparison

| Metric | Augmentation mean | Aug.+LOCO mean | Paired delta | 95% CI for delta | Relative reduction | Interval excludes zero? |
|---|---:|---:|---:|---:|---:|---|
| ID relative \(L^2\) | 0.955323 | 0.886213 | -0.069110 | [-0.080383, -0.057836] | 7.23% | Yes |
| Orbit-OOD relative \(L^2\) | 0.965210 | 0.897224 | -0.067986 | [-0.080874, -0.055098] | 7.04% | Yes |
| Equivariance defect | 0.698801 | 0.492062 | -0.206738 | [-0.255566, -0.157910] | 29.58% | Yes |

Every seed favors Aug.+LOCO over augmentation on all three primary metrics. This
supports a main-body claim that normalized LOCO can improve predictive accuracy
and equivariance for a non-Fourier neural operator while leaving its inference
graph unchanged. The predictive reductions are smaller than the corresponding
FNO reductions (7.2% versus 26.6% ID and 7.0% versus 25.1% OOD), whereas the
relative defect reductions are similar (29.6% versus 29.8%). These are
descriptive within-task comparisons, not a statistical comparison of backbone
effect sizes. The tested DeepONet is also fixed-grid and less accurate overall.

Suggested main-body wording:

> On a near-capacity-matched fixed-grid DeepONet, adding normalized LOCO to the
> forward-evaluation-matched augmentation baseline lowers ID error from
> 0.9553 +/- 0.0335 to 0.8862 +/- 0.0362 (7.2%), orbit-OOD error from
> 0.9652 +/- 0.0308 to 0.8972 +/- 0.0340 (7.0%), and equivariance defect from
> 0.6988 +/- 0.0219 to 0.4921 +/- 0.0295 (29.6%). The matched-seed 95%
> intervals for all three Aug.+LOCO-minus-augmentation differences exclude
> zero. This demonstrates portability to the tested non-Fourier operator, while
> the smaller predictive effect than for FNO shows backbone-dependent effect
> size.

## Appendix result: CNO2d

The campaign contains 20/20 complete runs: four methods by five seeds. Eighteen
runs were uninterrupted; two authorized checkpoint continuations are disclosed
below.

### Five-seed aggregate

| Method | ID relative \(L^2\) | Orbit-OOD relative \(L^2\) | Equivariance defect | Best validation relative \(L^2\) | Latency (ms/sample) |
|---|---:|---:|---:|---:|---:|
| CNO2d | 0.3745 +/- 0.0206 | 0.4457 +/- 0.0233 | 0.3734 +/- 0.0213 | 0.3720 +/- 0.0320 | 117.225 +/- 10.978 |
| CNO2d + normalized LOCO | 0.2127 +/- 0.0098 | 0.2391 +/- 0.0062 | 0.1549 +/- 0.0075 | 0.2120 +/- 0.0104 | 121.402 +/- 4.376 |
| **CNO2d + augmentation** | **0.1806 +/- 0.0101** | **0.2030 +/- 0.0112** | 0.1358 +/- 0.0123 | **0.1796 +/- 0.0116** | 120.722 +/- 5.968 |
| CNO2d + augmentation + normalized LOCO | 0.1932 +/- 0.0132 | 0.2156 +/- 0.0109 | **0.1276 +/- 0.0087** | 0.1902 +/- 0.0130 | 130.438 +/- 22.966 |

Augmentation has the lowest CNO2d mean ID and orbit-OOD errors. Aug.+LOCO has
the lowest mean equivariance defect.

### Primary paired comparison

| Metric | Augmentation mean | Aug.+LOCO mean | Paired delta | 95% CI for delta | Relative change | Interval excludes zero? |
|---|---:|---:|---:|---:|---:|---|
| ID relative \(L^2\) | 0.180602 | 0.193159 | +0.012557 | [-0.003033, +0.028148] | 6.95% higher (worse) | No |
| Orbit-OOD relative \(L^2\) | 0.202960 | 0.215589 | +0.012629 | [-0.004054, +0.029312] | 6.22% higher (worse) | No |
| Equivariance defect | 0.135820 | 0.127589 | -0.008231 | [-0.015841, -0.000622] | 6.06% lower (better) | Yes |

The full designated five-seed comparison gives a defect interval that excludes
zero, but **no evidence of improved ID or orbit-OOD prediction over augmentation
under the tested protocol**. Because both predictive intervals cross zero, the
result should not be described as proof that LOCO hurts CNO2d. It instead shows
that reducing equivariance defect does not guarantee a predictive gain for every
backbone and fixed training schedule.

Suggested appendix wording:

> As a further backbone-dependence check, we repeat the four-method N64 protocol
> with a near-capacity-matched periodic CNO2d. Relative to forward-evaluation-
> matched augmentation, augmentation plus normalized LOCO changes ID error from
> 0.1806 +/- 0.0101 to 0.1932 +/- 0.0132 and orbit-OOD error from
> 0.2030 +/- 0.0112 to 0.2156 +/- 0.0109; the matched-seed intervals for both
> predictive deltas include zero. Equivariance defect decreases from
> 0.1358 +/- 0.0123 to 0.1276 +/- 0.0087 (6.1%), with a paired 95% interval
> [-0.01584, -0.00062]. Thus the CNO2d control supports a modest symmetry-defect
> benefit but not a predictive improvement beyond augmentation, reinforcing that
> LOCO's effect size and predictive value depend on the backbone.

### CNO2d resume sensitivity

All five seedwise defect deltas favor Aug.+LOCO. However, the resumed Aug.+LOCO
seed 59 has the largest favorable defect delta and is the only seed for which the
combined method also improves both prediction metrics. A post-hoc sensitivity
calculation restricted to the four uninterrupted primary-pair candidate runs
(seeds 23, 31, 47, and 71) gives:

| Metric | Four-seed paired delta | Post-hoc 95% Student-\(t\) interval |
|---|---:|---:|
| ID relative \(L^2\) | +0.01645 | [-0.00016, +0.03307] |
| Orbit-OOD relative \(L^2\) | +0.01674 | [-0.00128, +0.03476] |
| Equivariance defect | -0.00604 | [-0.01277, +0.00070] |

This post-hoc subset does not replace the designated five-seed analysis, but it
does make the defect inference sensitive to the resumed candidate run. In the
paper, prefer the precise statement **"the full five-seed defect interval excludes
zero"** over calling the effect statistically robust.

### CNO2d checkpoint and timing disclosure

- Baseline seed 23 resumed after epoch 108 from fully saved model, optimizer,
  scheduler, best-validation, config, metadata, and Torch RNG state.
- Aug.+LOCO seed 59 resumed after epoch 103 under the same checkpoint protocol.
- Their post-resume minibatch streams need not exactly reproduce uninterrupted
  runs. Their final predictive metrics are retained, but their recorded wall
  times cover only the post-resume segments and are excluded from uninterrupted
  timing aggregates.
- All other 18 CNO2d jobs, including the final Aug.+LOCO seed 71 job, ran
  uninterrupted through epoch 150 and final evaluation.

For audit only, uninterrupted-run wall-time summaries are:

| Method | Uninterrupted runs | Mean +/- SD (s) | Approx. mean hours |
|---|---:|---:|---:|
| CNO2d | 4 | 5,171.6 +/- 242.5 | 1.44 |
| CNO2d + normalized LOCO | 5 | 9,366.6 +/- 212.5 | 2.60 |
| CNO2d + augmentation | 5 | 13,228.3 +/- 622.6 | 3.67 |
| CNO2d + augmentation + normalized LOCO | 4 | 13,299.2 +/- 183.3 | 3.69 |

These timing values are not a cross-backbone efficiency comparison. DeepONet
used `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`, and its seed shards ran
concurrently; CNO2d used eight-thread CPU execution. The unchanged-inference
claim follows from using an identical graph and parameter count across methods
within a backbone. Host-specific latency measurements are audit fields, not
evidence of an inference-time LOCO module.

## Provenance and reproducibility notes

- Both campaigns used CPU execution on Windows with Python 3.11.6 and
  PyTorch 2.12.0+cpu; CUDA was unavailable. The manifests record platform string
  `Windows-10-10.0.26100-SP0`. The host currently identifies as an 11th Gen Intel
  Core i7-1185G7 at 3.00 GHz, but CPU model, RAM, and peak memory were not captured
  in the original manifests, so this is contextual rather than run-level
  provenance.
- The run manifests record Git commit
  `fed40c8ad88d167a292e7c4ae53bf39dd49b05b5`, per-run resolved configs and
  config hashes, dataset hashes, checkpoints, metrics, and initial/final Torch
  RNG states.
- All DeepONet jobs were uninterrupted direct runs with one-thread OMP/MKL
  settings. They were nevertheless launched from a dirty worktree: the
  DeepONet implementation, configs, reporter, and result tables are untracked,
  and this campaign has no source-snapshot hash. The recorded commit alone does
  not identify the exact DeepONet implementation, so these files must be
  preserved in the final artifact/archive.
- The CNO2d implementation was run from a dirty worktree. Its authoritative
  source is the frozen snapshot with SHA-256
  `73a64bfda1a3e6ba63eab3a372e1cd6b64d22fda76c0405dfd5facaa765fdb33`,
  rather than the recorded Git commit alone. CNO2d used
  `OMP_NUM_THREADS=8` and `MKL_NUM_THREADS=8`.
- The reporting scripts reject missing seeds, protocol drift, non-finite metrics,
  inconsistent provenance, and undeclared checkpoint resumes before producing
  their aggregate, paired, per-run, and LaTeX artifacts.

Safe regeneration from already completed runs is:

```powershell
$env:PYTHONPATH='src'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest tests\test_deeponet_reporting.py tests\test_cno2d_reporting.py -q -p no:ddtrace -p no:cacheprovider
.\.venv\Scripts\python.exe scripts\make_deeponet_backbone_table.py
.\.venv\Scripts\python.exe scripts\make_cno2d_backbone_table.py
```

## Manuscript cautions

1. The generated CNO2d LaTeX currently bolds the entire Aug.+LOCO row. That is
   misleading for the realized results: augmentation has the best ID/OOD means,
   while Aug.+LOCO has the best defect. Correct the emphasis before importing the
   table, or use no boldface.
2. Describe the CNO2d model as a **reference-style periodic reimplementation**,
   not as the official CNO implementation.
3. Do not describe the CNO2d campaign as 20 uninterrupted runs; disclose the two
   checkpoint continuations.
4. Do not compare DeepONet and CNO2d wall times or latency as controlled hardware
   benchmarks because their thread settings and execution schedules differ.
5. Do not claim architecture-independent effect size. The combined evidence
   supports architecture-agnostic applicability of the objective, a positive
   DeepONet replication, and clear backbone dependence of predictive gains.
6. If CNO2d is added to the appendix, update the discussion sentence that says
   further non-Fourier operator families remain future work.
7. Add the Raonic et al. CNO citation to the bibliography when the appendix result
   is inserted; the current bibliography does not contain it.
