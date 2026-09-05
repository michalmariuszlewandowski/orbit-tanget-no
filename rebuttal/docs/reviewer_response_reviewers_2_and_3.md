# Point-by-Point Response to Reviewers 2 and 3

Thank you for the careful and constructive reviews. The comments led us to sharpen
the distinction between LOCO and related symmetry penalties, add controlled
comparators, expand the experimental protocol, narrow the empirical claims, and
make the boundary assumptions explicit.

## Reviewer 2

### Comment 1: Backbone dependence and the architecture-agnostic motivation

> Because LOCO is motivated as architecture-agnostic, please add one or two other
> backbones or discuss whether its benefit depends on the backbone.

**Response.** We agree that "architecture-agnostic" describes where the objective
can be applied; it does not imply that its empirical gain is independent of the
backbone. We now address this directly with a full second-neural-operator
experiment and retain two secondary reference points that expose different forms
of backbone dependence.

The direct experiment reproduces all four headline methods with a fixed-grid
DeepONet: supervision, normalized LOCO, supervised Galilean augmentation, and
augmentation plus normalized LOCO. Its circular CNN branch observes the full N64
input sensor grid, and its periodic-coordinate trunk is combined with the branch
at rank 256. The model has 2,688,033 trainable parameters, only 0.73% more than
the 2,668,609-parameter FNO2d, but has no spectral-convolution layer and no
hard-coded Galilean action.

The DeepONet study uses exactly the same immutable Navier--Stokes dataset
(SHA-256 `defc9703dbc41494fabf9fcfe9e3408eb2e1c3890814e47fe03d0f6b849e3f50`),
the same seed-indexed 20-of-1,024 labeled subsets, 128 validation examples, 256
test examples, seeds {23, 31, 47, 59, 71}, transform distributions, optimizer,
checkpoint rule, and 150-epoch schedules as the FNO comparison. We fixed the
architecture and copied the FNO LOCO coefficients before completing the outcomes;
there was no DeepONet-specific test-set or lambda sweep. Methods share the
initialization seed and labeled subset, and validation/test transform draws are
seed-matched, but their full stochastic training trajectories are not
common-random-number coupled.

Our predeclared primary comparison is DeepONet augmentation versus DeepONet
augmentation plus normalized LOCO. Both use 12 batch-level model evaluations per
epoch (1,800 over training); augmentation uses 900 optimizer updates/backward
passes, while Aug.+LOCO uses 600. Thus this pair is forward-evaluation-matched,
not optimizer-update-, backward-, or wall-clock-matched. Each row uses the same
DeepONet graph and parameter count at inference. The seed shards ran concurrently,
so recorded latency and wall time are retained as audit fields rather than used
for an efficiency claim.

Adding LOCO lowers ID relative $L^2$ from $0.9553\pm0.0335$ to
$0.8862\pm0.0362$, orbit OOD relative $L^2$ from $0.9652\pm0.0308$ to
$0.8972\pm0.0340$, and equivariance defect from $0.6988\pm0.0219$ to
$0.4921\pm0.0295$: reductions of 7.2%, 7.0%, and 29.6%, respectively. The
paired LOCO-minus-augmentation differences are $-0.06911$ for ID error (95% CI
$[-0.08038,-0.05784]$), $-0.06799$ for OOD error
($[-0.08087,-0.05510]$), and $-0.20674$ for defect
($[-0.25557,-0.15791]$). Every seed has the favorable sign on all three metrics,
and all three paired intervals exclude zero.

This answers the portability question positively for the tested second operator,
while also showing dependence on the backbone. The DeepONet predictive-error
reductions are smaller than the FNO reductions on the same task (7.2% versus
26.6% for ID and 7.0% versus 25.1% for OOD), whereas the relative defect
reductions are similar (29.6% versus 29.8%). These effect-size comparisons are
descriptive, not a cross-backbone significance test. DeepONet is also less
accurate overall in this 20-label fixed-grid setting, so we do not present it as
a competitive architecture study or claim resolution transfer.

As a secondary cross-domain mechanism check, we retain the five-seed rMD17
ethanol fixed-molecule MLP experiment. Under its matched batch-forward budget,
normalized LOCO reduces force MAE by 2.7%, transformed-input force MAE by 2.9%,
and force equivariance defect by 64.6%; paired 95% intervals exclude zero for all
three. The disclosed composite rigid-motion step is
$\epsilon=\max(|\theta|+\lVert t\rVert_2,10^{-6})$ with denominator
$\epsilon^2+10^{-3}$ and $\lambda_{\mathrm{orb}}=0.03$. Because that step mixes
angular and translational units, the MLP is not a neural operator, and its
transformed test uses the training action distribution, we label it a mechanism
check rather than a molecular benchmark or OOD result. The exact processed tensor
is preserved in the anonymous artifact with SHA-256
`fb5515b3355651459c2ad9078e7cfb5bb63dce96e55e7c36956cbe9ff25e5472`.

Finally, the supervised D4 G-FNO reference illustrates saturation: its defect is
$7.9\times10^{-8}$, leaving essentially no same-group consistency signal. It is
not a LOCO-trained row and is not parameter- or mode-matched to FNO. We therefore
claim architecture-independent applicability of the loss, not
architecture-independent effect size or uniform gains across arbitrary
backbones.

**Completed artifacts.**

- DeepONet implementation: `src/otno/models/deeponet.py`
- base configuration: `configs/pilot/2d_navier_stokes_galilean_deeponet.yaml`
- full matrix: `configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml`
- validated per-run manifest, aggregate, paired intervals, and LaTeX table:
  `runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.*`
- validation/aggregation script: `scripts/make_deeponet_backbone_table.py`
- DeepONet reporting and model tests: `tests/test_deeponet_reporting.py` and
  `tests/test_fno_shapes.py`
- rMD17 normalized-LOCO aggregate:
  `runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.*`
- hard-equivariant reference: `runs/paper_tables/2d_d4_rotation_with_gfno_5seed.*`
- complete architecture and protocol details: `paper/experimental_details.tex`

### Comment 2: Practical systems that motivate the method

> Please discuss practical systems in the Introduction. An additional real-world
> benchmark is not required.

**Response.** We have added a paragraph to the Introduction that motivates
symmetry-consistent operator learning in concrete systems: global atmospheric
forecasting on spherical fields, vehicle aerodynamics, frame-consistent turbulence
closures, MHD/tokamak plasma surrogates, and molecular force prediction. The new
discussion cites application studies rather than presenting these systems as
experiments performed in this paper.

We also added the main qualification needed for those examples. A transformation is
admissible only when the complete operator family is closed under it: geometry,
state variables, boundary and forcing data, diagnostic coordinates, and vector or
tensor outputs must transform together. Fixed geography, walls, vessel geometry,
or observation coordinates can reduce or break an otherwise plausible symmetry.
The practical examples are therefore motivations, while the empirical claims remain
restricted to the solver-validated benchmarks, the same-benchmark DeepONet
reproduction, and the clearly labelled rMD17 mechanism check.

**Completed manuscript components.**

- application paragraph in `paper/main.tex`
- primary-source citations in `paper/related_work.bib`
- corresponding scope language in `paper/related_work_section.tex`

## Reviewer 3

### Comment 1: Distinction from generic symmetry regularization

> Sharpen the distinction from symmetry regularization. Explain what the finite
> local Lie-group step and step normalization add. A comparable-range unnormalized
> finite-transform baseline would be useful.

**Response.** We agree. We now define the comparison explicitly in the Method section. A
plain finite-transform consistency loss penalizes

\[
  \mathbb E_{a,g}\!\left[\operatorname{MSE}(\Delta_\theta(a,g))\right],
\]

whereas LOCO uses finite elements $g=\exp(\epsilon X)$ and divides the same
numerator by $\epsilon^2+\eta$. Near the identity,

\[
  \Delta_\theta(a,\exp(\epsilon X))
  =\epsilon d_\theta(a;X)+O(\epsilon^2).
\]

Consequently, the plain loss gives larger sampled steps quadratically larger
leading-order weight, and its scale changes mechanically with the transform radius.
The normalized loss removes that leading dependence when
$\epsilon^2\gg\eta$, while retaining the exact finite group action and its
higher-order terms at nonzero steps. We also now state the stabilizer caveat: with
fixed $\eta>0$, taking $\epsilon\to0$ damps the implemented term rather than
producing an exact tangent limit. The infinitesimal interpretation applies when
$\epsilon^2\gg\eta$, or when $\eta=o(\epsilon^2)$; finite-step LOCO is not
described as an unbiased JVP estimator.

We also completed a scale-matched unnormalized N64 control using the same 20
labels, FNO2d, finite Galilean transforms, augmentation branch, four updates per
epoch, and seeds 23/31/47/59/71. We disable normalization and change the coefficient
from 0.10 to the scale-matched value below; all other data and training choices
are fixed. For componentwise
$\delta_i\sim\mathcal U[-0.25,0.25]$,

\[
  \mathbb E\lVert\delta\rVert_2^2
  =2(0.25)^2/3=1/24.
\]

We therefore use $\lambda_{\mathrm{raw}}=2.4$ as an approximate leading-order
average-scale match to normalized $\lambda_{\mathrm{orb}}=0.10$. This is a
principled scale match, not a claim that the two nonlinear losses have identical
realized magnitudes. Because square sampling jointly induces step norm and
direction, the match based on $\mathbb E\lVert\delta\rVert_2^2$ is not an exact
nonlinear equivalence.

The raw finite control obtains ID relative $L^2$ $0.3779\pm0.0421$,
orbit-shift relative $L^2$ $0.4031\pm0.0385$, and equivariance defect
$0.2687\pm0.0219$ over five seeds. On the same seeds, normalized LOCO obtains
$0.3676\pm0.0360$, $0.3950\pm0.0334$, and $0.2677\pm0.0174$. The paired
normalized-minus-raw differences are $-0.01025$ for ID error (95% CI
$[-0.01948,-0.00103]$), $-0.00813$ for orbit-shift error
($[-0.01604,-0.00021]$), and $-0.00103$ for defect
($[-0.00748,0.00543]$). Thus normalization gives modest resolved reductions of
2.7% in ID error and 2.0% in orbit-shift error at this fixed radius and
approximately scale-matched coefficient, while the defect difference remains
unresolved. This single comparison does not establish universal superiority of
normalization over raw finite consistency. Its identified role is to remove the
leading quadratic coupling between objective scale and sampled radius.

**Completed components.**

- `paper/method_section.tex`, paragraph "What locality and step normalization add"
- `paper/related_work_section.tex`, explicit finite/tangent/plain-loss comparison
- `configs/ablations/2d_galilean_n64_2pct_plain_finite_baseline.yaml`
- `runs/paper_tables/2d_galilean_n64_2pct_plain_finite_baseline.*`
- `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.*`
- `scripts/make_finite_objective_comparison.py`

### Comment 2: Tangent propagation and the finite-orbit interpretation

> Clarify the tangent comparison and whether finite steps add an effect beyond a
> tangent penalty.

**Response.** We added an explicit JVP tangent-propagation comparator in the same
2%-label N64 setting. It uses the same 20 labeled examples, FNO2d backbone, four
updates per epoch, five seeds, and coefficient $\lambda_{\mathrm{tangent}}=0.10$.
The input tangent changes only the two ambient-velocity channels; the output tangent
is
$-T(\delta_x\partial_x+\delta_y\partial_y)\mathcal G_\theta(a)$, and the JVP is
computed with graph construction enabled.

The completed results are:

| Method | ID relative $L^2$ | Orbit-shift relative $L^2$ | Eq. defect |
|---|---:|---:|---:|
| Tangent propagation | $0.4902\pm0.0309$ | $0.5349\pm0.0258$ | $0.3292\pm0.0155$ |
| Augmentation + tangent | $0.4150\pm0.0363$ | $0.4490\pm0.0336$ | $0.3157\pm0.0203$ |
| Augmentation + finite LOCO, $\lambda=.10$ | $0.3676\pm0.0360$ | $0.3950\pm0.0334$ | $0.2677\pm0.0174$ |

For finite LOCO minus augmentation plus tangent, the paired five-seed differences
are $-0.04734$ for ID error (95% CI
$[-0.05494,-0.03974]$), $-0.05397$ for orbit-shift error
($[-0.06037,-0.04758]$), and $-0.04799$ for defect
($[-0.05209,-0.04390]$). Thus the tested finite objective achieves lower means
than the tested tangent comparator, with paired intervals excluding zero.

We interpret this result conservatively. Only one tangent coefficient was tested,
and explicit JVP training has a different computational profile; augmentation plus
tangent took $29.1\pm0.9$ observed minutes per seed, compared with approximately
$21.0\pm2.6$ minutes for finite LOCO under the retained CPU runs. The tangent
times were directly logged, whereas the legacy finite-LOCO time is estimated from
checkpoint timestamps; the comparison
is therefore not compute-matched and does not establish that finite transforms are
universally superior to tangent propagation. It is consistent with the finite
objective using information away from the identity, but it does not causally isolate
higher-order orbit information from all optimization and weighting differences. The
plain finite control in Comment 1 provides the complementary normalization check.

**Completed artifacts.**

- `configs/ablations/2d_galilean_n64_2pct_tangent_baseline.yaml`
- `runs/paper_tables/2d_galilean_n64_2pct_tangent_baseline.*`
- `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.*`
- tangent implementation details in `paper/experimental_details.tex`

### Comment 3: Dataset sizes, labeled subsets, and Navier--Stokes generation

> State the absolute train/validation/test sizes, the labeled counts at 1%, 2%,
> 5%, and 10%, the subset-sampling rule, and the Navier--Stokes generation details.

**Response.** We have added a dedicated **Experimental Details and Evaluation
Protocol** section with these quantities stated explicitly.

The boosted N64 dataset contains 1,024 training, 128 validation, and 256 test
examples, generated independently. For a fraction $q$, seed $s$ generates a
`randperm` of the 1,024 training indices and retains the first
$\max(1,\operatorname{round}(1024q))$. The 1%, 2%, 5%, and 10% settings therefore
contain 10, 20, 51, and 102 labeled examples. Methods with the same seed and fraction
use the same subset. Validation and test sets are always used in full; variation
across training seeds includes initialization and labeled-subset variation, while the
underlying generated dataset is fixed.

The data solve the unforced vorticity equation on periodic $[0,1)^2$ at
$64\times64$ resolution from $t=0$ to $T=0.5$. Generation uses a
pseudo-spectral RK4 solver, $\Delta t=10^{-3}$, 500 steps,
$\nu=10^{-3}$, and tensor-product $2/3$ de-aliasing. Initial vorticity is
filtered Gaussian noise with Fourier filter
$\exp(-\lVert k\rVert_2^2/8)$, followed by samplewise zero-mean/unit-SD scaling.
Ambient components are independently uniform on $[-0.5,0.5]$. Inputs contain
$(\omega_0,c_x,c_y)$, targets contain terminal vorticity, tensors are `float32`,
and there is no model-side normalization, exclusion, clipping, or other curation.
The section also records split seeds, shapes, solver batch size, dataset SHA-256,
model hyperparameters, optimizer settings, partial batches, checkpoint rule,
evaluation RNG seeds, uncertainty convention, and latency protocol.

During this audit we also corrected the paper's loss description to match the code:
the supervised/augmentation term is an unsquared per-example relative $L^2$ loss,
whereas the orbit numerator is a mean square. We corrected the transformed-example
accounting for the alternating 16/4 batches in the 20-label setting. These are
description/accounting corrections and do not change any reported prediction metric.

**Completed component.** `paper/experimental_details.tex`

### Comment 4: Calibration of the empirical claims

> Calibrate the broad claims. The strongest evidence is the Galilean
> Navier--Stokes result; Burgers and D4 should be supporting evidence.

**Response.** We agree and have narrowed the abstract, Introduction, Results, and
limitations accordingly. The primary claim is now restricted to the periodic N64
Galilean experiment with the same FNO2d inference map. At 2% labels and five seeds,
the fixed $\lambda_{\mathrm{orb}}=0.10$ analysis changes orbit-shift relative
$L^2$ from $0.5276\pm0.0525$ for the forward-count-matched augmentation baseline
to $0.3950\pm0.0334$, and equivariance defect from
$0.3816\pm0.0322$ to $0.2677\pm0.0174$. The stronger
$\lambda=0.30$ row is identified as the best value in a post-hoc weight sweep,
not as the pre-specified setting used throughout the fixed-weight analyses.

The added DeepONet study is a same-dataset backbone test, not a new equation or a
broader application claim. Its 7.0% OOD-error reduction is smaller than FNO's
25.1%, while its 29.6% defect reduction is close to FNO's 29.8%. We therefore use
it to support portability to the tested second neural-operator family and to make
backbone dependence explicit, not to promote it to a second headline benchmark.

Burgers is now described as an exact-symmetry sanity check. Augmentation is already
near saturation: the mean orbit-shift error changes from 0.00405 to 0.00381, but its
paired interval crosses zero; the defect changes from 0.00265 to 0.00228 and its
paired interval remains below zero, while mean ID error worsens from 0.00251 to
0.00265. D4 is likewise supporting evidence: the FNO
means change only from 0.06538 to 0.06459 for orbit-shift error and from 0.04570 to
0.04499 for defect, while the hard D4-GFNO reference essentially saturates the
defect metric. Neither setting is presented as a second headline result.

We also distinguish an unchanged **within-backbone inference graph** from unchanged
training cost. LOCO adds no inference-time module, model call, or parameter to the
base FNO or DeepONet, but it does add training work. We describe the primary
augmentation comparisons as matched in batch-level forward evaluations, rather
than using an unqualified "compute matched" claim. Correlation between defect and orbit-shift
error is described as diagnostic association, not causal evidence. Action-mismatch
controls are described as inconsistent with generic two-view consistency alone in
this benchmark, not as causal identification of the physical mechanism.

The revised scope statement is:

> The strongest evidence is the solver-validated periodic N64 Galilean experiment.
> The fixed-grid DeepONet is a same-benchmark second-operator check with a smaller
> predictive effect; Burgers and D4 are supporting checks in other symmetry
> settings, and the fixed-molecule MLP is a mechanism-portability check. These do not
> establish uniform gains across arbitrary equations, boundary conditions, groups,
> or architectures.

### Comment 5: Maximum boosts and the distribution of $\epsilon$

> State the maximum-boost and $\epsilon$ distributions explicitly.

**Response.** We have added the complete distributions in the **Experimental
Details and Evaluation Protocol** section. The stored
ambient-frame components are
$c_x,c_y\stackrel{\mathrm{iid}}{\sim}\mathcal U[-0.5,0.5]$. A relative Galilean
transform draws

\[
  \delta_x,\delta_y\stackrel{\mathrm{iid}}{\sim}\mathcal U[-r,r],
  \qquad \epsilon=\max(\lVert\delta\rVert_2,10^{-6}).
\]

Thus "radius" means a componentwise square, not a Euclidean ball, and
$0\leq\lVert\delta\rVert_2\leq\sqrt{2}r$ before clamping. The guard changes every
draw with $\lVert\delta\rVert_2<10^{-6}$, a tiny but nonzero region; $\epsilon$ is
not uniformly distributed. Ordinary training and standard transformed-test
evaluation use $r=0.25$. The exact pre-clamp moment is
$\mathbb E\lVert\delta\rVert_2^2=2r^2/3=1/24$; the implemented
$\mathbb E\epsilon^2$ differs negligibly because of the clamp. The severity experiment uses
$r\in\{0.10,0.20,0.25,0.35,0.50\}$, from below the training bound through twice
that bound. Because the original ambient frame lies in $[-0.5,0.5]^2$, the
transformed observable frame can reach $[-1,1]^2$ at $r=0.50$, although not every
sampled transformed frame lies outside the original support. We therefore define
"symmetry-shifted OOD" operationally rather than implying that every draw is
componentwise outside the training range.

Each reported N64 checkpoint is evaluated on all 256 test examples with four
independent transforms per example, for 1,024 transformed pairs. The standard test
seed is the training seed plus 200,000; the severity seed is the training seed plus
700,000. The fixed qualitative panel uses the disclosed deterministic stress case
$\delta=(0.50,0)$ and is not substituted for the independently sampled five-seed
severity analysis.

**Completed component.** `paper/experimental_details.tex`, especially the opening
definition and "Galilean training and transformed-test distributions."

### Comment 6: Boundary-condition diagnostic

> Highlight the boundary-condition diagnostic.

**Response.** We have added and explicitly delimited a one-dimensional homogeneous
Dirichlet heat-equation diagnostic in the new boundary-condition subsection. It uses 512/128/128
train/validation/test examples, 51 labeled examples, three seeds, shifts
$s\sim\mathcal U[-0.1,0.1]$, zero-padded linear interpolation, and a masked variant
that scores source coordinates in $[0.05,0.95]$.

We discovered during the audit that the original cached summary evaluated each
checkpoint with its own training-mask setting, making the masked and unmasked
orbit metrics incomparable. We therefore re-evaluated all nine checkpoints on the
same transformed examples under both a common full-domain score and a common
valid-overlap score. The corrected means over three seeds are:

| Training | ID $L^2$ | Pseudo-target error, full | Defect, full | Pseudo-target error, overlap | Defect, overlap |
|---|---:|---:|---:|---:|---:|
| Supervised baseline | $0.02704\pm0.00122$ | $0.04509\pm0.00071$ | $0.03408\pm0.00083$ | $0.03325\pm0.00148$ | $0.02427\pm0.00065$ |
| Aug.+LOCO, unmasked losses | $0.03470\pm0.00186$ | $0.04328\pm0.00137$ | $0.02461\pm0.00085$ | $0.03362\pm0.00157$ | $0.01488\pm0.00061$ |
| Aug.+LOCO, masked losses | $0.03625\pm0.00124$ | $0.04469\pm0.00082$ | $0.03005\pm0.00063$ | $0.02303\pm0.00137$ | $0.01174\pm0.00035$ |

The result shows a boundary tradeoff rather than a new positive benchmark. Relative
to unmasked transformed losses, masking improves the chosen valid-overlap
pseudo-target error and defect, but it does not improve full-domain pseudo-target
error and both regularized variants have worse true ID error than the supervised
baseline.

Most importantly, we do not label this as physical OOD accuracy. A fixed Dirichlet
heat problem is not translation-equivariant, boundary influence propagates into the
interior, and the transformed target here is the shifted cached output
$T_sS(a)$, not a fresh solve $S(T_sa)$. The mask restricts where the approximate
constraint is scored; it does not restore exact equivariance. The current mask is
also a source-coordinate overlap mask, not a proof that target-wall effects have
vanished. We therefore use the terms **approximate-action pseudo-target error** and
**valid-overlap consistency**, and keep this diagnostic outside the main Galilean
claim.

**Completed artifacts.**

- `configs/ablations/1d_heat_dirichlet_boundary_mask.yaml`
- `configs/ablations/1d_heat_dirichlet_common_mask_evaluation.yaml`
- `runs/paper_tables/1d_heat_dirichlet_common_mask_evaluation.*`
