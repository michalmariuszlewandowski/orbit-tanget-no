# Response to Reviewer: Experimental Details and Presentation

## Comment 1

> Add a full experimental-details section, including every dataset parameter,
> dataset curation and preparation detail, model hyperparameters, and a detailed
> training setup (batches, learning rate, optimizer, train/test split, etc.).

**Response.** Thank you; we agree that the submitted Empirical Design section
defined the objectives and metrics but did not provide enough implementation detail
for full reproduction. We have expanded it into a dedicated **Experimental Details
and Evaluation Protocol** section and added exhaustive configuration tables
for the headline and auxiliary experiments.

The revision now reports the complete Navier--Stokes dataset-generation protocol:
domain and grid, initial-condition distribution, ambient-frame distribution, split
sizes, solver and time step, viscosity, de-aliasing, forcing, precision, tensor
shapes, preprocessing, seeds, and dataset hash. It gives the exact labeled counts
(10, 20, 51, and 102 at 1%, 2%, 5%, and 10%), subset-selection rule, full FNO
architecture, parameter count, optimizer defaults, learning-rate schedule, batch
sizes (including partial batches), update counts, loss weights, transform sampling,
gradient clipping, checkpoint criterion, evaluation seeds, uncertainty convention,
and latency protocol. We explicitly distinguish the ambient velocity used to
generate an example, the relative training transform, and the symmetry-shifted test
transform; in two dimensions the reported radius is a componentwise bound, not a
Euclidean-ball radius. The new DeepONet control is documented at the same level:
every branch/trunk width, coordinate feature, latent rank, parameter count, fixed
sensor grid, seed, update/forward/backward count, and copied LOCO coefficient is
stated, including the absence of a DeepONet-specific sweep. Equivalent dataset,
model, and optimization details are also provided for Burgers, rMD17 ethanol, the
D4 comparison, and the Dirichlet-boundary diagnostic.

During this audit we also corrected two presentation inconsistencies: the objective
is now written as the implemented unsquared per-example relative L2 supervised loss
with a mean-square orbit numerator, and the compute table now uses the actual
16/4 partial-minibatch cycle at 2% labels. These corrections affect only the
description and transformed-example accounting; none of the reported prediction
metrics or model comparisons changes.

## Comment 2

> The submission is table-heavy and lacks figures. Visualizations of 2D
> Navier--Stokes, highlighting where the proposed approach improves on baselines,
> and ID/OOD error heatmaps or visual artifacts would strengthen the claims.

**Response.** Thank you; we agree that the submitted version did not expose the
spatial behavior underlying the aggregate errors. We have added a full-width
Navier--Stokes qualitative prediction-and-error figure for the same held-out flow
in distribution and under the fixed axis-aligned relative Galilean stress
$\delta=(0.50,0)$, twice the componentwise training radius. The shifted target is
the exact periodic Galilean output action applied to the cached target. The
columns show the target vorticity, direct predictions from forward-evaluation-
matched augmentation and augmentation+LOCO, their pointwise absolute-error maps,
and the paired spatial reduction
`|e_aug| - |e_LOCO|`, with shared field and error scales.

The displayed case is algorithmically selected by a disclosed, outcome-conditioned
median-improvement rule and is illustrative: for training seed 23, we select the
held-out example whose paired fixed-stress relative-L2 improvement is closest to
the median across all 256 test examples. This gives zero-based test index 97. The
caption discloses the seed, stress, ambient frame, sample index, and selection rule,
while a machine-readable sidecar records the dataset and checkpoint hashes, color
limits, selected metrics, and full-test fixed-stress means. In the displayed case,
augmentation+LOCO reduces relative L2 from 0.489 to 0.310 in distribution and from
0.519 to 0.392 under the fixed stress; it has lower absolute error on 59.7% of
shifted pixels. Visually, the larger residuals concentrate along displaced,
high-gradient vorticity structures and are not confined to domain edges, while
the mixed-sign gain map makes clear that improvement is not uniform at every pixel.

Two companion diagnostics make the selection context and equivariance mechanism
auditable. The fixed-stress diagnostic
`runs/figures/2d_galilean_n64_fixed_stress_diagnostics.{png,pdf}` shows the
ID-to-stress error shift, the empirical distribution of paired gains used for
selection, and full-test mean absolute error binned by target-gradient percentile.
For the 256 seed-23 test cases, the mean paired gain is 0.1253, the
10th/50th/90th percentiles are 0.0527/0.1267/0.1932, and 97.7% favor
augmentation+LOCO; index 97 lies at empirical CDF 0.504. Full-test seed-23 mean
relative L2 decreases from 0.4750 to 0.3517 ID and from 0.6815 to 0.5563 under
the fixed stress. Augmentation+LOCO has lower mean absolute error in all ten
target-gradient bins, from 0.3669 versus 0.4725 in the lowest-gradient bin to
0.4906 versus 0.5727 in the highest.

The equivariance-residual diagnostic
`runs/figures/2d_galilean_n64_equivariance_residuals.{png,pdf}` visualizes
$|G(Ta)-T G(a)|$ for both methods and their signed pixelwise reduction, with the
upper target-gradient quartile outlined. For the illustrative index, the relative
equivariance defect decreases from 0.5272 to 0.3868 and 64.9% of pixels have lower
absolute residual. Across the full seed-23 test split under the same fixed stress,
the mean relative defect decreases from 0.7157 to 0.5762.

We also include the label-efficiency curve, the five-seed OOD-severity
curve, and the defect--OOD correlation plot. The qualitative and diagnostic panels
are deliberately presented as the axis-aligned stress $\delta=(0.50,0)$; the
independently randomized five-seed severity evaluation remains the quantitative
evidence for the OOD claim. All new visualizations use the existing held-out data
and reported checkpoints, and their generation is included in the artifact
reproduction command.
