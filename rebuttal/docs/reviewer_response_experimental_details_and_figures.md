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
Navier--Stokes qualitative prediction-and-error figure for the same held-out flow in distribution and
under a fixed relative Galilean stress case at twice the componentwise training
radius. The columns show the target vorticity, direct predictions from
forward-evaluation-matched augmentation and augmentation+LOCO, their pointwise absolute-error
maps, and the paired spatial reduction
`|e_aug| - |e_LOCO|`, with shared field and error scales.

The visualization is not hand-picked: using the first reported training seed, we
select the held-out example whose paired fixed-boost improvement is closest to the
median across all 256 test examples. The caption discloses the seed, boost, ambient
frame, sample index, and selection rule, while a machine-readable sidecar records
the dataset and checkpoint hashes, color limits, selected metrics, and full-test
fixed-boost means. In the displayed case, augmentation+LOCO reduces relative L2 from
0.489 to 0.310 in distribution and from 0.519 to 0.392 under the shift; it has lower
absolute error on 59.7% of shifted pixels. The heatmaps show that the main residuals
trace displaced high-gradient vorticity structures rather than domain boundaries,
while the mixed-sign gain map makes clear that improvement is not uniform at every
pixel.

We also include the label-efficiency curve, the five-seed OOD-severity
curve, and the defect--OOD correlation plot. The qualitative panel
is deliberately presented as an axis-aligned stress case; the independently sampled
five-seed severity curve remains the basis for the quantitative OOD claim. All new
visualizations use the existing held-out data and reported checkpoints, and their
generation is included in the artifact reproduction command.
