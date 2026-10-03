# Code guide

## Runnable advection example

From the repository root, after installing the package, run:

```bash
python scripts/train.py --config configs/examples/advection_quickstart.yaml
```

The [annotated YAML](../configs/examples/advection_quickstart.yaml) explains
every setting in this three-epoch CPU demonstration. It learns the map from
an initial scalar field to the solution of $u_t+c u_x=0$ at time $T=0.25$,
with $c=0.7$, on a periodic unit interval. The target is the Fourier shift
$a(x-cT)$; there is no numerical time step. The small model and dataset
demonstrate the workflow; they do not establish converged accuracy.

Training should produce:

```text
runs/examples/advection/
├── data.pt                     # Generated train/val/test tensors and metadata
└── run/
    ├── config.yaml             # Resolved settings, including CLI overrides
    ├── meta.json               # Config/data hashes, environment, model size
    ├── source_manifest.json    # Source file hashes and repository state
    ├── rng_state_initial.pt    # Initial Python/NumPy/PyTorch RNG streams
    ├── rng_state_final.pt      # Streams after final evaluation
    ├── train_metrics.jsonl     # One loss/optimizer record per epoch
    ├── val_metrics.jsonl       # Validation records (each epoch in this example)
    ├── test_metrics.json       # best.pt test errors, defects, and latency
    └── checkpoints/
        ├── best.pt             # Weights selected by validation relative L2
        └── last.pt             # Final weights, optimizer, scheduler, RNG, results
```

These are written by `generate_advection_1d` in
[`data/generators.py`](../src/otno/data/generators.py) and
`train_from_config` / `_prepare_run_outputs` in
[`training/trainer.py`](../src/otno/training/trainer.py).
`partial_metrics.json` is produced only when `training.stop_after_epochs`
intentionally stops a run early; it is removed on completion. Ordinary training
does not produce figures or CSV reports. A separate evaluation can write
another JSON file:

```bash
python scripts/evaluate.py --checkpoint runs/examples/advection/run/checkpoints/best.pt --device cpu --out runs/examples/advection/evaluation.json
```

The example sets `overwrite: false`, so a second invocation protects its existing
run. To change the model or loss, override `runtime.run_dir` to a fresh path.
If changing grid, solver, or sample-generation settings, also override
`dataset.path` to generate a separate tensor:

```bash
python scripts/train.py --config configs/examples/advection_quickstart.yaml --override dataset.n=64 --override dataset.path=runs/examples/advection_n64/data.pt --override runtime.run_dir=runs/examples/advection_n64/run
```

## From an experiment to a result

Start with a problem configuration such as
[`2d_navier_stokes_galilean.yaml`](../configs/problems/2d_navier_stokes_galilean.yaml).
It specifies the dataset, backbone, symmetry, objective, and output directory.
Experiment matrices under `configs/ablations/` override these settings and
expand independent training seeds. `configs/release.yaml` collects the matrices
into suites and declares their evaluation and report dependencies.

For one run, [`scripts/train.py`](../scripts/train.py) loads YAML and applies
`--override key.path=value` arguments. The resulting configuration is passed to
[`train_from_config`](../src/otno/training/trainer.py). This function validates
the settings, prepares data and loaders, builds the model and objective, runs
optimization, and evaluates the validation-selected checkpoint.

The loss calculation is isolated in
[`training/objectives.py`](../src/otno/training/objectives.py).
`build_training_objective` resolves the method and its weights;
[`compute_batch_loss`](../src/otno/training/objectives.py#L104) evaluates the selected loss terms in a fixed order.
The definitions themselves are in
[`training/losses.py`](../src/otno/training/losses.py). For finite-orbit
consistency, both model branches retain gradients. Normalization divides the
per-example mean-square defect by the squared transformation magnitude plus
`orbit_eta`. The equations and reduction order are detailed below.

[`training/metrics.py`](../src/otno/training/metrics.py) evaluates ID error,
transformed-input error, and equivariance defect with a separate evaluation
seed. [`reporting.py`](../src/otno/reporting.py) checks run protocols and computes
statistics across independent seeds. Experiment-specific table scripts declare
the required settings and metrics before calling those shared calculations.

## Equations and loss implementation

Let $F_\theta$ be the model, $(a_i,u_i)$ an input/target pair, and
$g_i$ a sampled transformation with paired input and output actions
$T^{\rm in}_{g_i}$, $T^{\rm out}_{g_i}$. The default supervised loss is

$$
L_{\rm sup}=\frac1B\sum_i
\frac{\|F_\theta(a_i)-u_i\|_2}{\max(\|u_i\|_2,10^{-8})}.
$$

[`relative_l2_per_sample`](../src/otno/training/losses.py#L20) flattens all non-batch axes, computes one norm ratio
per example, and [`relative_l2_loss`](../src/otno/training/losses.py#L43) averages those ratios. This is an unsquared
relative norm. With `training.loss: trajectory_nmse`,
[`trajectory_nmse_per_sample`](../src/otno/training/losses.py#L53) instead sums squared errors over the spatial axis
of `[B,N,T]`, divides by each time channel's squared target norm (clamped at
`1e-8`), averages time channels, and then averages examples. For tensors with
other ranks, it uses squared per-example relative L2.

Define the finite defect and its masked mean square as

$$
d_i=F_\theta(T^{\rm in}_{g_i}a_i)
       -T^{\rm out}_{g_i}F_\theta(a_i),\qquad
q_i=\frac{\sum_j m_{ij}d_{ij}^2}{\max(\sum_j m_{ij},1)}.
$$

Here $j$ covers spatial points and output channels, and the supported masks
$m$ are binary. A spatial mask is broadcast across output channels before
counting valid entries. Without a mask, $q_i$ is the mean square over all
entries. [`mean_squared_per_sample`](../src/otno/training/losses.py#L98) implements this reduction; an empty mask
gives zero. [`orbit_consistency_loss`](../src/otno/training/losses.py#L118) implements

$$
L_{\rm orbit}=\frac1B\sum_i\frac{q_i}{\epsilon_i^2+\eta},
\quad \eta=\texttt{training.orbit\_eta}.
$$

With `normalize_by_epsilon: false`, it averages $q_i$ directly. The
denominator depends only on the recorded transformation magnitude; it does
not include the target or prediction norm. In contrast, the logged
`orbit_defect_relative` and evaluation `equivariance_defect_relative` use
$\|m_i d_i\|_2/\max(\|m_iT^{\rm out}_{g_i}F_\theta(a_i)\|_2,10^{-8})$.
They are diagnostic relative errors, separate from the optimized objective.

[`augmented_supervised_loss`](../src/otno/training/losses.py#L211) applies the same sample to inputs and labels and
uses the selected supervised loss with `output_mask`. `compute_batch_loss` in
[`training/objectives.py`](../src/otno/training/objectives.py) assembles

$$
L_{\texttt{aug\_orbit}}=L_{\rm sup}
       +\lambda_{\rm aug}L_{\rm aug}
       +\lambda_{\rm orbit}L_{\rm orbit}.
$$

It evaluates supervision, augmentation, finite-orbit consistency, then tangent
propagation, computing only the terms enabled by the method. Augmentation and
orbit terms draw separate samples. A zero weight still evaluates its enabled
term and advances sampling. `semi_aug_orbit` draws an independent input batch
after augmentation; its labels do not enter the orbit calculation.

The reduction order is **entries → per-example normalization → batch mean**.
Training epoch logs average optimizer-step losses equally, so a smaller final
batch has the same log weight as another batch. [`evaluate_model`](../src/otno/training/metrics.py#L60) in
[`training/metrics.py`](../src/otno/training/metrics.py) aggregates individual
examples and transformation draws, including a smaller final batch. Report
statistics then aggregate independent training seeds.

Both finite-orbit predictions retain gradients, including the output action
on the original prediction. Reusing `base_pred` avoids a forward pass without
detaching it. [`tangent_propagation_loss`](../src/otno/training/losses.py#L175) computes
$D F_\theta(a)[\delta a]-\delta u(F_\theta(a))$ using a JVP with
`create_graph=True`, retaining parameter gradients through the derivative and
output tangent; it uses the same $q_i/(\epsilon_i^2+\eta)$ reduction.
Logged scalar statistics detach after the loss is built. Evaluation uses
`torch.no_grad()` and a separate RNG stream.

The implementations are the named functions in
[`training/losses.py`](../src/otno/training/losses.py). Relevant checks include:

- [`test_losses.py`](../tests/test_losses.py):
  [`test_orbit_loss_normalization_only_divides_by_recorded_step_size`](../tests/test_losses.py#L178),
  [`test_masked_orbit_loss_reduces_valid_entries_before_normalizing`](../tests/test_losses.py#L61),
  [`test_relative_l2_reduces_each_example_before_batch_average`](../tests/test_losses.py#L36),
  [`test_trajectory_nmse_reduces_space_then_time_then_batch`](../tests/test_losses.py#L43),
  [`test_orbit_loss_backpropagates_through_both_predictions`](../tests/test_losses.py#L85), and
  [`test_tangent_loss_retains_parameter_gradients`](../tests/test_losses.py#L115).
- [`test_training_objectives.py`](../tests/test_training_objectives.py):
  [`test_method_aliases_preserve_loss_and_sampler_stream`](../tests/test_training_objectives.py#L46) checks objective
  selection and sampling; [`test_reproducibility.py`](../tests/test_reproducibility.py)
  checks exact resume and that semi-supervised orbit updates ignore labels.
- [`test_transforms.py`](../tests/test_transforms.py):
  [`test_nonperiodic_translation1d_mask_excludes_shifted_boundary`](../tests/test_transforms.py#L43) checks common
  support. The fixed-wall diagnostic's scientific limits are in
  [numerical assumptions](numerics.md#boundaries-and-transformations).

### Transformation magnitude epsilon

`epsilon` belongs to each `TransformSample`; it is separate from the sampling
radius and supervised denominator clamp. The samplers in
[`symmetry/transforms.py`](../src/otno/symmetry/transforms.py) use:

| Action / sampler | Recorded `epsilon` |
| --- | --- |
| `Translation1D`, `NonPeriodicTranslation1D` | `max(abs(shift), 1e-6)` |
| `Translation2D` | `max(norm([shift_x, shift_y]), 1e-6)` |
| `Burgers1DGalilean` | `max(abs(boost), 1e-6)` |
| `NavierStokes2DGalilean` | `max(norm([boost_x, boost_y]), 1e-6)` |
| `D4Scalar2D`, `D4Pseudoscalar2D` | `1` for every discrete action, including identity |
| `MolecularRigidMotion` | `max(abs(angle) + norm(translation), 1e-6)` |
| `CompositeTransform` | Magnitude from the selected component action |

Two-dimensional shifts and boosts sample each component uniformly inside the
configured interval: the sampling region is a square, so the norm can exceed
`max_shift` or `max_boost` by a factor of $\sqrt2$. A composite chooses one
action for the whole minibatch with `probabilities`, then samples parameters
per example; it does not compose all listed actions. Magnitudes use the
configured physical units; molecular angle and translation are combined by
the implemented convention. Rescaling domain or coordinate units changes
loss weighting. Manually supplied samples use their supplied `epsilon`
without recomputing it. D4 and molecular actions do not implement tangent
training; tangent mixtures must contain only supported actions.

## Tensor and transformation conventions

Models and datasets use channels in the last dimension. The `a` entry of a
minibatch is the input; `u` is the target.

| Problem | Input shape | Target shape |
| --- | --- | --- |
| Scalar 1D PDE | `[batch, points, 1]` | `[batch, points, 1]` |
| Scalar 2D PDE | `[batch, height, width, 1]` | `[batch, height, width, 1]` |
| Boosted 2D Navier–Stokes | `[batch, height, width, 3]` | `[batch, height, width, 1]` |
| rMD17 force prediction | `[batch, atoms, 4]` | `[batch, atoms, 3]` |

The boosted Navier–Stokes channels are vorticity and two constant ambient-velocity
components. Molecular inputs contain three coordinates and nuclear charge
divided by `dataset.charge_scale`; outputs contain three force components per atom.

[`symmetry/transforms.py`](../src/otno/symmetry/transforms.py) defines paired
input and output actions. A `TransformSample` contains the sampled parameters
and their magnitude `epsilon`. `apply_input` and `apply_output` must use the
same sample. For masked boundary diagnostics, `output_mask` identifies the
valid comparison region. Continuous transformations can also supply
`input_tangent` and `output_tangent` for tangent propagation; discrete D4 actions
do not define those infinitesimal operations.

Periodic translations use Fourier shifts with the convention `f(x - shift)`.
Two-dimensional shift vectors are ordered as `[x, y]`; tensor spatial dimensions
are `[height, width]`. See [numerical assumptions and verified cases](numerics.md)
for boundaries, grids, channel semantics, and supported combinations.

## Configuration

[`config.py`](../src/otno/config.py) handles loading, nested overrides, and
validation. Dataset, model, training, and symmetry checks are separate functions.
Accepted names and aliases are defined in
[`definitions.py`](../src/otno/definitions.py) and shared with the factories.
Aliases are resolved when constructing objects; saved configurations retain
their original values and hashes.

| Section | Main settings |
| --- | --- |
| `dataset` | `kind`, `path`, grid size, split counts, solver parameters, generation seed |
| `model` | `name`, channel counts, width/depth, Fourier modes or backbone-specific settings |
| `symmetry` | Transformation name or mixture, sampling radius, domain length, final time |
| `training` | Method, epochs, minibatch size, learning rate, loss weights, evaluation cadence |
| `runtime` | Device, run directory, overwrite policy, deterministic kernels, resume |

The main method names are `baseline`, `aug`, `orbit`, `aug_orbit`, `tangent`,
`aug_tangent`, and `semi_aug_orbit`. The semi-supervised method draws an
independent input batch for its orbit term; labels from that batch do not enter
the loss. `training.data_fraction` controls labeled data, while
`training.steps_per_epoch` controls optimizer steps. Transformation-control
methods and normalization settings are explicit configuration fields.

The `configs/problems/` directory holds reusable problem and backbone settings;
`configs/baselines/` holds the single-run 1D comparisons. The seed and method
sweeps used in the release are under `configs/ablations/`.

## Checkpoints and figures

`best.pt` contains the weights selected by validation relative L2 error.
`last.pt` retains the final model, optimizer, scheduler, and RNG streams for
continuation. The resolved configuration, data checksum, source manifest,
environment, and metric records are saved beside the checkpoints.
[`scripts/evaluate.py`](../scripts/evaluate.py) evaluates a checkpoint using its
saved configuration and dataset checksum.

Field calculations and rendering are separated under
[`plotting/`](../src/otno/plotting/):

- `field_analysis.py` evaluates fixed-stress predictions and numerical diagnostics.
- `field_panels.py` renders prediction and absolute-error heatmaps.
- `field_diagnostics.py` renders selection context and equivariance residuals.

[`scripts/plot_navier_stokes_qualitative.py`](../scripts/plot_navier_stokes_qualitative.py)
loads the figure specification, validates checkpoints, and connects those
pieces. Its metadata records the selected test case, diagnostic values, source
hashes, and generated files. Frozen paper figure specifications refer to
historical checkpoint bytes; experiment retraining and frozen figure
regeneration are separate workflows. See [Reproduction](reproducibility.md).

## Adaptation and artifact orchestration

`adapt_from_config` in
[`training/adaptation.py`](../src/otno/training/adaptation.py) loads a source
checkpoint and a frozen copy of its model, builds separate orbit/evaluation/
target-input actions, then follows this sequence:

1. `_evaluate_adaptation` measures the source model.
2. `_build_adaptation_state` selects trainable parameters and freezes their
   initial values; `_adaptation_batch_loss` computes
   $L_{\rm orbit}+\gamma L_{\rm preserve}+\beta L_{\rm anchor}$.
   Preservation is per-example squared prediction error divided by the frozen
   source prediction's squared norm (clamped at `1e-8`). Anchoring is the mean
   squared distance across selected parameter elements, using absolute squares
   for complex Fourier parameters.
3. `_adaptation_epochs` runs fixed-epoch optimization and yields log records.
   It consumes `a` only; `u` is used exclusively for before/after evaluation.
   `projector` selects `fc1`/`fc2`; `last_block` adds the final spectral and
   pointwise block; `all` selects all parameters. These partial-selection modes
   are verified for FNO naming.
4. `_evaluate_adaptation` repeats the same seeded evaluation actions; inference
   latency is measured separately. `_summarize_adaptation` calculates deltas,
   and `_write_adaptation_outputs` writes `adapt_results.json` and `adapted.pt`.

The adaptation `random_orbit` control rolls the transformed predictions across
the batch and detaches that reference branch. It differs from the training
`shuffle_output` control, which retains gradients and applies the output mask.
The adaptation random control uses all entries and is verified on periodic
inputs. Before/after labels never select an adaptation epoch. Adapting on the
test split is a transductive protocol, recorded separately from supervised
training. Adaptation does not support resume. Checks are in
[`test_adaptation.py`](../tests/test_adaptation.py), including label independence,
frozen-source gradients, parameter selection, and repeatable saved weights.

[`scripts/reproduce_release.py`](../scripts/reproduce_release.py) assembles
data, training, adaptation, evaluation, and artifact steps with `_plan_datasets`,
`_plan_training`, `_plan_adaptations`, `_plan_evaluations`, and `_plan_artifacts`.
Preflight and execution check the resulting dependencies and outputs.
[`scripts/reproduce_paper_artifacts.py`](../scripts/reproduce_paper_artifacts.py)
separates record collection (`_collect_artifact_tables`), CSV output
(`_write_artifact_tables`), evidence inspection (`_build_artifact_manifest`),
and numerical claim calculations (`_claim_summary`). Missing evidence remains
visible in a diagnostic manifest. Figure evaluation/rendering stays in the
plotting scripts. Dependency, cache, and CSV protocol tests live in
[`test_release_reproduction.py`](../tests/test_release_reproduction.py) and
[`test_csv_reports.py`](../tests/test_csv_reports.py).

## Extending the implementation

A new backbone implements a channels-last `forward` method and is constructed
by `models/build_model`. A new symmetry supplies paired actions and a sampler
and is constructed by `symmetry/build_transform`. Register names in
`definitions.py` and validate any additional configuration fields in the
corresponding domain validator.

For a new objective, keep the mathematical loss in `training/losses.py` and
its method/weight selection in `training/objectives.py`. For a new experiment,
reuse a problem configuration and declare the seed sweep and overrides in a
matrix. Table scripts should state the expected protocol explicitly and reuse
shared statistical calculations.

Run `python scripts/quality_gate.py` after changes. The checks cover formatting,
configuration construction, solver convergence, symmetry identities, training
and resume behavior, reporting, and a small CPU smoke run. The saved statistical
results can be verified separately without training or rewriting artifacts.
