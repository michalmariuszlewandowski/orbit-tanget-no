# Experiment registry

## E0: CPU smoke

Command:

```bash
make smoke
```

Purpose: test installation, data generation, FNO shape logic, orbit loss, checkpoint writing.

## E1: 1D advection translation

Configs:

```text
configs/pilot/1d_advection_translation.yaml
```

Variants:

```text
baseline, aug, orbit, aug_orbit
```

Purpose: exact-symmetry sign and loss-scale diagnostic.

Expected result: orbit consistency reduces equivariance defect. Since the PDE is simple, all methods may have similar supervised error.

## E2: 1D Burgers translation + Galilean

Configs:

```text
configs/baselines/1d_burgers_baseline.yaml
configs/baselines/1d_burgers_aug.yaml
configs/pilot/1d_burgers_translation_galilean.yaml
configs/baselines/1d_burgers_aug_orbit.yaml
```

Purpose: main 1D experiment.

Report:

```text
relative_l2
orbit_ood_relative_l2
equivariance_defect_relative
latency_ms_per_sample
```

## E3: label efficiency

Config matrix:

```text
configs/ablations/label_efficiency_methods.yaml
configs/ablations/data_fraction_sweep.yaml
configs/ablations/label_efficiency_aug_orbit_seeds.yaml
configs/ablations/aug_orbit_lambda_refinement.yaml
```

Purpose: show whether orbit consistency matters most in low-label regimes. The
method matrix is the paper-facing comparison; the data-fraction sweep is the
orbit-only diagnostic. The focused seed-replication matrix compares `aug` and
`aug_orbit` at 5%, 25%, and 100% labels with seeds 23, 31, and 47. The
`aug_orbit` lambda-refinement matrix tests whether lower orbit weights preserve
the symmetry gains while reducing the full-label supervised-error tradeoff.

## E4: lambda sweep

Config matrix:

```text
configs/ablations/lambda_sweep.yaml
```

Purpose: identify stable range for `lambda_orbit`.

## E5: 2D Navier--Stokes vorticity

Config:

```text
configs/pilot/2d_navier_stokes_translation_d4.yaml
configs/pilot/2d_navier_stokes_translation_d4_small.yaml
configs/ablations/2d_small_aug_orbit_lambda.yaml
configs/ablations/2d_full_aug_orbit_lambda_pilot.yaml
```

Purpose: demonstrate that the phenomenon is not restricted to 1D. Use the
small `n=32` diagnostic first to tune the 2D orbit weight before spending on
full `n=64` runs. Long CPU runs can be chunked with
`training.stop_after_epochs` and `runtime.resume`.

## E6: unlabeled adaptation

Status: implemented through `scripts/adapt.py` and `configs/pilot/1d_burgers_unlabeled_adapt.yaml`. Run after E2 creates the supervised baseline checkpoint.

Objective:

\[
\min_{\theta\in\Theta_{\mathrm{adapt}}}
\mathcal{L}_{\mathrm{orb}}^{\mathrm{target}}(\theta)+\beta\|\theta-\theta_0\|_2^2.
\]

Compare full finetuning and last-block finetuning first. Adapter-only updates can be added after the main result is stable.

## E7: approximate symmetry masks

Status: hook exists through `BaseTransform.output_mask`. Implement only after exact symmetry results are stable.
