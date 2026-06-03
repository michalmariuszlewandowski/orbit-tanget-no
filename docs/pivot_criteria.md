# Pivot criteria

## Continue with the main paper if

1. Orbit consistency reduces `equivariance_defect_relative` by at least 30% versus supervised FNO on the 1D Burgers experiment.
2. Orbit consistency reduces `orbit_ood_relative_l2` versus supervised FNO.
3. Inference latency is statistically unchanged because no canonicalizer or extra inference pass is used.
4. Augmentation plus orbit consistency is at least competitive with augmentation alone, or orbit consistency wins in low-label or unlabeled-adaptation settings.

## Pivot to unlabeled adaptation if

1. Orbit consistency reliably lowers equivariance defect;
2. supervised augmentation beats orbit training on labeled data;
3. target-domain unlabeled inputs are available or can be simulated by symmetry-induced shifts.

Paper framing after this pivot:

```text
Local orbit consistency is a post-training adaptation objective for pretrained neural operators under known PDE symmetries.
```

## Pivot to diagnostic/negative result only if

1. The transform identity tests pass;
2. lambda and epsilon sweeps are stable;
3. coordinate-grid and normalization effects have been tested;
4. orbit consistency still does not reduce equivariance defect.

A negative result is unlikely to be competitive for ICLR unless it reveals a clear failure mode of widely used FNO backbones.

## Scope cuts

Cut the following first:

1. 3D PDEs;
2. arbitrary manifolds;
3. full gauge-equivariant models;
4. external benchmark integration beyond simple loaders;
5. canonicalization baselines beyond translation/Galilean toy baselines.

## Required sanity checks before any pivot

1. `solve_burgers_1d(T_in u0) ≈ T_out solve_burgers_1d(u0)` for the Galilean action.
2. `periodic_shift_1d(periodic_shift_1d(x, s), -s) ≈ x`.
3. Orbit loss is near zero for an identity model under matching input/output translation actions.
4. `add_grid=false` diagnostic has been run for pure translation.
