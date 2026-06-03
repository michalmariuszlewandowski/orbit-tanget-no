# Instructions for coding agents

This repo is intended to test a narrow research hypothesis quickly: stochastic local orbit consistency for neural operators. Keep the implementation minimal, measurable, and aligned with the ICLR 2027 timeline.

## Primary objective

Produce evidence for or against the following claim:

```text
A neural operator trained with local orbit consistency has lower symmetry-induced OOD error and lower equivariance defect than a supervised FNO baseline, with unchanged inference cost.
```

The strongest version of the claim includes label efficiency or unlabeled adaptation.

## Safe edit order

1. Run `make quality` before changing experiment code; run `make test` and `make smoke` after small patches.
2. Fix transform or solver bugs before adding new models.
3. Validate symmetry identities before running long training jobs.
4. Add experiments by adding YAML configs, not by hard-coding variants in scripts.
5. Keep every run reproducible: seed, config, checkpoint, metrics JSON, dataset SHA-256 hash, config hash, and RNG snapshots.
6. Log wall-clock and latency for every reported model.

## Do not expand scope prematurely

Avoid adding arbitrary manifolds, gauge-equivariant layers, 3D PDEs, or large external datasets until the 1D and 2D periodic experiments support the claim. The deadline favors a narrow positive result over a broad incomplete system.

## Tests that matter most

The critical tests are:

```bash
make quality
pytest tests/test_transforms.py -q
pytest tests/test_symmetry_identities.py -q
make smoke
```

If the Galilean identity test fails, fix that before training. A model cannot learn the intended equivariance if the coded group action is wrong.

## Paper-facing metrics

Every final experiment needs these fields in `test_metrics.json`:

```text
relative_l2
orbit_ood_relative_l2
equivariance_defect_relative
latency_ms_per_sample
best_val_relative_l2
parameters
method
```

## Target baselines

Minimum table:

```text
FNO
FNO + supervised Lie augmentation
FNO + local orbit consistency
FNO + augmentation + local orbit consistency
```

Optional table:

```text
translation/Galilean canonicalization baseline
G-FNO for discrete Euclidean symmetries, if integrated as an external baseline
adapter-only unlabeled orbit adaptation
```

## Pivot rules

Use `docs/pivot_criteria.md`. The main pivot is toward unlabeled adaptation if joint orbit training is weaker than augmentation but still reduces equivariance defect.
