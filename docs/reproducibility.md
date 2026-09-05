# Reproducing TMLR rebuttal v11

The release target is the 30-page supplied PDF, SHA-256
`35fd8a544457b1e784e2cee1a82e5693df7d9d813b2992b7baa5fc534d850f25`.
The research workspace's older manuscript source and earlier artifact notes do
not define the release. In particular, v11 uses weight 0.30 in the stronger
headline, includes CNO and 10% labels, and reports the three-seed, five-method
rMD17 mechanism study in Table 11.

## Choose the level of reproduction

1. **Check published statistics without training.** Run
   `python scripts/verify_paper_results.py`. It recomputes sample means and
   standard deviations from per-seed CSV records for Tables 2, 3, 4, 11, 14,
   15, and 16; checks the exact seed sets, parameter counts where specified,
   and nonmissing config/dataset hashes; and compares with the PDF's printed
   precision. The expected values are in `figure_specs/paper_results_v11.yaml`.
   This does not independently validate the underlying predictions.
2. **Evaluate archived trained models.** Extract the exact-data bundle and
   supply the corresponding archived `checkpoints/best.pt`. Run
   `python scripts/evaluate.py --checkpoint PATH --device cpu --out evaluation.json`.
   Evaluation restores the checkpoint's seed, orbit sample count, model, and
   dataset settings. `--dataset PATH` relocates an input while enforcing its
   saved SHA-256. Timing must be measured again on the target machine.
3. **Train from scratch.** In a fresh checkout with the data bundle extracted,
   run `python scripts/reproduce_release.py --suite main --execute`, or omit
   `--suite main` for all suites. This is a scientific recipe rerun. Some
   historical FNO and CNO jobs used a chunk-resume implementation that changed
   minibatch RNG progression, so fresh training is not a bitwise recreation
   of those historical checkpoints. The current trainer fixes that bug;
   legacy checkpoints without complete RNG snapshots cannot resume.

The code archive includes cached tables and figures, not every trained model.
When checkpoints are absent, report-only and checkpoint-dependent figure stages
fail preflight with their missing inputs. They do not fabricate cached results.

## Experiment map

Every suite is defined once in `configs/release.yaml`. Dependencies are selected
automatically, and duplicate training jobs are collapsed by output directory.

| Suite | v11 result |
| --- | --- |
| `main` | Table 2, Table 4 at 1/2/5%, severity, orbit weights, training compute, and data-based figures |
| `deeponet` | Table 3: five-seed second-operator comparison |
| `controls` | Table 4 at 10%, Table 5 paired comparisons, Table 6 action mismatch |
| `canonicalization` | Tables 8 and 13 observable-frame comparisons |
| `unlabeled` | Table 10 target adaptation and Appendix A semi-supervised result |
| `rmd17` | Table 11: historical three-seed, five-method force mechanism study |
| `objectives` | Table 14 finite/tangent comparison and stabilizer/radius controls |
| `cno2d` | Table 15: five-seed CNO comparison, including unresolved predictive differences |
| `d4` | Table 16: FNO and hard-equivariant D4 reference |
| `burgers` | Table 5 and Appendix A exact-symmetry sanity check |
| `boundary` | Fixed-wall common-support diagnostic |
| `closure` | Table 1: solver closure at training and stress radii |

Conceptual diagrams and asymptotic complexity tables are explanatory manuscript
content, not measured experiment outputs. The registry covers numerical evidence
and data-based figures. Run with `--dry-run` to inspect exact commands; `--preflight`
is read-only; `--execute` runs the selected plan. `--stage reports` regenerates
tables from completed runs and may replace derived CSV/TeX outputs.

## Data and environment

Prefer `local_orbit_consistency_data.zip` for comparison with historical results.
It contains the exact Galilean N64, unboosted N64, Burgers N128, Dirichlet heat,
and processed rMD17 files, with a checksum manifest. All five expected hashes are
recorded in the registry. PDE generation is available through `--stage data` or
`scripts/generate_data.py`; it does not overwrite existing inputs by default.

The current generator matched representative inputs and solver outputs bit for
bit for all train/validation/test splits in this release audit. This sample check
does not establish whole-file hash equality after regeneration: PyTorch file
serialization and numerical kernels can vary across environments. The strict
DeepONet/CNO historical report builders expect the archived Galilean dataset hash.
Do not replace those inputs with a different cache and relabel it as the same run.

The rMD17 split is an immutable evidence input. Its historical `old_indices`
selection differs from the current random-permutation regeneration path. Keep
`data/rmd17/rmd17_ethanol_force_500.pt`; do not regenerate it from the seed alone.
`scripts/verify_rmd17_cached_dataset.py` can compare it against the separately
obtained raw NPZ using the recorded indices and raw-data hash.

Historical inspected runs used Python 3.11.6 and PyTorch 2.12.0+cpu on Windows.
The workflow fixes OMP and MKL thread counts to one, matching the declared
backbone reporting protocol. `uv.lock` freezes installation dependencies;
`VALIDATION.md` records the environment actually tested. CPU/GPU, platform,
thread count, and library changes can affect numerical results and latency.

## Run integrity

New training saves the resolved config, config hash, dataset SHA-256, environment,
source-file hashes, metrics, and selected/final checkpoints. RNG snapshots cover
Python, NumPy, CPU/CUDA PyTorch, and data-loader generators. `best.pt` is selected
on validation error; `last.pt` retains the final model and matching optimizer.
Some old completed runs lack a last checkpoint; it is a resume artifact, not
necessary for evaluating their best model.

The main paper metrics are `relative_l2`, `orbit_ood_relative_l2`,
`equivariance_defect_relative`, `latency_ms_per_sample`,
`best_val_relative_l2`, `parameters`, and `method`. Training wall time is recorded
separately. Historical chunk-local timings cannot be reinterpreted as a complete
training cost. Source hashes added for new runs do not retroactively identify
the uncommitted code used to produce older checkpoints.

## Packaging

The curated builder checks source/config dependency closure, scans the selected
payload for local identities/paths, and writes deterministic ZIP entries plus
SHA-256 manifests. Verify either an archive or its extracted directory with
`python scripts/make_anonymous_submission.py --verify PATH`.
The original research workspace is preserved. `make clean` removes disposable
validation caches and leaves paper tables, data, and experiment runs intact.

To export original run records and checkpoints as a separate large artifact:

```bash
python scripts/make_anonymous_submission.py --evidence-out dist/local_orbit_consistency_evidence.zip
```

Use `--evidence-suite main` to export the main suite and its dependencies only;
add `--evidence-last` if final optimizer/resume checkpoints are also needed.
The exporter preserves original bytes and checksums. The inspected all-suite
inventory contains about 9.2 GB before compression; this optional archive is not
part of the concise code/data package. Extract evidence at the same root as code
and data before running reports or evaluating archived checkpoints.
