# Release validation

Audited on 2026-09-05 against the supplied 30-page TMLR rebuttal v11 PDF.
Environment: Windows, Python 3.11.6, PyTorch 2.12.0+cpu, NumPy 2.4.6,
SciPy 1.17.1, pandas 3.0.3, PyYAML 6.0.3, matplotlib 3.10.9,
pytest 9.0.3, Ruff 0.16.6. Test/solver/smoke checks set OMP, MKL,
OpenBLAS, and NumExpr threads to one and disable external pytest plugins.

| Check | Result |
| --- | --- |
| Full source test suite after fixes | **170 passed** |
| Ruff | Passed |
| Config validation and experiment dry run | Passed |
| Clean extracted code/data quality gate | Passed: lint, configs, tests, solver identities, CPU training smoke |
| Fresh-run CNO/workflow regression checks | 49 passed |
| DeepONet/reporting provenance checks | 24 passed |
| Release workflow preflight | 265 steps, 12 suites |
| Dependency lock consistency | `uv lock --check --offline` passed |
| Source distribution and wheel build | Passed |
| Published statistics | 33 rows / 218 mean and sample-standard-deviation values match v11 |
| Archived dataset bytes | All five registry SHA-256 hashes verified |

The extracted quality gate initially exercised 153 tests; final reporting changes
were also checked through their affected suites. Test counts overlap between
rows and should not be added together. Make is unavailable on the audit machine;
the Python commands invoked by the Make targets were used directly.

## Checks against actual scientific evidence

- Re-evaluated the seed-23 N64 FNO Aug.+LOCO checkpoints at orbit weights 0.10
  and 0.30. All 16 and 19 comparable non-timing metrics, respectively, were
  exactly equal to their archived JSON values. At weight 0.30, ID relative L2
  is `0.30216033570468426`, orbit OOD is `0.3246146063320339`, and defect is
  `0.1862196121364832`. This checks two trained models, not every released row.
- Rebuilt the core N64 tables, claim summary, compute accounting, manifest, and
  correlation diagnostic from completed local run records into a separate
  validation directory. Recomputed headline, extended-weight, and label-efficiency
  records still match the PDF references. The strict historical DeepONet table
  builder also succeeded.
- Regenerated representative inputs and solver outputs with exact tensor equality
  on every train/validation/test split: first 64 Burgers samples, first 128 heat
  samples, and first 8 samples for each N64 Navier-Stokes dataset.
- New regression tests establish bitwise uninterrupted/resumed equivalence for
  short and repeated loader cycles, including the restored semi-supervised path.
  Replacing unlabeled targets with NaNs leaves its trained weights identical.
- Solver validation includes Galilean closure: Burgers relative residual
  approximately `1.80e-7`; boosted N64 validation residual approximately `1.52e-6`.

## Limits

No full multi-day retraining campaign or accelerator validation was performed.
Historical chunked runs used different RNG progression and sometimes chunk-local
wall times; fresh training does not promise identical historical weights/timing.
New source/RNG provenance cannot retroactively recover missing historical state.
Some old severity metrics lack config/checkpoint hashes and are checked by their
recorded paths and values rather than a historical cryptographic binding.

Equivalent tensors can serialize to different `.pt` file hashes; use the exact
data archive for historical checks. The rMD17 historical split must be preserved.
See [docs/reproducibility.md](docs/reproducibility.md) for reproduction levels and
optional archived-run export. Source research files and existing runs were preserved;
the concise release is the curated archive and extracted directory under `dist/`.
