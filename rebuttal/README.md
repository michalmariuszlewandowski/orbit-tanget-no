# Rebuttal workspace

This folder is the compact handoff for the TMLR revision. It contains copies of
the working manuscript, reviewer responses, paper-facing figures and audited
tables, the configurations and scripts used for the reviewer controls, and the
anonymous reproducibility archive.

Important source note: `paper/main.pdf` is the verified working integration
build in this repository. The exact TeX source that produced the originally
submitted TMLR PDF was not available in the workspace. Port these changes into
a fresh export of the true submission master before resubmitting, and compare
the rendered result against that master.

## DeepONet reviewer experiment

The added DeepONet study is a full four-method reproduction on the same N64
Galilean dataset, 2% labeled subsets (20 of 1,024 cases), and five seeds. The
predeclared primary comparison is supervised Galilean augmentation versus the
same DeepONet with augmentation plus normalized LOCO. Those two rows use the
same 12 batch-level operator evaluations per epoch, while their optimizer and
backward-pass counts differ and are reported explicitly.

Across seeds 23, 31, 47, 59, and 71, Aug.+LOCO reduces mean ID relative L2 by
7.2%, orbit OOD relative L2 by 7.0%, and equivariance defect by 29.6% relative
to augmentation. The paired 95% intervals are `[-0.08038, -0.05784]`,
`[-0.08087, -0.05510]`, and `[-0.25557, -0.15791]`, respectively; all exclude
zero. The manuscript treats the smaller DeepONet predictive effect relative to
FNO as evidence of backbone dependence, not uniform architecture-independent
gain.

These are exactly the five training seeds used by every method in the primary
FNO comparison. The DeepONet aggregation script reads the FNO per-run manifest
and refuses to produce a table if any method has a missing, extra, or duplicate
seed, or if the DeepONet and FNO seed sets differ.

The architecture, complete matrix, aggregation script, per-run manifest,
aggregate statistics, paired confidence intervals, and LaTeX table are copied
into this bundle. Checkpoints and generated PDE data are intentionally omitted;
the recorded dataset SHA-256 identifies the immutable dataset used by every
run.

From the repository root, the complete matrix and audited summaries are produced
with:

```powershell
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
.\.venv\Scripts\python.exe scripts/run_matrix.py --matrix configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml
.\.venv\Scripts\python.exe scripts/make_deeponet_backbone_table.py
```

`run_matrix.py` also accepts repeatable `--seed` and `--method` filters for
parallel direct-run shards; the canonical matrix remains the source of every
override. The reported numbers come from uninterrupted direct runs. The current
chunked-resume utility is not a bitwise-equivalent reproduction path for this
small-subset schedule and does not accumulate wall time across chunks. The
`--skip-completed` option can fill missing jobs in a partial direct-run matrix;
it does not resume a partially written job.

## Layout

- `paper/`: working manuscript source components and compiled PDF.
- `docs/`: point-by-point reviewer responses and revision handoff.
- `runs/figures/`: paper figures in PDF and PNG.
- `runs/paper_tables/`: per-run manifests, aggregate statistics, paired tests,
  and LaTeX tables.
- `configs/`, `scripts/`, `src/`, `tests/`: readable reproduction materials for
  the reviewer additions.
- `dist/local_orbit_consistency_anonymous.zip`: complete anonymous code/data
  artifact selected by the repository packaging audit.
- `MANIFEST.sha256`: SHA-256 for every copied file in this folder.

Regenerate the copies and checksum manifest from the repository root with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_rebuttal_bundle.ps1
```

The bundle builder only creates directories and copies the declared files. It
does not delete the destination or move the working originals.

The included anonymous ZIP has 167 entries and is 1,529,019 bytes. Its SHA-256
is `90abc128b7412c8c04646514ef3b64fd973f7ac19d6c6e6f9599816d25c207f2`;
the extracted DeepONet/config/reporting test subset passes all 34 tests.
