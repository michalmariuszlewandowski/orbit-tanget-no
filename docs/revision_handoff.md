# Revision Handoff

The submitted PDF was audited directly. It is a 21-page TMLR manuscript with 14
tables and only two figures; it contains no field prediction or spatial-error map,
so the presentation comment is well founded.

The exact TeX source that generated the submitted PDF is not present in this
repository. The nearby `LOCO_submission_revised_edits.tex` download is the closest
located source, but it differs from the PDF and is missing parts of the
submitted appendices. Do not overwrite the revision master with that file or with
the repository's current generic `paper/main.tex`. Export the latest complete TMLR
source bundle from Overleaf before preparing the resubmission.

When the exact source is available:

1. Rename Section 4 from **Empirical Design** to **Experimental Details and
   Evaluation Protocol**.
2. Insert the content of `paper/experimental_details.tex` after the governing
   equations/method section and before the results, adapting section levels to the
   TMLR source.
3. Replace the submitted supervised/augmentation loss equations with the implemented
   equations in `paper/method_section.tex`.
4. Correct the 2%-label transformed-example counts to 6,000 for four-step methods and
   9,000 for the six-step augmentation baseline.
5. Insert `runs/figures/2d_galilean_n64_id_ood_fields.pdf` immediately after the
   headline N64 table and before the label-efficiency subsection. Cite the two
   companion diagnostics
   `runs/figures/2d_galilean_n64_fixed_stress_diagnostics.pdf` and
   `runs/figures/2d_galilean_n64_equivariance_residuals.pdf` from that discussion
   and place them directly afterward or in the appendix figure sequence.
6. Include the label-efficiency, OOD-severity, and defect--OOD correlation plots in
   the main or appendix figure sequence.
7. Copy the point-by-point text from
   `docs/reviewer_response_experimental_details_and_figures.md`, adding final
   section and figure numbers after pagination if desired.
8. Port the Reviewer 2/3 positioning and application paragraphs from
   `paper/main.tex`, `paper/related_work_section.tex`, `paper/method_section.tex`,
   `paper/theory_section.tex`, and `paper/discussion_section.tex`.
9. Insert the full five-seed DeepONet four-method table as the direct second-neural-
   operator response. Lead with the paired augmentation-versus-Aug.+LOCO result,
   then keep the rMD17 MLP as a cross-domain mechanism check and D4 G-FNO as a
   supervised hard-equivariant saturation reference rather than a LOCO-trained
   second operator.
10. Insert the finite-versus-tangent table from
    `runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.*` and
    describe the raw finite coefficient as an approximate leading-order
    average-scale match. In the completed five-seed normalized-versus-raw
    comparison, normalized LOCO has 2.7% lower ID error and 2.0% lower orbit-OOD
    error; the paired intervals exclude zero narrowly. The equivariance-defect
    interval crosses zero. Treat this as a modest predictive advantage at fixed
    radius 0.25 and the approximate scale match, not evidence that normalization
    is universally necessary or superior.
11. Replace the original boundary-mask summary with the common-support
    reevaluation and use the terms "approximate-action pseudo-target error" and
    "valid-overlap consistency." A mask does not restore the broken Dirichlet
    translation symmetry.
12. Place `runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.*` in
    the appendix with its generating config and script. The radius rows change both
    the training-step distribution and the default transformed-test distribution,
    so they are not a pure training-radius ablation.
13. Copy the point-by-point text from
   `docs/reviewer_response_reviewers_2_and_3.md` and verify any descriptive
   section references against the final source.

The qualitative example is seed 23, zero-based test index 97, under the fixed
axis-aligned relative Galilean stress $\delta=(0.50,0)$. Its target is the exact
periodic Galilean output action applied to the cached held-out target. Describe the
case as illustrative and algorithmically selected by the disclosed,
outcome-conditioned median-improvement rule; do not characterize it as an
unconditioned or representative draw.
The selection-context diagnostic shows that 97.7% of the 256 seed-23 paired gains
are positive (mean 0.1253; 10th/50th/90th percentiles
0.0527/0.1267/0.1932), with index 97 at empirical CDF 0.504, and that Aug.+LOCO
has lower mean absolute error in every target-gradient bin. Full-test seed-23 mean
relative L2 is 0.4750 versus 0.3517 ID and 0.6815 versus 0.5563 under fixed
stress. The residual diagnostic shows selected relative equivariance defect 0.5272
to 0.3868, with 64.9% of pixels improved, and full-test seed-23 fixed-stress mean
defect 0.7157 to 0.5762. Keep the independently randomized five-seed severity
evaluation as the quantitative OOD evidence; do not elevate this disclosed
fixed-axis visualization to that role.

The DeepONet campaign is complete: 20 uninterrupted direct runs (four methods by
five seeds). Against its forward-evaluation-matched augmentation baseline,
Aug.+LOCO reduces ID error by 7.2%, orbit OOD error by 7.0%, and defect by 29.6%.
The paired 95% intervals are `[-0.08038, -0.05784]`,
`[-0.08087, -0.05510]`, and `[-0.25557, -0.15791]`. Port the generated table
`runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.tex` and preserve the
smaller DeepONet predictive effect as evidence of backbone dependence.

The final anonymous archive is `dist/local_orbit_consistency_anonymous.zip`
(1,529,019 bytes, 167 entries; SHA-256
`90abc128b7412c8c04646514ef3b64fd973f7ac19d6c6e6f9599816d25c207f2`).
Its extracted DeepONet/config/reporting test subset passes all 34 tests.
This checksum and entry count predate the two new diagnostic figure families.
Add both PNG/PDF pairs to the archive allowlist, rebuild the archive, rerun
preflight, and refresh the recorded size, entry count, and SHA-256 before
submission.

Before submission, the authors should verify the current host CPU/RAM statement.
The cached run manifests did not record those fields. The raw rMD17 NPZ hash has
now been added from the local source file; confirm the final downloaded archive
matches it if the rMD17 appendix remains.

Two archived auxiliary experiments also need an author-side reproducibility check
before they are described as rerunnable from the current tree. The semi-supervised
artifact used the historical `semi_aug_orbit` training path, which the current
trainer no longer recognizes. Restore or archive that implementation before a fresh
rerun. Likewise, the cached rMD17 dataset predates the current split
implementation. The nested raw NPZ path and hash are now recorded, but a fresh
`torch.randperm` does not reconstruct the historical selection. Preserve
`data/rmd17/rmd17_ethanol_force_500.pt` in the artifact as the authoritative input;
its stored raw `old_indices` make the selection auditable against the NPZ.

Reviewer-specific tables are regenerated from completed runs by
`scripts/run_reviewer_followup_experiments.ps1`. The reported DeepONet values use
uninterrupted direct runs; do not substitute the current chunked-resume utility,
which does not preserve this small-subset loader RNG stream or cumulative timing.
The direct matrix runner's `--skip-completed` option skips only jobs that already
contain final test metrics and deliberately refuses a partially written protected
run. Before claiming that the alternate-backbone caches are reproducible, ensure
the final archive includes `src/otno/models/deeponet.py`,
`src/otno/models/molecular.py`, `src/otno/models/gfno.py`, their configs,
aggregation scripts, and associated run manifests/tables. The historical run JSONs
were produced from dirty working trees, so the submitted archive, rather than the
recorded Git commit alone, must contain the exact implementations used for
interpretation.

The archive preflight must also retain
`scripts/make_finite_objective_comparison.py`,
`scripts/make_eta_epsilon_sensitivity_table.py`,
`configs/ablations/2d_galilean_n64_2pct_eta_epsilon_sensitivity.yaml`, and both
`runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.*` and
`runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.*`.
It must also contain
`configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml`,
`scripts/make_deeponet_backbone_table.py`,
`src/otno/models/deeponet.py`, and
`runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.*` for the direct
second-neural-operator result. Retain
`configs/ablations/rmd17_ethanol_500_normalized_loco_5seed.yaml` and
`runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.*` for the normalized
non-Fourier portability claim.
For the remaining reviewer evidence, retain
`configs/pilot/2d_navier_stokes_translation_d4.yaml`,
`scripts/make_solver_closure_tables.py`, the Galilean N64 and Burgers base configs,
the two `runs/paper_tables/*solver_closure*` output families, the qualitative
figure specification and plotting script, and all six paper-facing figure
families, including
`runs/figures/2d_galilean_n64_fixed_stress_diagnostics.{png,pdf}` and
`runs/figures/2d_galilean_n64_equivariance_residuals.{png,pdf}`. Generated PDE
tensors may be omitted because their checked-in configs recreate them; the
preserved rMD17 tensor is the documented exception.
