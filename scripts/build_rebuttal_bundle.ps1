param(
    [string]$Destination = "rebuttal"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$destinationRoot = if ([System.IO.Path]::IsPathRooted($Destination)) {
    [System.IO.Path]::GetFullPath($Destination)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Destination))
}

$requiredFiles = @(
    "paper/main.pdf",
    "paper/main.tex",
    "paper/main.bbl",
    "paper/related_work.bib",
    "paper/related_work_section.tex",
    "paper/method_section.tex",
    "paper/lie_group_actions_diagram.tex",
    "paper/orbit_consistency_diagram.tex",
    "paper/theory_section.tex",
    "paper/experimental_details.tex",
    "paper/results_section.tex",
    "paper/results_tables.tex",
    "paper/discussion_section.tex",
    "paper/new_results.tex",
    "docs/reviewer_response_experimental_details_and_figures.md",
    "docs/reviewer_response_reviewers_2_and_3.md",
    "docs/revision_handoff.md",
    "docs/final_paper_artifacts.md",
    "runs/figures/2d_galilean_n64_id_ood_fields.pdf",
    "runs/figures/2d_galilean_n64_id_ood_fields.png",
    "runs/figures/2d_galilean_n64_fixed_stress_diagnostics.pdf",
    "runs/figures/2d_galilean_n64_fixed_stress_diagnostics.png",
    "runs/figures/2d_galilean_n64_equivariance_residuals.pdf",
    "runs/figures/2d_galilean_n64_equivariance_residuals.png",
    "runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.pdf",
    "runs/figures/2d_galilean_n64_label_efficiency_lambda_0p1.png",
    "runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.pdf",
    "runs/figures/2d_galilean_n64_2pct_ood_severity_lambda_0p1.png",
    "runs/figures/2d_galilean_n64_defect_ood_correlation.pdf",
    "runs/figures/2d_galilean_n64_defect_ood_correlation.png",
    "figure_specs/2d_galilean_n64_id_ood_seed23.yaml",
    "runs/paper_tables/2d_galilean_n64_id_ood_fields.json",
    "runs/paper_tables/2d_galilean_n64_2pct_tangent_baseline.aggregate.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_tangent_baseline.aggregate.tex",
    "runs/paper_tables/2d_galilean_n64_2pct_tangent_baseline.runs.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_plain_finite_baseline.aggregate.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_plain_finite_baseline.aggregate.tex",
    "runs/paper_tables/2d_galilean_n64_2pct_plain_finite_baseline.runs.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.aggregate.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.paired.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.runs.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_finite_objective_comparison.tex",
    "runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.aggregate.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.aggregate.tex",
    "runs/paper_tables/2d_galilean_n64_2pct_eta_epsilon_sensitivity.runs.csv",
    "runs/paper_tables/2d_d4_rotation_with_gfno_5seed.aggregate.csv",
    "runs/paper_tables/2d_d4_rotation_with_gfno_5seed.aggregate.tex",
    "runs/paper_tables/2d_d4_rotation_with_gfno_5seed.runs.csv",
    "runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.aggregate.csv",
    "runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.paired.csv",
    "runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.runs.csv",
    "runs/paper_tables/rmd17_ethanol_500_normalized_loco_5seed.tex",
    "runs/paper_tables/1d_heat_dirichlet_common_mask_evaluation.aggregate.csv",
    "runs/paper_tables/1d_heat_dirichlet_common_mask_evaluation.aggregate.tex",
    "runs/paper_tables/1d_heat_dirichlet_common_mask_evaluation.runs.csv",
    "runs/paper_tables/1d_final_four_method.aggregate.csv",
    "runs/paper_tables/1d_final_four_method.aggregate.tex",
    "runs/paper_tables/1d_final_four_method.runs.csv",
    "runs/paper_tables/1d_burgers_solver_closure.samples.csv",
    "runs/paper_tables/1d_burgers_solver_closure.summary.csv",
    "runs/paper_tables/1d_burgers_solver_closure.tex",
    "runs/paper_tables/2d_galilean_n64_solver_closure.samples.csv",
    "runs/paper_tables/2d_galilean_n64_solver_closure.summary.csv",
    "runs/paper_tables/2d_galilean_n64_solver_closure.tex",
    "runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.runs.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.aggregate.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_headline_lambda_0p1.aggregate.tex",
    "runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.runs.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.aggregate.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.paired.csv",
    "runs/paper_tables/2d_galilean_n64_2pct_deeponet_5seed.tex",
    "configs/experiments_core.yaml",
    "configs/baselines/1d_burgers_aug.yaml",
    "configs/baselines/1d_burgers_aug_orbit.yaml",
    "configs/baselines/1d_burgers_baseline.yaml",
    "configs/baselines/1d_burgers_canonical.yaml",
    "configs/pilot/1d_advection_translation.yaml",
    "configs/pilot/2d_navier_stokes_galilean.yaml",
    "configs/pilot/2d_navier_stokes_galilean_deeponet.yaml",
    "configs/pilot/2d_navier_stokes_translation_d4.yaml",
    "configs/pilot/rmd17_ethanol_force_500.yaml",
    "configs/pilot/1d_heat_dirichlet_nonperiodic.yaml",
    "configs/pilot/1d_burgers_translation_galilean.yaml",
    "configs/ablations/2d_galilean_n64_2pct_lambda_0p1_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_deeponet_5seed.yaml",
    "configs/ablations/2d_galilean_n64_2pct_plain_finite_baseline.yaml",
    "configs/ablations/2d_galilean_n64_2pct_tangent_baseline.yaml",
    "configs/ablations/2d_galilean_n64_2pct_eta_epsilon_sensitivity.yaml",
    "configs/ablations/2d_d4_focused_low_label_pilot.yaml",
    "configs/ablations/2d_d4_focused_lambda_refinement.yaml",
    "configs/ablations/2d_d4_focused_seed_replication.yaml",
    "configs/ablations/2d_d4_gfno_baseline.yaml",
    "configs/ablations/2d_d4_five_seed_completion.yaml",
    "configs/ablations/rmd17_ethanol_500_augmentation_5seed.yaml",
    "configs/ablations/rmd17_ethanol_500_normalized_loco_5seed.yaml",
    "configs/ablations/1d_heat_dirichlet_boundary_mask.yaml",
    "configs/ablations/1d_heat_dirichlet_common_mask_evaluation.yaml",
    "scripts/reproduce_paper_artifacts.py",
    "scripts/run_reviewer_followup_experiments.ps1",
    "scripts/make_paper_tables.py",
    "scripts/make_finite_objective_comparison.py",
    "scripts/make_eta_epsilon_sensitivity_table.py",
    "scripts/make_rmd17_tables.py",
    "scripts/make_solver_closure_tables.py",
    "scripts/make_deeponet_backbone_table.py",
    "scripts/plot_galilean_n64_paper_figures.py",
    "scripts/plot_navier_stokes_qualitative.py",
    "scripts/verify_rmd17_cached_dataset.py",
    "scripts/run_matrix.py",
    "scripts/run_chunked_matrix.py",
    "scripts/make_anonymous_submission.py",
    "scripts/build_rebuttal_bundle.ps1",
    ".submissionignore",
    "src/otno/config.py",
    "src/otno/models/__init__.py",
    "src/otno/models/deeponet.py",
    "tests/test_config_and_metrics.py",
    "tests/test_deeponet_reporting.py",
    "tests/test_fno_shapes.py",
    "dist/local_orbit_consistency_anonymous.zip"
)

$optionalFiles = @()

New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null

foreach ($relativePath in $requiredFiles) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        throw "Required rebuttal file is missing: $relativePath"
    }
    $targetPath = Join-Path $destinationRoot $relativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null
    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
}

foreach ($relativePath in $optionalFiles) {
    $sourcePath = Join-Path $repoRoot $relativePath
    if (Test-Path -LiteralPath $sourcePath -PathType Leaf) {
        $targetPath = Join-Path $destinationRoot $relativePath
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $targetPath) | Out-Null
        Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    }
}

$manifestPath = Join-Path $destinationRoot "MANIFEST.sha256"
$manifestLines = Get-ChildItem -LiteralPath $destinationRoot -Recurse -File |
    Where-Object { $_.FullName -ne $manifestPath } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($destinationRoot.Length).TrimStart("\", "/").Replace("\", "/")
        $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        "$hash  $relative"
    }
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding UTF8

$files = Get-ChildItem -LiteralPath $destinationRoot -Recurse -File
$size = ($files | Measure-Object -Property Length -Sum).Sum
Write-Output ("Rebuttal bundle: {0} files, {1:N2} MiB at {2}" -f $files.Count, ($size / 1MB), $destinationRoot)
