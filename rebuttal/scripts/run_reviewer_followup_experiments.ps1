param(
    [int]$WaitForPid = 0,
    [double]$MinFreeGb = 10.0
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

function Get-FreeGb {
    $drive = [System.IO.DriveInfo]::GetDrives() | Where-Object { $_.Name -eq "C:\" }
    if ($null -eq $drive) {
        return 0.0
    }
    return [math]::Round($drive.AvailableFreeSpace / 1GB, 2)
}

function Assert-DiskHeadroom {
    param([string]$Phase)
    $freeGb = Get-FreeGb
    Write-Host ("[{0}] free disk: {1} GB" -f $Phase, $freeGb)
    if ($freeGb -lt $MinFreeGb) {
        throw ("Stopping before {0}: only {1} GB free, below MinFreeGb={2}" -f $Phase, $freeGb, $MinFreeGb)
    }
}

function Count-Metrics {
    param([string]$Root)
    if (-not (Test-Path $Root)) {
        return 0
    }
    return @(Get-ChildItem $Root -Recurse -Filter test_metrics.json -ErrorAction SilentlyContinue).Count
}

function Invoke-Python {
    param(
        [string]$Phase,
        [string[]]$Arguments
    )
    Assert-DiskHeadroom $Phase
    Write-Host ("[{0}] {1} {2}" -f $Phase, $Python, ($Arguments -join " "))
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ("{0} failed with exit code {1}" -f $Phase, $LASTEXITCODE)
    }
}

if ($WaitForPid -gt 0) {
    $existing = Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue
    if ($null -ne $existing) {
        Write-Host ("[wait] waiting for PID {0}" -f $WaitForPid)
        Wait-Process -Id $WaitForPid
    } else {
        Write-Host ("[wait] PID {0} is not running" -f $WaitForPid)
    }
}

$tangentRoot = "runs\ablations\2d_galilean_n64_2pct_tangent_baseline\fraction_0.02"
$tangentCount = Count-Metrics $tangentRoot
Write-Host ("[tangent] completed metrics: {0}/10" -f $tangentCount)
if ($tangentCount -lt 10) {
    throw ("Tangent baseline is incomplete after wait: found {0}/10 metrics" -f $tangentCount)
}

Invoke-Python "table:tangent" @(
    "scripts\make_paper_tables.py",
    "--runs", $tangentRoot,
    "--out-prefix", "runs\paper_tables\2d_galilean_n64_2pct_tangent_baseline",
    "--group-cols", "method,data_fraction,steps_per_epoch,config.training.lambda_tangent"
)

$plainFiniteRoot = "runs\ablations\2d_galilean_n64_2pct_plain_finite_baseline"
$plainFiniteCount = Count-Metrics $plainFiniteRoot
Write-Host ("[plain-finite] completed metrics before launch: {0}/5" -f $plainFiniteCount)
if ($plainFiniteCount -lt 5) {
    Invoke-Python "matrix:plain-finite" @(
        "scripts\run_matrix.py",
        "--matrix", "configs\ablations\2d_galilean_n64_2pct_plain_finite_baseline.yaml"
    )
}

Invoke-Python "table:plain-finite" @(
    "scripts\make_paper_tables.py",
    "--runs", $plainFiniteRoot,
    "--out-prefix", "runs\paper_tables\2d_galilean_n64_2pct_plain_finite_baseline",
    "--group-cols", "method,data_fraction,steps_per_epoch,config.training.lambda_orbit,config.training.normalize_by_epsilon"
)

Invoke-Python "table:finite-objective-comparison" @(
    "scripts\make_finite_objective_comparison.py"
)

$sensitivityRoot = "runs\ablations\2d_galilean_n64_2pct_eta_epsilon_sensitivity"
$sensitivityCount = Count-Metrics $sensitivityRoot
Write-Host ("[sensitivity] completed metrics before launch: {0}/15" -f $sensitivityCount)
if ($sensitivityCount -lt 15) {
    Invoke-Python "matrix:sensitivity" @(
        "scripts\run_matrix.py",
        "--matrix", "configs\ablations\2d_galilean_n64_2pct_eta_epsilon_sensitivity.yaml"
    )
}

Invoke-Python "table:sensitivity" @(
    "scripts\make_eta_epsilon_sensitivity_table.py"
)

$d4Required = @(
    "runs\ablations\2d_d4_focused_low_label_pilot\fraction_0.01\aug_steps_4\seed_23\test_metrics.json",
    "runs\ablations\2d_d4_focused_lambda_refinement\fraction_0.01\aug_orbit_lambda_0.2_steps_4\seed_23\test_metrics.json",
    "runs\ablations\2d_d4_focused_seed_replication\fraction_0.01\aug_steps_4\seed_31\test_metrics.json",
    "runs\ablations\2d_d4_focused_seed_replication\fraction_0.01\aug_steps_4\seed_47\test_metrics.json",
    "runs\ablations\2d_d4_focused_seed_replication\fraction_0.01\aug_orbit_lambda_0.2_steps_4\seed_31\test_metrics.json",
    "runs\ablations\2d_d4_focused_seed_replication\fraction_0.01\aug_orbit_lambda_0.2_steps_4\seed_47\test_metrics.json",
    "runs\ablations\2d_d4_five_seed_completion\fraction_0.01\aug_steps_4\seed_59\test_metrics.json",
    "runs\ablations\2d_d4_five_seed_completion\fraction_0.01\aug_steps_4\seed_71\test_metrics.json",
    "runs\ablations\2d_d4_five_seed_completion\fraction_0.01\aug_orbit_lambda_0.2_steps_4\seed_59\test_metrics.json",
    "runs\ablations\2d_d4_five_seed_completion\fraction_0.01\aug_orbit_lambda_0.2_steps_4\seed_71\test_metrics.json",
    "runs\ablations\2d_d4_gfno_baseline\fraction_0.01\d4_gfno2d_steps_4\seed_23\test_metrics.json",
    "runs\ablations\2d_d4_gfno_baseline\fraction_0.01\d4_gfno2d_steps_4\seed_31\test_metrics.json",
    "runs\ablations\2d_d4_gfno_baseline\fraction_0.01\d4_gfno2d_steps_4\seed_47\test_metrics.json",
    "runs\ablations\2d_d4_five_seed_completion\fraction_0.01\d4_gfno2d_steps_4\seed_59\test_metrics.json",
    "runs\ablations\2d_d4_five_seed_completion\fraction_0.01\d4_gfno2d_steps_4\seed_71\test_metrics.json"
)
foreach ($path in $d4Required) {
    if (-not (Test-Path $path)) {
        throw ("Missing D4 metric: {0}" -f $path)
    }
}

Invoke-Python "table:d4" @(
    "scripts\make_paper_tables.py",
    "--runs", "runs",
    "--out-prefix", "runs\paper_tables\2d_d4_rotation_with_gfno_5seed",
    "--group-cols", "method,model_name,data_fraction,steps_per_epoch,lambda_orbit",
    "--include-run-dir", "runs/(ablations/2d_d4_five_seed_completion/fraction_0\.01/(aug_steps_4|aug_orbit_lambda_0\.2_steps_4|d4_gfno2d_steps_4)/seed_(59|71)|ablations/2d_d4_focused_low_label_pilot/fraction_0\.01/aug_steps_4/seed_23|ablations/2d_d4_focused_lambda_refinement/fraction_0\.01/aug_orbit_lambda_0\.2_steps_4/seed_23|ablations/2d_d4_focused_seed_replication/fraction_0\.01/(aug_steps_4|aug_orbit_lambda_0\.2_steps_4)/seed_(31|47)|ablations/2d_d4_gfno_baseline/fraction_0\.01/d4_gfno2d_steps_4/seed_(23|31|47))$"
)

$rmdRoot = "runs\appendix\rmd17_ethanol\forward_matched_500_5seed"
$rmdAugRoot = "$rmdRoot\aug_steps_24"
$rmdAugCount = Count-Metrics $rmdAugRoot
Write-Host ("[rmd17-augmentation-baseline] completed metrics before launch: {0}/5" -f $rmdAugCount)
if ($rmdAugCount -lt 5) {
    Invoke-Python "matrix:rmd17-augmentation-baseline" @(
        "scripts\run_matrix.py",
        "--matrix", "configs\ablations\rmd17_ethanol_500_augmentation_5seed.yaml"
    )
}

$rmdNormalizedRoot = "runs\appendix\rmd17_ethanol\normalized_loco_500_5seed"
$rmdNormalizedCount = Count-Metrics $rmdNormalizedRoot
Write-Host ("[rmd17-normalized-loco] completed metrics before launch: {0}/5" -f $rmdNormalizedCount)
if ($rmdNormalizedCount -lt 5) {
    Invoke-Python "matrix:rmd17-normalized-loco" @(
        "scripts\run_matrix.py",
        "--matrix", "configs\ablations\rmd17_ethanol_500_normalized_loco_5seed.yaml"
    )
}

Invoke-Python "table:rmd17-normalized-loco" @(
    "scripts\make_rmd17_tables.py",
    "--runs", "runs",
    "--run-root", "runs/appendix/rmd17_ethanol",
    "--include-run-dir", "runs/appendix/rmd17_ethanol/(forward_matched_500_5seed/aug_steps_24|normalized_loco_500_5seed/aug_loco_lambda_0\.03_steps_16)/seed_(23|31|47|59|71)",
    "--expected-seeds", "23,31,47,59,71",
    "--out-prefix", "runs\paper_tables\rmd17_ethanol_500_normalized_loco_5seed",
    "--caption", "rMD17 ethanol force prediction at 500 labeled conformations under matched batch-level forward-evaluation budgets. The LOCO row uses the normalized finite objective with a disclosed composite rigid-motion step; transformed evaluation uses the training action distribution.",
    "--label", "tab:rmd17-ethanol-500-normalized-loco"
)

$boundaryRoot = "runs\ablations\1d_heat_dirichlet_boundary_mask"
$boundaryCount = Count-Metrics $boundaryRoot
Write-Host ("[boundary] completed metrics before launch: {0}/9" -f $boundaryCount)
if ($boundaryCount -lt 9) {
    Invoke-Python "matrix:boundary" @(
        "scripts\run_matrix.py",
        "--matrix", "configs\ablations\1d_heat_dirichlet_boundary_mask.yaml"
    )
}

Invoke-Python "table:boundary" @(
    "scripts\make_paper_tables.py",
    "--runs", $boundaryRoot,
    "--out-prefix", "runs\paper_tables\1d_heat_dirichlet_boundary_mask",
    "--group-cols", "method,data_fraction,config.symmetry.use_mask"
)

Invoke-Python "evaluate:boundary-common-support" @(
    "scripts\run_ood_severity_matrix.py",
    "--matrix", "configs\ablations\1d_heat_dirichlet_common_mask_evaluation.yaml",
    "--out-prefix", "runs\paper_tables\1d_heat_dirichlet_common_mask_evaluation"
)

Invoke-Python "table:navier-stokes-solver-closure" @(
    "scripts\make_solver_closure_tables.py",
    "--config", "configs\pilot\2d_navier_stokes_galilean.yaml",
    "--dataset-path", "data\2d_ns_vorticity_galilean_n64.pt",
    "--num-samples", "32",
    "--seed", "2027",
    "--max-boosts", "0.25,0.35,0.50",
    "--out-prefix", "runs\paper_tables\2d_galilean_n64_solver_closure"
)

Invoke-Python "table:burgers-solver-closure" @(
    "scripts\make_solver_closure_tables.py",
    "--config", "configs\pilot\1d_burgers_translation_galilean.yaml",
    "--dataset-path", "data\1d_burgers_n128.pt",
    "--num-samples", "32",
    "--seed", "2027",
    "--max-boosts", "0.35,0.525,0.70,1.05",
    "--out-prefix", "runs\paper_tables\1d_burgers_solver_closure"
)

$deepONetRoot = "runs\ablations\2d_galilean_n64_2pct_deeponet_5seed"
$deepONetCount = Count-Metrics $deepONetRoot
Write-Host ("[deeponet] completed metrics before launch: {0}/20" -f $deepONetCount)
if ($deepONetCount -lt 20) {
    $env:OMP_NUM_THREADS = "1"
    $env:MKL_NUM_THREADS = "1"
    Invoke-Python "matrix:deeponet" @(
        "scripts\run_matrix.py",
        "--matrix", "configs\ablations\2d_galilean_n64_2pct_deeponet_5seed.yaml",
        "--skip-completed"
    )
}

Invoke-Python "table:deeponet" @(
    "scripts\make_deeponet_backbone_table.py"
)

Write-Host "[done] reviewer follow-up experiments complete"
