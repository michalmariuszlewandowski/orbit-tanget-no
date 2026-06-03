$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$env:PYTHONPATH = "src"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"

$Python = Join-Path $Root ".venv\Scripts\python.exe"

function Invoke-PythonStep {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Output "[$(Get-Date -Format o)] START $Name"
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Output "[$(Get-Date -Format o)] DONE $Name"
}

Invoke-PythonStep "1D final four-method training" @(
    "scripts\run_chunked_matrix.py",
    "--matrix", "configs\ablations\1d_final_four_method_seeds.yaml",
    "--chunk-epochs", "25",
    "--max-chunks-per-job", "20"
)

Invoke-PythonStep "1D final four-method table" @(
    "scripts\make_paper_tables.py",
    "--runs", "runs\final_1d\four_method",
    "--out-prefix", "runs\paper_tables\1d_final_four_method",
    "--group-cols", "method,model_name,dataset_kind"
)

Invoke-PythonStep "1D expanded label-efficiency training" @(
    "scripts\run_chunked_matrix.py",
    "--matrix", "configs\ablations\1d_label_efficiency_expanded_aug_orbit.yaml",
    "--chunk-epochs", "25",
    "--max-chunks-per-job", "20"
)

Invoke-PythonStep "1D expanded label-efficiency table" @(
    "scripts\make_paper_tables.py",
    "--runs", "runs",
    "--out-prefix", "runs\paper_tables\1d_label_efficiency_expanded",
    "--group-cols", "data_fraction,method,model_name,dataset_kind",
    "--include-run-dir", "runs/(ablations/label_efficiency_seed_replication|final_1d/label_efficiency_expanded)/"
)

Invoke-PythonStep "1D OOD severity evaluation" @(
    "scripts\run_ood_severity_matrix.py",
    "--matrix", "configs\ablations\1d_ood_severity.yaml",
    "--out-prefix", "runs\paper_tables\1d_ood_severity"
)

Invoke-PythonStep "1D guarded unlabeled adaptation" @(
    "scripts\run_adapt_matrix.py",
    "--matrix", "configs\ablations\1d_guarded_unlabeled_adaptation.yaml"
)

Invoke-PythonStep "1D guarded unlabeled adaptation table" @(
    "scripts\make_adaptation_tables.py",
    "--runs", "runs\final_1d\unlabeled_adaptation",
    "--out-prefix", "runs\paper_tables\1d_guarded_unlabeled_adaptation",
    "--group-cols", "source_method,trainable,beta_l2_initial,epochs"
)

Write-Output "[$(Get-Date -Format o)] FINAL 1D EXPERIMENT BATCH COMPLETE"
