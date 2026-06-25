@echo off
setlocal

cd /d "%~dp0\.."
set PYTHONPATH=src
set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set PYTHON=.venv\Scripts\python.exe
set RUN_LOG=runs\final_1d\final_1d_batch.log

echo [%DATE% %TIME%] wrapper started in %CD% > %RUN_LOG%

if "%~1"=="--plan" (
  echo plan: 1D final four-method training
  echo plan: 1D final four-method table
  echo plan: 1D expanded label-efficiency training
  echo plan: 1D expanded label-efficiency table
  echo plan: 1D OOD severity evaluation
  echo plan: 1D guarded unlabeled adaptation
  echo plan: 1D guarded unlabeled adaptation table
  exit /b 0
)

echo [%DATE% %TIME%] START 1D final four-method training
%PYTHON% scripts\run_chunked_matrix.py --matrix configs\ablations\1d_final_four_method_seeds.yaml --chunk-epochs 25 --max-chunks-per-job 20
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D final four-method training

echo [%DATE% %TIME%] START 1D final four-method table
%PYTHON% scripts\make_paper_tables.py --runs runs\final_1d\four_method --out-prefix runs\paper_tables\1d_final_four_method --group-cols method,model_name,dataset_kind
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D final four-method table

echo [%DATE% %TIME%] START 1D expanded label-efficiency training
%PYTHON% scripts\run_chunked_matrix.py --matrix configs\ablations\1d_label_efficiency_expanded_aug_orbit.yaml --chunk-epochs 25 --max-chunks-per-job 20
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D expanded label-efficiency training

echo [%DATE% %TIME%] START 1D expanded label-efficiency table
%PYTHON% scripts\make_paper_tables.py --runs runs --out-prefix runs\paper_tables\1d_label_efficiency_expanded --group-cols data_fraction,method,model_name,dataset_kind --include-run-dir "runs/(ablations/label_efficiency_seed_replication|final_1d/label_efficiency_expanded)/"
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D expanded label-efficiency table

echo [%DATE% %TIME%] START 1D OOD severity evaluation
%PYTHON% scripts\run_ood_severity_matrix.py --matrix configs\ablations\1d_ood_severity.yaml --out-prefix runs\paper_tables\1d_ood_severity
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D OOD severity evaluation

echo [%DATE% %TIME%] START 1D guarded unlabeled adaptation
%PYTHON% scripts\run_adapt_matrix.py --matrix configs\ablations\1d_guarded_unlabeled_adaptation.yaml
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D guarded unlabeled adaptation

echo [%DATE% %TIME%] START 1D guarded unlabeled adaptation table
%PYTHON% scripts\make_adaptation_tables.py --runs runs\final_1d\unlabeled_adaptation --out-prefix runs\paper_tables\1d_guarded_unlabeled_adaptation --group-cols source_method,trainable,beta_l2_initial,epochs
if errorlevel 1 exit /b %errorlevel%
echo [%DATE% %TIME%] DONE 1D guarded unlabeled adaptation table

echo [%DATE% %TIME%] FINAL 1D EXPERIMENT BATCH COMPLETE
