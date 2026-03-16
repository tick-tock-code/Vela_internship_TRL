$ErrorActionPreference = "Stop"

$root = "C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\3_pipeline_for_features"
$featuresJson = Join-Path $root "features.json"
$featuresParquet = Join-Path $root "features_full.parquet"

Write-Host "Stage 1: Extract features (vela_TRL)"
& C:\Users\joelb\.conda\envs\vela_TRL\python.exe `
  "$root\vcbench_pipeline.py" `
  --dataset full `
  --mode human `
  --feature_config "$featuresJson" `
  --output_parquet "$featuresParquet" `
  --extract_only

if ($LASTEXITCODE -ne 0) {
  Write-Host "Stage 1 failed. Exiting."
  exit $LASTEXITCODE
}

Write-Host "Stage 2: Train from Parquet (torch_env2)"
& C:\ProgramData\anaconda3\envs\torch_env2\python.exe `
  "$root\train_from_parquet.py" `
  --input_parquet "$featuresParquet"

Write-Host "Done."
