$ErrorActionPreference = "Stop"

$root = "C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\3_pipeline_for_features"

Write-Host "Launching feature selector GUI..."
& C:\ProgramData\anaconda3\envs\torch_env2\python.exe "$root\feature_selector_gui.py"
