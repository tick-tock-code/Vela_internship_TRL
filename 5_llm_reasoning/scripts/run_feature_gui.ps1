$ErrorActionPreference = "Stop"

$root = Resolve-Path "$PSScriptRoot\.."
$python = "C:\Users\joelb\.conda\envs\vela_TRL\python.exe"

Write-Host "Launching feature selector GUI..."
& $python "$root\python\feature_selector_gui.py"
