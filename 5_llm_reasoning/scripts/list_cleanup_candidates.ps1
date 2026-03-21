param(
  [string]$RootPath = 'C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\5_llm_reasoning',
  [string[]]$SubPaths = @('features_storage\llm_reasoning', 'training_logs'),
  [string]$RunStartIso = ''
)

if (-not $RunStartIso) {
  Write-Output 'RunStartIso is required.'
  exit 1
}

$runStart = [DateTimeOffset]::Parse($RunStartIso)
$toDelete = @()
foreach ($sub in $SubPaths) {
  $path = Join-Path $RootPath $sub
  if (-not (Test-Path $path)) { continue }
  Get-ChildItem -Path $path -Recurse -File | ForEach-Object {
    if ($_.LastWriteTimeUtc -gt $runStart.UtcDateTime) {
      $toDelete += $_.FullName
    }
  }
}

$toDelete | Sort-Object | Set-Content -Path (Join-Path $RootPath 'cleanup_candidates.txt')
Write-Output ("Candidates: {0}" -f $toDelete.Count)
