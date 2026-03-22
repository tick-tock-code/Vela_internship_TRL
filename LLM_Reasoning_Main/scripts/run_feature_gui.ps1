$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = Split-Path -Parent $root
$python = (Get-Command python).Source
& $python "$root\python\tools\feature_selector_gui.py"
