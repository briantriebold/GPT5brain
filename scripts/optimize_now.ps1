$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root\..
& .\.venv\Scripts\gpt5 optimize now --objective "Nightly Optimize" --strategy pipeline --json | Out-Host
& git add reports gpt5\data\memory.db | Out-Null
& git commit -m "chore: nightly optimize report & dashboard" | Out-Host
Pop-Location

