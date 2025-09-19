$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $root\..
& .\\.venv\\Scripts\\gpt5 optimize now --objective "Nightly Optimize" --strategy pipeline --json | Out-Host
& git add reports gpt5\data\memory.db | Out-Null
& git commit -m "chore: nightly optimize report & dashboard" | Out-Host
try {
  $branch = git rev-parse --abbrev-ref HEAD
  git push -u origin $branch | Out-Host
} catch {}
Pop-Location

.\\.venv\\Scripts\\gpt5 report index --dir reports --json | Out-Host

