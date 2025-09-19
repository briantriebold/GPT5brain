$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$taskName = "GPT5OptimizeNow"
$ps = "$($root)\optimize_now.ps1"
Write-Host "To register a daily task at 2:00 AM, run:" -ForegroundColor Yellow
Write-Host "schtasks /Create /SC DAILY /TN $taskName /TR \"powershell -ExecutionPolicy Bypass -File `\"$ps`\"\" /ST 02:00" -ForegroundColor Yellow

