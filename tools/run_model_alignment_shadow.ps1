param(
  [ValidateRange(3, 120)]
  [int]$UniqueBars = 3,
  [string]$Symbols = "BTCUSDT,ETHUSDT",
  [switch]$FreshLogs,
  [switch]$DryRun,
  [string]$CampaignDir = "",
  [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
  # These values exist only in this runbook process and its diagnostic child.
  $env:LIVE_TRADING = "false"
  $env:PAPER_TRADING = "true"
  $env:LIVE_MODE = "false"
  $env:EXEC_PAPER = "true"
  $env:PLACE_REAL_ORDERS = "false"
  $env:DL_TIMEFRAME = "5m"
  $env:DL_SEQ_LEN = "64"

  $arguments = @(
    "tools/model_alignment_shadow.py",
    "live-shadow",
    "--unique-bars", [string]$UniqueBars,
    "--symbols", $Symbols,
    "--output-root", "reports/model_alignment_live"
  )
  if ($CampaignDir) {
    $arguments += @("--campaign-dir", $CampaignDir)
  }
  if ($DryRun) {
    $arguments += "--dry-run"
  }
  if ($FreshLogs) {
    $arguments += "--fresh-logs"
  }

  Write-Host ("[alignment-shadow] research_only=1 orders_allowed=0 timeframe=5m seq_len=64 unique_bars={0}" -f $UniqueBars)
  & $Python @arguments
  if ($LASTEXITCODE -ne 0) {
    throw "model alignment shadow exited with status $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}
