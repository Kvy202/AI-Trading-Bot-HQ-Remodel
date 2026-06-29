param(
  [ValidateRange(30,60)] [int]$Minutes = 45,
  [string]$Artifact = "model_artifacts/isolation_forest.joblib",
  [switch]$FreshShadowLog,
  [switch]$SkipVerifier
)

$ErrorActionPreference = 'Stop'

function Resolve-FullPath([string]$PathValue, [string]$BaseDir) {
  if ([string]::IsNullOrWhiteSpace($PathValue)) { return $null }
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return [System.IO.Path]::GetFullPath($PathValue)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $BaseDir $PathValue))
}

function Escape-Regex([string]$Value) {
  return [regex]::Escape($Value)
}

function Get-ScopedLiveProcess([string]$RootDir) {
  $rootRx = Escape-Regex $RootDir
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    ($_.CommandLine -match $rootRx) -and
    (($_.CommandLine -match 'tools(\\|/)live_(writer|executor)\.py') -or
     ($_.CommandLine -match 'isolation_forest_blocking_writer_launcher\.py'))
  }
}

function Quote-ProcessArg([string]$Value) {
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Get-CsvDataRowCount([string]$PathValue) {
  if (-not (Test-Path $PathValue)) { return 0 }
  $lineCount = 0
  foreach ($line in [System.IO.File]::ReadLines($PathValue)) {
    if (-not [string]::IsNullOrWhiteSpace($line)) { $lineCount++ }
  }
  return [Math]::Max(0, $lineCount - 1)
}

function Stop-OwnedWriter([System.Diagnostics.Process]$Proc, [string]$LockPath) {
  if ($null -eq $Proc) { return }
  try {
    $p = Get-Process -Id $Proc.Id -ErrorAction Stop
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
  } catch {}

  if (Test-Path $LockPath) {
    try {
      $owner = ((Get-Content $LockPath -Raw).Trim() -split ',')[0]
      if ($owner -eq [string]$Proc.Id) {
        Remove-Item $LockPath -Force -ErrorAction SilentlyContinue
      }
    } catch {}
  }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir '..')).Path
Set-Location $root

$logsDir = Join-Path $root 'logs'
$reportsDir = Join-Path $root 'reports'
New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $reportsDir -Force | Out-Null

$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) {
  $py = "python"
}

$artifactFull = Resolve-FullPath -PathValue $Artifact -BaseDir $root
if (-not (Test-Path $artifactFull)) {
  Write-Host "[isolation-paper] REFUSING: artifact not found: $artifactFull" -ForegroundColor Red
  exit 1
}

$existing = Get-ScopedLiveProcess -RootDir $root
if ($existing) {
  Write-Host "[isolation-paper] REFUSING: live writer/executor process is already running under this repo." -ForegroundColor Red
  Write-Host "Stop the bot first, then rerun this paper-only runbook:" -ForegroundColor Yellow
  Write-Host "  .\tools\stop_live.ps1"
  $existing | Select-Object ProcessId, CommandLine | Format-Table -AutoSize
  exit 1
}

$preflightCode = @'
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

try:
    from runtime.loader import apply_run_config
    apply_run_config(root)
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(root / ".env", override=True)
except Exception:
    pass

from runtime.guardrails import resolve_trading_mode
from runtime.settings import Settings

s = Settings.from_env()
d = resolve_trading_mode(s)
live_requested = bool((s.live_trading and not s.paper_trading) or (s.live_mode and not s.exec_paper))
hyperliquid_mainnet_selected = bool(s.exchange == "hyperliquid" and not s.hl_testnet)
production_detected = bool(s.environment == "production")
unsafe = bool(live_requested or d.place_real_orders or production_detected or hyperliquid_mainnet_selected)
reasons = []
if live_requested:
    reasons.append("runtime config requests live orders (LIVE_TRADING/PAPER_TRADING or LIVE_MODE/EXEC_PAPER)")
if d.place_real_orders:
    reasons.append("guardrail resolves to a real-order mode")
if production_detected:
    reasons.append("ENVIRONMENT=production")
if hyperliquid_mainnet_selected:
    reasons.append("Hyperliquid mainnet selected (HL_TESTNET=false)")

print(json.dumps({
    "refuse": unsafe,
    "reasons": reasons,
    "exchange": s.exchange,
    "environment": s.environment,
    "live_trading": s.live_trading,
    "paper_trading": s.paper_trading,
    "live_mode": s.live_mode,
    "exec_paper": s.exec_paper,
    "hl_testnet": s.hl_testnet,
    "bitget_sandbox": s.bitget_sandbox,
    "guardrail_mode": d.mode.value,
    "place_real_orders": d.place_real_orders,
    "testnet": d.testnet,
    "sandbox": d.sandbox,
}))
'@

$preflightRaw = & $py -c $preflightCode $root
if ($LASTEXITCODE -ne 0) {
  Write-Host "[isolation-paper] REFUSING: mode preflight failed." -ForegroundColor Red
  exit 1
}
$preflight = $preflightRaw | ConvertFrom-Json
if ($preflight.refuse) {
  Write-Host "[isolation-paper] REFUSING: live/mainnet mode detected." -ForegroundColor Red
  foreach ($reason in $preflight.reasons) {
    Write-Host "  - $reason" -ForegroundColor Yellow
  }
  Write-Host ("  effective: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
    $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading)
  exit 1
}

Write-Host ("[isolation-paper] Mode preflight OK: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
  $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading) -ForegroundColor Green

if (-not $SkipVerifier) {
  Write-Host "[isolation-paper] Running deterministic Isolation Forest verifier..."
  & $py "tools\verify_isolation_forest.py" --artifact $artifactFull
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[isolation-paper] REFUSING: verifier failed." -ForegroundColor Red
    exit 1
  }
}

$shadowLog = Join-Path $logsDir 'isolation_forest_shadow.csv'
if ($FreshShadowLog -and (Test-Path $shadowLog)) {
  $stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss')
  $archive = Join-Path $logsDir "isolation_forest_shadow.$stamp.csv"
  Move-Item -LiteralPath $shadowLog -Destination $archive
  Write-Host "[isolation-paper] Archived previous shadow log to $archive"
}

$rowsBefore = Get-CsvDataRowCount -PathValue $shadowLog
$writerOut = Join-Path $logsDir 'isolation_forest_blocking_paper_writer.out'
$writerErr = Join-Path $logsDir 'isolation_forest_blocking_paper_writer.err'
$writerLock = Join-Path $logsDir 'live_writer.lock'
$launcherPath = Join-Path $logsDir 'isolation_forest_blocking_writer_launcher.py'
$reportJson = Join-Path $reportsDir 'isolation_forest_blocking_paper_summary.json'

$launcherCode = @'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
artifact = sys.argv[2]
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import tools.live_writer as live_writer

# Force the runbook's process-local paper/test Isolation Forest settings after
# live_writer imports config/run.json and .env. This changes no repo defaults.
os.environ["USE_ISOLATION_FOREST"] = "true"
os.environ["ISOLATION_FOREST_BLOCKING"] = "true"
os.environ["ISOLATION_FOREST_ARTIFACT"] = artifact
os.environ["USE_XGBOOST_SIGNAL"] = "false"
os.environ["USE_SURVIVAL_EXIT"] = "false"
os.environ["LIVE_TRADING"] = "false"
os.environ["PAPER_TRADING"] = "true"
os.environ["LIVE_MODE"] = "0"
os.environ["EXEC_PAPER"] = "1"
os.environ["CONFIRM_LIVE_TRADING"] = ""
os.environ.setdefault("HL_TESTNET", "true")

sys.argv = ["tools/live_writer.py"]
live_writer.main()
'@

Set-Content -Path $launcherPath -Value $launcherCode -Encoding UTF8

Write-Host "[isolation-paper] Starting live_writer only; executor is not started."
Write-Host "[isolation-paper] Forced flags:"
Write-Host "  USE_ISOLATION_FOREST=true"
Write-Host "  ISOLATION_FOREST_BLOCKING=true"
Write-Host "  ISOLATION_FOREST_ARTIFACT=$artifactFull"
Write-Host "  USE_XGBOOST_SIGNAL=false"
Write-Host "  USE_SURVIVAL_EXIT=false"
Write-Host ("[isolation-paper] Duration: {0} minutes" -f $Minutes)

$writerArgs = @($launcherPath, $root, $artifactFull) | ForEach-Object { Quote-ProcessArg $_ }
$writer = Start-Process -FilePath $py `
  -ArgumentList ($writerArgs -join ' ') `
  -WorkingDirectory $root `
  -RedirectStandardOutput $writerOut `
  -RedirectStandardError $writerErr `
  -PassThru -WindowStyle Hidden

try {
  $endAt = (Get-Date).AddMinutes($Minutes)
  while ((Get-Date) -lt $endAt) {
    Start-Sleep -Seconds 30
    try {
      $null = Get-Process -Id $writer.Id -ErrorAction Stop
    } catch {
      Write-Host "[isolation-paper] REFUSING: writer exited early. Check $writerErr" -ForegroundColor Red
      exit 1
    }
    $remaining = [Math]::Ceiling(($endAt - (Get-Date)).TotalMinutes)
    Write-Host ("[isolation-paper] writer PID={0}; about {1} minute(s) remaining" -f $writer.Id, [Math]::Max(0, $remaining))
  }
}
finally {
  Stop-OwnedWriter -Proc $writer -LockPath $writerLock
}

$rowsAfter = Get-CsvDataRowCount -PathValue $shadowLog
if (-not (Test-Path $shadowLog)) {
  Write-Host "[isolation-paper] FAIL: live_writer did not create $shadowLog" -ForegroundColor Red
  exit 1
}
if ($rowsAfter -le $rowsBefore) {
  Write-Host ("[isolation-paper] FAIL: no new shadow rows written (before={0}, after={1})" -f $rowsBefore, $rowsAfter) -ForegroundColor Red
  Write-Host "Check $writerOut and $writerErr"
  exit 1
}

Write-Host ("[isolation-paper] Shadow log OK: {0} new row(s) in {1}" -f ($rowsAfter - $rowsBefore), $shadowLog) -ForegroundColor Green

Write-Host "[isolation-paper] Generating report..."
& $py "tools\experimental_shadow_report.py" --logs-dir $logsDir --json --json-out $reportJson
if ($LASTEXITCODE -ne 0) {
  Write-Host "[isolation-paper] FAIL: report generation failed." -ForegroundColor Red
  exit 1
}

Write-Host ""
Write-Host "[isolation-paper] Done."
Write-Host "  shadow_log: $shadowLog"
Write-Host "  report_json: $reportJson"
Write-Host "  writer_stdout: $writerOut"
Write-Host "  writer_stderr: $writerErr"
