<#
Usage:
  .\tools\run_combined_shadow_paper_test.ps1 -Minutes 5
  .\tools\run_combined_shadow_paper_test.ps1 -Minutes 30
  .\tools\run_combined_shadow_paper_test.ps1 -Minutes 60 -FreshShadowLogs -FreshPaperLogs
#>

param(
  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,
  [switch]$FreshShadowLogs,
  [switch]$FreshPaperLogs,
  [switch]$SkipVerifier
)

$ErrorActionPreference = 'Stop'

function Escape-Regex([string]$Value) {
  return [regex]::Escape($Value)
}

function Quote-ProcessArg([string]$Value) {
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Get-ScopedLiveProcess([string]$RootDir) {
  $rootRx = Escape-Regex $RootDir
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    ($_.CommandLine -match $rootRx) -and
    (($_.CommandLine -match 'tools(\\|/)live_(writer|executor)\.py') -or
     ($_.CommandLine -match 'combined_shadow_(writer|executor)_launcher\.py'))
  }
}

function Get-CsvDataRowCount([string]$PathValue) {
  if (-not (Test-Path $PathValue)) { return 0 }
  $lineCount = 0
  foreach ($line in [System.IO.File]::ReadLines($PathValue)) {
    if (-not [string]::IsNullOrWhiteSpace($line)) { $lineCount++ }
  }
  return [Math]::Max(0, $lineCount - 1)
}

function Get-CsvHeader([string]$PathValue) {
  if (-not (Test-Path $PathValue)) { return @() }
  $first = [System.IO.File]::ReadLines($PathValue) | Select-Object -First 1
  if ([string]::IsNullOrWhiteSpace($first)) { return @() }
  return ,($first -split ',')
}

function Test-CsvColumn([string]$PathValue, [string]$ColumnName) {
  $header = Get-CsvHeader -PathValue $PathValue
  return ($header -contains $ColumnName)
}

function Archive-FileIfExists([string]$PathValue, [string]$ArchiveDir) {
  if (-not (Test-Path $PathValue)) { return }
  New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
  $dest = Join-Path $ArchiveDir ([System.IO.Path]::GetFileName($PathValue))
  Move-Item -LiteralPath $PathValue -Destination $dest -Force
  Write-Host "[combined-shadow] Archived $PathValue to $dest"
}

function Stop-OwnedProcess([System.Diagnostics.Process]$Proc, [string]$LockPath) {
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

function Assert-ReportWritten([string]$PathValue) {
  if (-not (Test-Path $PathValue)) {
    Write-Host "[combined-shadow] FAIL: report was not written: $PathValue" -ForegroundColor Red
    exit 1
  }
  if ((Get-Item $PathValue).Length -le 0) {
    Write-Host "[combined-shadow] FAIL: report is empty: $PathValue" -ForegroundColor Red
    exit 1
  }
  Write-Host "[combined-shadow] Report OK: $PathValue" -ForegroundColor Green
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
. (Join-Path $scriptDir 'apply_experiment_mode.ps1')

$experimentMode = 'combined_shadow'
$forcedPaperEnv = Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode
$forcedPaperEnv['LIVE_TRADING'] = 'false'
$forcedPaperEnv['PAPER_TRADING'] = 'true'
$forcedPaperEnv['LIVE_MODE'] = 'false'
$forcedPaperEnv['EXEC_PAPER'] = 'true'
$forcedPaperEnv['PLACE_REAL_ORDERS'] = 'false'
$forcedPaperEnv['ISOLATION_FOREST_BLOCKING'] = 'false'
$forcedPaperEnv['XGBOOST_SIGNAL_BLOCKING'] = 'false'
$forcedPaperEnv['SURVIVAL_EXIT_ACTIVE'] = 'false'
$forcedPaperEnv['ADVANCED_RISK_ACTIVE'] = 'false'
$forcedPaperEnv['EXEC_RESTORE_STATE'] = 'false'
Set-ExperimentModeEnvironment -Overrides $forcedPaperEnv
Set-Item -Path "Env:CONFIRM_LIVE_TRADING" -Value ""
$forcedEnvPath = Join-Path $logsDir 'combined_shadow_mode_env.json'
Set-Content -Path $forcedEnvPath -Value ($forcedPaperEnv | ConvertTo-Json -Depth 3) -Encoding UTF8

$existing = Get-ScopedLiveProcess -RootDir $root
if ($existing) {
  Write-Host "[combined-shadow] REFUSING: live writer/executor process is already running under this repo." -ForegroundColor Red
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
mode_env = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
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

os.environ.update(mode_env)
os.environ["CONFIRM_LIVE_TRADING"] = ""

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

$preflightPath = Join-Path $logsDir 'combined_shadow_preflight.py'
Set-Content -Path $preflightPath -Value $preflightCode -Encoding UTF8
$preflightRaw = & $py $preflightPath $root $forcedEnvPath
if ($LASTEXITCODE -ne 0) {
  Write-Host "[combined-shadow] REFUSING: mode preflight failed." -ForegroundColor Red
  exit 1
}
$preflight = $preflightRaw | ConvertFrom-Json
if ($preflight.refuse) {
  Write-Host "[combined-shadow] REFUSING: live/mainnet mode detected." -ForegroundColor Red
  foreach ($reason in $preflight.reasons) {
    Write-Host "  - $reason" -ForegroundColor Yellow
  }
  Write-Host ("  effective: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
    $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading)
  exit 1
}

Write-Host ("[combined-shadow] Mode preflight OK: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
  $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading) -ForegroundColor Green

if (-not $SkipVerifier) {
  Write-Host "[combined-shadow] Verifying Isolation Forest missing-artifact safety..."
  & $py "tools\verify_isolation_forest.py" --artifact "model_artifacts/isolation_forest.joblib" --missing-artifact-check
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[combined-shadow] REFUSING: Isolation Forest verifier failed." -ForegroundColor Red
    exit 1
  }

  Write-Host "[combined-shadow] Verifying XGBoost missing-artifact safety..."
  & $py "tools\verify_xgboost_signal.py" --artifact "model_artifacts/xgboost_signal.joblib" --missing-artifact-check
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[combined-shadow] REFUSING: XGBoost verifier failed." -ForegroundColor Red
    exit 1
  }

  Write-Host "[combined-shadow] Verifying Survival Exit missing-artifact safety..."
  & $py "tools\verify_survival_exit.py" --artifact "model_artifacts/survival_exit.joblib" --missing-artifact-check
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[combined-shadow] REFUSING: Survival Exit verifier failed." -ForegroundColor Red
    exit 1
  }

  Write-Host "[combined-shadow] Running deterministic Advanced Risk verifier..."
  & $py "tools\verify_advanced_risk.py"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[combined-shadow] REFUSING: Advanced Risk verifier failed." -ForegroundColor Red
    exit 1
  }
}

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss')
$archiveDir = Join-Path $logsDir "combined_shadow_archive_$stamp"
$shadowLogNames = @(
  'isolation_forest_shadow.csv',
  'xgboost_signal_shadow.csv',
  'survival_exit_shadow.csv',
  'advanced_risk_shadow.csv'
)

if ($FreshShadowLogs) {
  foreach ($name in $shadowLogNames) {
    Archive-FileIfExists -PathValue (Join-Path $logsDir $name) -ArchiveDir $archiveDir
  }
}

$closedMaster = Join-Path $logsDir 'trades_closed.csv'
if ($FreshPaperLogs) {
  foreach ($file in Get-ChildItem -Path $logsDir -Filter 'trades_paper_*.csv' -File -ErrorAction SilentlyContinue) {
    Archive-FileIfExists -PathValue $file.FullName -ArchiveDir $archiveDir
  }
  foreach ($file in Get-ChildItem -Path $logsDir -Filter 'trades_closed_*.csv' -File -ErrorAction SilentlyContinue) {
    Archive-FileIfExists -PathValue $file.FullName -ArchiveDir $archiveDir
  }
  Archive-FileIfExists -PathValue $closedMaster -ArchiveDir $archiveDir
}

$shadowRowsBefore = @{}
foreach ($name in $shadowLogNames) {
  $shadowRowsBefore[$name] = Get-CsvDataRowCount -PathValue (Join-Path $logsDir $name)
}

$writerOut = Join-Path $logsDir 'combined_shadow_paper_writer.out'
$writerErr = Join-Path $logsDir 'combined_shadow_paper_writer.err'
$executorOut = Join-Path $logsDir 'combined_shadow_paper_executor.out'
$executorErr = Join-Path $logsDir 'combined_shadow_paper_executor.err'
$writerLock = Join-Path $logsDir 'live_writer.lock'
$executorLock = Join-Path $logsDir 'live_executor.lock'
$writerLauncherPath = Join-Path $logsDir 'combined_shadow_writer_launcher.py'
$executorLauncherPath = Join-Path $logsDir 'combined_shadow_executor_launcher.py'
$reportJson = Join-Path $reportsDir 'combined_shadow_paper_summary.json'
$auditJson = Join-Path $reportsDir 'combined_shadow_xgboost_audit.json'

$writerLauncherCode = @'
import os
import sys
import json
from pathlib import Path

root = Path(sys.argv[1])
forced_env = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import tools.live_writer as live_writer

os.environ.update(forced_env)
os.environ["CONFIRM_LIVE_TRADING"] = ""
os.environ.setdefault("HL_TESTNET", "true")

sys.argv = ["tools/live_writer.py"]
live_writer.main()
'@

$executorLauncherCode = @'
import os
import sys
import json
from pathlib import Path

root = Path(sys.argv[1])
FORCED = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8-sig"))
os.chdir(root)
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

import tools.live_executor as live_executor

FORCED["CONFIRM_LIVE_TRADING"] = ""

def force_env():
    os.environ.update(FORCED)

_orig_load_dotenv = live_executor.load_dotenv

def _forced_load_dotenv(*args, **kwargs):
    _orig_load_dotenv(*args, **kwargs)
    force_env()

live_executor.load_dotenv = _forced_load_dotenv
force_env()

sys.argv = ["tools/live_executor.py", "--paper", "--signals", "logs/live_signals.csv"]
live_executor.main()
'@

Set-Content -Path $writerLauncherPath -Value $writerLauncherCode -Encoding UTF8
Set-Content -Path $executorLauncherPath -Value $executorLauncherCode -Encoding UTF8

Write-Host "[combined-shadow] Starting live_writer and live_executor in paper mode."
Write-Host "[combined-shadow] Experiment mode: $experimentMode"
Write-Host "[combined-shadow] Forced flags:"
foreach ($name in $forcedPaperEnv.Keys) {
  Write-Host ("  {0}={1}" -f $name, $forcedPaperEnv[$name])
}
Write-Host ("[combined-shadow] Duration: {0} minutes" -f $Minutes)

$writerArgs = @($writerLauncherPath, $root, $forcedEnvPath) | ForEach-Object { Quote-ProcessArg $_ }
$executorArgs = @($executorLauncherPath, $root, $forcedEnvPath) | ForEach-Object { Quote-ProcessArg $_ }

$writer = $null
$executor = $null
try {
  $writer = Start-Process -FilePath $py `
    -ArgumentList ($writerArgs -join ' ') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $writerOut `
    -RedirectStandardError $writerErr `
    -PassThru -WindowStyle Hidden

  Start-Sleep -Seconds 5

  $executor = Start-Process -FilePath $py `
    -ArgumentList ($executorArgs -join ' ') `
    -WorkingDirectory $root `
    -RedirectStandardOutput $executorOut `
    -RedirectStandardError $executorErr `
    -PassThru -WindowStyle Hidden

  $endAt = (Get-Date).AddMinutes($Minutes)
  while ((Get-Date) -lt $endAt) {
    Start-Sleep -Seconds 30
    foreach ($procInfo in @(@("writer", $writer, $writerErr), @("executor", $executor, $executorErr))) {
      $label = $procInfo[0]
      $proc = $procInfo[1]
      $errPath = $procInfo[2]
      try {
        $null = Get-Process -Id $proc.Id -ErrorAction Stop
      } catch {
        Write-Host "[combined-shadow] REFUSING: $label exited early. Check $errPath" -ForegroundColor Red
        exit 1
      }
    }
    $remaining = [Math]::Ceiling(($endAt - (Get-Date)).TotalMinutes)
    Write-Host ("[combined-shadow] writer PID={0}; executor PID={1}; about {2} minute(s) remaining" -f `
      $writer.Id, $executor.Id, [Math]::Max(0, $remaining))
  }
}
finally {
  Stop-OwnedProcess -Proc $executor -LockPath $executorLock
  Stop-OwnedProcess -Proc $writer -LockPath $writerLock
}

Write-Host "[combined-shadow] Generating experimental shadow report..."
& $py "tools\experimental_shadow_report.py" --logs-dir $logsDir --json --json-out $reportJson
if ($LASTEXITCODE -ne 0) {
  Write-Host "[combined-shadow] FAIL: experimental shadow report generation failed." -ForegroundColor Red
  exit 1
}

Write-Host "[combined-shadow] Generating XGBoost rejection audit..."
& $py "tools\audit_xgboost_rejections.py" --json --json-out $auditJson
if ($LASTEXITCODE -ne 0) {
  Write-Host "[combined-shadow] FAIL: XGBoost audit generation failed." -ForegroundColor Red
  exit 1
}

Assert-ReportWritten -PathValue $reportJson
Assert-ReportWritten -PathValue $auditJson

Write-Host "[combined-shadow] Validating shadow logs..."
foreach ($name in $shadowLogNames) {
  $path = Join-Path $logsDir $name
  $before = [int]$shadowRowsBefore[$name]
  $after = Get-CsvDataRowCount -PathValue $path
  $delta = $after - $before
  if ($delta -gt 0) {
    Write-Host ("[combined-shadow] OK: {0} wrote {1} new row(s)" -f $name, $delta) -ForegroundColor Green
  } elseif ($name -eq 'survival_exit_shadow.csv') {
    Write-Host "[combined-shadow] INFO: survival_exit_shadow.csv has no new rows. This is expected if no open position was evaluated."
  } elseif ($name -eq 'advanced_risk_shadow.csv') {
    Write-Host "[combined-shadow] INFO: advanced_risk_shadow.csv has no new rows. This is expected if executor processed no candidate signals."
  } else {
    Write-Host ("[combined-shadow] WARN: {0} has no new rows. Check artifact availability, writer output, and market data." -f $name) -ForegroundColor Yellow
  }
}

$liveSignals = Join-Path $logsDir 'live_signals.csv'
if (Test-CsvColumn -PathValue $liveSignals -ColumnName 'signal_id') {
  Write-Host "[combined-shadow] OK: live_signals.csv includes signal_id" -ForegroundColor Green
} else {
  Write-Host "[combined-shadow] WARN: live_signals.csv is missing signal_id or was not written." -ForegroundColor Yellow
}

$paperRows = 0
$paperMissingSignalId = @()
foreach ($file in Get-ChildItem -Path $logsDir -Filter 'trades_paper_*.csv' -File -ErrorAction SilentlyContinue) {
  $rows = Get-CsvDataRowCount -PathValue $file.FullName
  $paperRows += $rows
  if ($rows -gt 0 -and -not (Test-CsvColumn -PathValue $file.FullName -ColumnName 'signal_id')) {
    $paperMissingSignalId += $file.FullName
  }
}
if ($paperRows -eq 0) {
  Write-Host "[combined-shadow] INFO: no paper trades occurred during this run; not failing."
} elseif ($paperMissingSignalId.Count -gt 0) {
  Write-Host "[combined-shadow] FAIL: paper trade logs with rows are missing signal_id:" -ForegroundColor Red
  $paperMissingSignalId | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
  exit 1
} else {
  Write-Host ("[combined-shadow] OK: paper trade logs include signal_id across {0} row(s)" -f $paperRows) -ForegroundColor Green
}

$closedRows = 0
$closedMissingSignalId = @()
$closedFiles = @()
if (Test-Path $closedMaster) {
  $closedFiles += Get-Item $closedMaster
}
$closedFiles += @(Get-ChildItem -Path $logsDir -Filter 'trades_closed_*.csv' -File -ErrorAction SilentlyContinue)
foreach ($file in $closedFiles) {
  $rows = Get-CsvDataRowCount -PathValue $file.FullName
  $closedRows += $rows
  if ($rows -gt 0 -and -not (Test-CsvColumn -PathValue $file.FullName -ColumnName 'signal_id')) {
    $closedMissingSignalId += $file.FullName
  }
}
if ($closedRows -eq 0) {
  Write-Host "[combined-shadow] INFO: no closed trades occurred during this run; not failing."
} elseif ($closedMissingSignalId.Count -gt 0) {
  Write-Host "[combined-shadow] FAIL: closed trade logs with rows are missing signal_id:" -ForegroundColor Red
  $closedMissingSignalId | ForEach-Object { Write-Host "  $_" -ForegroundColor Yellow }
  exit 1
} else {
  Write-Host ("[combined-shadow] OK: closed trade logs include signal_id across {0} row(s)" -f $closedRows) -ForegroundColor Green
}

Write-Host ""
Write-Host "[combined-shadow] Done."
Write-Host "  experiment_mode: $experimentMode"
Write-Host "  report_json: $reportJson"
Write-Host "  xgboost_audit_json: $auditJson"
Write-Host "  writer_stdout: $writerOut"
Write-Host "  writer_stderr: $writerErr"
Write-Host "  executor_stdout: $executorOut"
Write-Host "  executor_stderr: $executorErr"
