<#
Usage:
  .\tools\run_survival_active_paper_test.ps1 -Minutes 5
  .\tools\run_survival_active_paper_test.ps1 -Minutes 30
  .\tools\run_survival_active_paper_test.ps1 -Minutes 60 -FreshShadowLog -FreshPaperLogs
#>

param(
  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,
  [string]$Artifact = "model_artifacts/survival_exit.joblib",
  [double]$RiskThreshold = 0.60,
  [switch]$FreshShadowLog,
  [switch]$FreshPaperLogs,
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

function Quote-ProcessArg([string]$Value) {
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Get-ScopedLiveProcess([string]$RootDir) {
  $rootRx = Escape-Regex $RootDir
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    ($_.CommandLine -match $rootRx) -and
    (($_.CommandLine -match 'tools(\\|/)live_(writer|executor)\.py') -or
     ($_.CommandLine -match 'survival_active_(writer|executor)_launcher\.py'))
  }
}

function Archive-FileIfExists([string]$PathValue, [string]$ArchiveDir) {
  if (-not (Test-Path $PathValue)) { return }
  New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
  $dest = Join-Path $ArchiveDir ([System.IO.Path]::GetFileName($PathValue))
  Move-Item -LiteralPath $PathValue -Destination $dest -Force
  Write-Host "[survival-active] Archived $PathValue to $dest"
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

$artifactFull = Resolve-FullPath -PathValue $Artifact -BaseDir $root
if (-not (Test-Path $artifactFull)) {
  Write-Host "[survival-active] REFUSING: artifact not found: $artifactFull" -ForegroundColor Red
  exit 1
}

$experimentMode = 'survival_shadow'
$forcedPaperEnv = Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode
$forcedPaperEnv['SURVIVAL_EXIT_ACTIVE'] = 'true'
$forcedPaperEnv['SURVIVAL_EXIT_ARTIFACT'] = $artifactFull
$forcedPaperEnv['SURVIVAL_EXIT_RISK_THRESHOLD'] = ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0:0.####}", $RiskThreshold))
$forcedPaperEnv['EXEC_RESTORE_STATE'] = 'false'
Set-ExperimentModeEnvironment -Overrides $forcedPaperEnv
Set-Item -Path "Env:CONFIRM_LIVE_TRADING" -Value ""
$forcedEnvPath = Join-Path $logsDir 'survival_active_mode_env.json'
Set-Content -Path $forcedEnvPath -Value ($forcedPaperEnv | ConvertTo-Json -Depth 3) -Encoding UTF8

$existing = Get-ScopedLiveProcess -RootDir $root
if ($existing) {
  Write-Host "[survival-active] REFUSING: live writer/executor process is already running under this repo." -ForegroundColor Red
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

$preflightPath = Join-Path $logsDir 'survival_active_preflight.py'
Set-Content -Path $preflightPath -Value $preflightCode -Encoding UTF8
$preflightRaw = & $py $preflightPath $root $forcedEnvPath
if ($LASTEXITCODE -ne 0) {
  Write-Host "[survival-active] REFUSING: mode preflight failed." -ForegroundColor Red
  exit 1
}
$preflight = $preflightRaw | ConvertFrom-Json
if ($preflight.refuse) {
  Write-Host "[survival-active] REFUSING: live/mainnet mode detected." -ForegroundColor Red
  foreach ($reason in $preflight.reasons) {
    Write-Host "  - $reason" -ForegroundColor Yellow
  }
  Write-Host ("  effective: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
    $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading)
  exit 1
}

Write-Host ("[survival-active] Mode preflight OK: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
  $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading) -ForegroundColor Green

if (-not $SkipVerifier) {
  Write-Host "[survival-active] Verifying missing artifact path safety..."
  & $py "tools\verify_survival_exit.py" --artifact $artifactFull --missing-artifact-check
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[survival-active] REFUSING: missing artifact verifier failed." -ForegroundColor Red
    exit 1
  }

  Write-Host "[survival-active] Running deterministic Survival active verifier..."
  & $py "tools\verify_survival_exit.py" --artifact $artifactFull
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[survival-active] REFUSING: verifier failed." -ForegroundColor Red
    exit 1
  }
}

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss')
$archiveDir = Join-Path $logsDir "survival_active_archive_$stamp"
$shadowLog = Join-Path $logsDir 'survival_exit_shadow.csv'
$closedMaster = Join-Path $logsDir 'trades_closed.csv'

if ($FreshShadowLog) {
  Archive-FileIfExists -PathValue $shadowLog -ArchiveDir $archiveDir
}

if ($FreshPaperLogs) {
  foreach ($file in Get-ChildItem -Path $logsDir -Filter 'trades_paper_*.csv' -File -ErrorAction SilentlyContinue) {
    Archive-FileIfExists -PathValue $file.FullName -ArchiveDir $archiveDir
  }
  foreach ($file in Get-ChildItem -Path $logsDir -Filter 'trades_closed_*.csv' -File -ErrorAction SilentlyContinue) {
    Archive-FileIfExists -PathValue $file.FullName -ArchiveDir $archiveDir
  }
  Archive-FileIfExists -PathValue $closedMaster -ArchiveDir $archiveDir
}

$writerOut = Join-Path $logsDir 'survival_active_paper_writer.out'
$writerErr = Join-Path $logsDir 'survival_active_paper_writer.err'
$executorOut = Join-Path $logsDir 'survival_active_paper_executor.out'
$executorErr = Join-Path $logsDir 'survival_active_paper_executor.err'
$writerLock = Join-Path $logsDir 'live_writer.lock'
$executorLock = Join-Path $logsDir 'live_executor.lock'
$writerLauncherPath = Join-Path $logsDir 'survival_active_writer_launcher.py'
$executorLauncherPath = Join-Path $logsDir 'survival_active_executor_launcher.py'
$reportJson = Join-Path $reportsDir 'survival_active_paper_summary.json'

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

Write-Host "[survival-active] Starting live_writer and live_executor in paper mode."
Write-Host "[survival-active] Experiment mode: $experimentMode + SURVIVAL_EXIT_ACTIVE=true"
Write-Host "[survival-active] Forced flags:"
foreach ($name in $forcedPaperEnv.Keys) {
  Write-Host ("  {0}={1}" -f $name, $forcedPaperEnv[$name])
}
Write-Host ("[survival-active] Duration: {0} minutes" -f $Minutes)

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
        Write-Host "[survival-active] REFUSING: $label exited early. Check $errPath" -ForegroundColor Red
        exit 1
      }
    }
    $remaining = [Math]::Ceiling(($endAt - (Get-Date)).TotalMinutes)
    Write-Host ("[survival-active] writer PID={0}; executor PID={1}; about {2} minute(s) remaining" -f `
      $writer.Id, $executor.Id, [Math]::Max(0, $remaining))
  }
}
finally {
  Stop-OwnedProcess -Proc $executor -LockPath $executorLock
  Stop-OwnedProcess -Proc $writer -LockPath $writerLock
}

Write-Host "[survival-active] Generating report..."
& $py "tools\experimental_shadow_report.py" --logs-dir $logsDir --json --json-out $reportJson
if ($LASTEXITCODE -ne 0) {
  Write-Host "[survival-active] FAIL: report generation failed." -ForegroundColor Red
  exit 1
}

Write-Host ""
Write-Host "[survival-active] Done."
Write-Host "  shadow_log: $shadowLog"
Write-Host "  report_json: $reportJson"
Write-Host "  writer_stdout: $writerOut"
Write-Host "  writer_stderr: $writerErr"
Write-Host "  executor_stdout: $executorOut"
Write-Host "  executor_stderr: $executorErr"
