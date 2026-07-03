<#
Usage:
  .\tools\run_experiment_matrix.ps1 -Mode combined_shadow -Minutes 30 -FreshLogs
  .\tools\run_experiment_matrix.ps1 -All -Minutes 30 -FreshLogs
  .\tools\run_experiment_matrix.ps1 -Mode xgboost_shadow_outcome -Minutes 5 -DryRun
#>

param(
  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,
  [string]$Mode,
  [switch]$All,
  [switch]$FreshLogs,
  [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$SupportedModes = @(
  'baseline',
  'iforest_shadow',
  'iforest_blocking',
  'xgboost_shadow_outcome',
  'survival_shadow',
  'survival_active',
  'advanced_risk_shadow',
  'combined_shadow'
)

$GenericExperimentModes = @{
  baseline = 'baseline'
  iforest_shadow = 'iforest_shadow'
  survival_shadow = 'survival_shadow'
}

$RunbookModes = @{
  iforest_blocking = @{
    Script = 'run_isolation_forest_blocking_paper_test.ps1'
    FreshShadowSwitch = '-FreshShadowLog'
    FreshPaperSwitch = $null
    XGBoostAudit = $false
  }
  xgboost_shadow_outcome = @{
    Script = 'run_xgboost_shadow_outcome_paper_test.ps1'
    FreshShadowSwitch = '-FreshShadowLog'
    FreshPaperSwitch = '-FreshPaperLogs'
    XGBoostAudit = $true
  }
  survival_active = @{
    Script = 'run_survival_active_paper_test.ps1'
    FreshShadowSwitch = '-FreshShadowLog'
    FreshPaperSwitch = '-FreshPaperLogs'
    XGBoostAudit = $false
  }
  advanced_risk_shadow = @{
    Script = 'run_advanced_risk_shadow_paper_test.ps1'
    FreshShadowSwitch = '-FreshShadowLog'
    FreshPaperSwitch = '-FreshPaperLogs'
    XGBoostAudit = $false
  }
  combined_shadow = @{
    Script = 'run_combined_shadow_paper_test.ps1'
    FreshShadowSwitch = '-FreshShadowLogs'
    FreshPaperSwitch = '-FreshPaperLogs'
    XGBoostAudit = $true
  }
}

function Quote-ProcessArg([string]$Value) {
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Escape-Regex([string]$Value) {
  return [regex]::Escape($Value)
}

function Get-PythonExe([string]$Root) {
  $py = Join-Path $Root '.venv\Scripts\python.exe'
  if (Test-Path $py) { return $py }
  return 'python'
}

function Get-PowerShellExe {
  if ($PSVersionTable.PSEdition -eq 'Core') {
    $cmd = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
  }
  $proc = Get-Process -Id $PID -ErrorAction SilentlyContinue
  if ($proc -and $proc.Path) { return $proc.Path }
  $win = Get-Command powershell -ErrorAction SilentlyContinue
  if ($win) { return $win.Source }
  $core = Get-Command pwsh -ErrorAction SilentlyContinue
  if ($core) { return $core.Source }
  throw 'Could not locate a PowerShell executable for child runbooks.'
}

function Get-ScopedLiveProcess([string]$RootDir) {
  $rootRx = Escape-Regex $RootDir
  Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -and
    ($_.CommandLine -match $rootRx) -and
    (($_.CommandLine -match 'tools(\\|/)live_(writer|executor)\.py') -or
     ($_.CommandLine -match 'matrix_(baseline|iforest_shadow|survival_shadow)_(writer|executor)_launcher\.py'))
  }
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

function Archive-FileIfExists([string]$PathValue, [string]$ArchiveDir) {
  if (-not (Test-Path $PathValue)) { return }
  New-Item -ItemType Directory -Path $ArchiveDir -Force | Out-Null
  $dest = Join-Path $ArchiveDir ([System.IO.Path]::GetFileName($PathValue))
  Move-Item -LiteralPath $PathValue -Destination $dest -Force
  Write-Host "[matrix] Archived $PathValue to $dest"
}

function Archive-MatrixLogs([string]$LogsDir, [string]$ModeName, [string]$Stamp) {
  $archiveDir = Join-Path $LogsDir "matrix_${ModeName}_archive_$Stamp"
  foreach ($name in @(
    'isolation_forest_shadow.csv',
    'xgboost_signal_shadow.csv',
    'survival_exit_shadow.csv',
    'advanced_risk_shadow.csv'
  )) {
    Archive-FileIfExists -PathValue (Join-Path $LogsDir $name) -ArchiveDir $archiveDir
  }
  foreach ($file in Get-ChildItem -Path $LogsDir -Filter 'trades_paper_*.csv' -File -ErrorAction SilentlyContinue) {
    Archive-FileIfExists -PathValue $file.FullName -ArchiveDir $archiveDir
  }
  foreach ($file in Get-ChildItem -Path $LogsDir -Filter 'trades_closed_*.csv' -File -ErrorAction SilentlyContinue) {
    Archive-FileIfExists -PathValue $file.FullName -ArchiveDir $archiveDir
  }
  Archive-FileIfExists -PathValue (Join-Path $LogsDir 'trades_closed.csv') -ArchiveDir $archiveDir
}

function Assert-MatrixSelection {
  if (-not $All -and [string]::IsNullOrWhiteSpace($Mode)) {
    Write-Host "ERROR: specify either -Mode <mode> or -All." -ForegroundColor Red
    exit 2
  }
  if ($All -and -not [string]::IsNullOrWhiteSpace($Mode)) {
    Write-Host "ERROR: use either -Mode <mode> or -All, not both." -ForegroundColor Red
    exit 2
  }
  if (-not [string]::IsNullOrWhiteSpace($Mode) -and ($SupportedModes -notcontains $Mode)) {
    Write-Host ("ERROR: invalid mode '{0}'. Supported modes: {1}" -f $Mode, ($SupportedModes -join ', ')) -ForegroundColor Red
    exit 2
  }
}

function Get-MatrixReportPaths([string]$ReportsDir, [string]$ModeName, [string]$Stamp, [bool]$HasAudit) {
  $paths = [ordered]@{
    unified = Join-Path $ReportsDir "matrix_${ModeName}_${Stamp}_unified.json"
    shadow_summary = Join-Path $ReportsDir "matrix_${ModeName}_${Stamp}_shadow_summary.json"
  }
  if ($HasAudit) {
    $paths['xgboost_audit'] = Join-Path $ReportsDir "matrix_${ModeName}_${Stamp}_xgboost_audit.json"
  }
  return $paths
}

function Get-ModePlan([string]$ModeName, [int]$Duration, [string]$ScriptDir, [string]$ReportsDir, [string]$Stamp) {
  $hasAudit = $false
  if ($RunbookModes.ContainsKey($ModeName)) {
    $meta = $RunbookModes[$ModeName]
    $args = @('-Minutes', [string]$Duration)
    if ($FreshLogs) {
      if ($meta.FreshShadowSwitch) { $args += $meta.FreshShadowSwitch }
      if ($meta.FreshPaperSwitch) { $args += $meta.FreshPaperSwitch }
    }
    $scriptPath = Join-Path $ScriptDir $meta.Script
    $command = "& `"$scriptPath`" " + ($args -join ' ')
    $hasAudit = [bool]$meta.XGBoostAudit
    return [ordered]@{
      mode = $ModeName
      kind = 'runbook'
      script = $scriptPath
      args = $args
      command = $command
      report_paths = Get-MatrixReportPaths -ReportsDir $ReportsDir -ModeName $ModeName -Stamp $Stamp -HasAudit $hasAudit
    }
  }

  $experimentMode = $GenericExperimentModes[$ModeName]
  $command = "generic paper writer+executor with experiment_mode=$experimentMode -Minutes $Duration"
  return [ordered]@{
    mode = $ModeName
    kind = 'generic'
    experiment_mode = $experimentMode
    script = ''
    args = @()
    command = $command
    report_paths = Get-MatrixReportPaths -ReportsDir $ReportsDir -ModeName $ModeName -Stamp $Stamp -HasAudit $false
  }
}

function Get-ForcedPaperEnv([string]$Python, [string]$Root, [string]$ModeName) {
  $overrides = Get-ExperimentModeOverrides -Python $Python -Root $Root -Mode $ModeName
  $overrides['LIVE_TRADING'] = 'false'
  $overrides['PAPER_TRADING'] = 'true'
  $overrides['LIVE_MODE'] = 'false'
  $overrides['EXEC_PAPER'] = 'true'
  $overrides['PLACE_REAL_ORDERS'] = 'false'
  $overrides['CONFIRM_LIVE_TRADING'] = ''
  $overrides['EXEC_RESTORE_STATE'] = 'false'
  return ,$overrides
}

function Invoke-MatrixPreflight([string]$Python, [string]$Root, [string]$ModeEnvJson) {
  $code = @'
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
os.environ["LIVE_TRADING"] = "false"
os.environ["PAPER_TRADING"] = "true"
os.environ["LIVE_MODE"] = "false"
os.environ["EXEC_PAPER"] = "true"
os.environ["PLACE_REAL_ORDERS"] = "false"
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
    "guardrail_mode": d.mode.value,
    "place_real_orders": d.place_real_orders,
    "hl_testnet": s.hl_testnet,
}))
'@
  $preflightPath = Join-Path (Split-Path -Parent $ModeEnvJson) 'matrix_preflight.py'
  Set-Content -Path $preflightPath -Value $code -Encoding UTF8
  $raw = & $Python $preflightPath $Root $ModeEnvJson
  if ($LASTEXITCODE -ne 0) {
    throw 'matrix mode preflight failed'
  }
  $preflight = $raw | ConvertFrom-Json
  if ($preflight.refuse) {
    $reasonText = ($preflight.reasons -join '; ')
    throw "REFUSING: live/mainnet/real-order mode detected: $reasonText"
  }
  Write-Host ("[matrix] Mode preflight OK: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} paper_trading={4}" -f `
    $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.paper_trading) -ForegroundColor Green
}

function Invoke-GenericPaperMode {
  param(
    [string]$ModeName,
    [string]$ExperimentMode,
    [int]$Duration,
    [string]$Root,
    [string]$LogsDir,
    [string]$Python,
    [string]$Stamp
  )

  $existing = Get-ScopedLiveProcess -RootDir $Root
  if ($existing) {
    $existing | Select-Object ProcessId, CommandLine | Format-Table -AutoSize | Out-String | Write-Host
    throw 'REFUSING: live writer/executor process is already running under this repo.'
  }

  if ($FreshLogs) {
    Archive-MatrixLogs -LogsDir $LogsDir -ModeName $ModeName -Stamp $Stamp
  }

  $forcedPaperEnv = Get-ForcedPaperEnv -Python $Python -Root $Root -ModeName $ExperimentMode
  $envPath = Join-Path $LogsDir "matrix_${ModeName}_mode_env.json"
  Set-Content -Path $envPath -Value ($forcedPaperEnv | ConvertTo-Json -Depth 4) -Encoding UTF8
  Invoke-MatrixPreflight -Python $Python -Root $Root -ModeEnvJson $envPath

  $writerOut = Join-Path $LogsDir "matrix_${ModeName}_writer.out"
  $writerErr = Join-Path $LogsDir "matrix_${ModeName}_writer.err"
  $executorOut = Join-Path $LogsDir "matrix_${ModeName}_executor.out"
  $executorErr = Join-Path $LogsDir "matrix_${ModeName}_executor.err"
  $writerLock = Join-Path $LogsDir 'live_writer.lock'
  $executorLock = Join-Path $LogsDir 'live_executor.lock'
  $writerLauncherPath = Join-Path $LogsDir "matrix_${ModeName}_writer_launcher.py"
  $executorLauncherPath = Join-Path $LogsDir "matrix_${ModeName}_executor_launcher.py"

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
os.environ["LIVE_TRADING"] = "false"
os.environ["PAPER_TRADING"] = "true"
os.environ["LIVE_MODE"] = "false"
os.environ["EXEC_PAPER"] = "true"
os.environ["PLACE_REAL_ORDERS"] = "false"
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

FORCED["LIVE_TRADING"] = "false"
FORCED["PAPER_TRADING"] = "true"
FORCED["LIVE_MODE"] = "false"
FORCED["EXEC_PAPER"] = "true"
FORCED["PLACE_REAL_ORDERS"] = "false"
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

  Write-Host "[matrix] Starting generic paper mode: $ModeName ($ExperimentMode)"
  $writerArgs = @($writerLauncherPath, $Root, $envPath) | ForEach-Object { Quote-ProcessArg $_ }
  $executorArgs = @($executorLauncherPath, $Root, $envPath) | ForEach-Object { Quote-ProcessArg $_ }
  $writer = $null
  $executor = $null
  try {
    $writer = Start-Process -FilePath $Python `
      -ArgumentList ($writerArgs -join ' ') `
      -WorkingDirectory $Root `
      -RedirectStandardOutput $writerOut `
      -RedirectStandardError $writerErr `
      -PassThru -WindowStyle Hidden

    Start-Sleep -Seconds 5

    $executor = Start-Process -FilePath $Python `
      -ArgumentList ($executorArgs -join ' ') `
      -WorkingDirectory $Root `
      -RedirectStandardOutput $executorOut `
      -RedirectStandardError $executorErr `
      -PassThru -WindowStyle Hidden

    $endAt = (Get-Date).AddMinutes($Duration)
    while ((Get-Date) -lt $endAt) {
      Start-Sleep -Seconds 30
      foreach ($procInfo in @(@("writer", $writer, $writerErr), @("executor", $executor, $executorErr))) {
        $label = $procInfo[0]
        $proc = $procInfo[1]
        $errPath = $procInfo[2]
        try {
          $null = Get-Process -Id $proc.Id -ErrorAction Stop
        } catch {
          throw "$label exited early. Check $errPath"
        }
      }
      $remaining = [Math]::Ceiling(($endAt - (Get-Date)).TotalMinutes)
      Write-Host ("[matrix] {0}: writer PID={1}; executor PID={2}; about {3} minute(s) remaining" -f `
        $ModeName, $writer.Id, $executor.Id, [Math]::Max(0, $remaining))
    }
  }
  finally {
    Stop-OwnedProcess -Proc $executor -LockPath $executorLock
    Stop-OwnedProcess -Proc $writer -LockPath $writerLock
  }
}

function Invoke-ChildRunbook {
  param(
    [string]$PowerShellExe,
    [string]$ScriptPath,
    [array]$Arguments,
    [string]$Root
  )

  $childArgs = @('-NoProfile')
  if ($PowerShellExe -match 'powershell(\.exe)?$') {
    $childArgs += @('-ExecutionPolicy', 'Bypass')
  }
  $childArgs += @('-File', $ScriptPath)
  $childArgs += $Arguments
  & $PowerShellExe @childArgs
}

function Write-MatrixReports {
  param(
    [string]$Python,
    [string]$LogsDir,
    [System.Collections.IDictionary]$ReportPaths
  )

  & $Python "tools\experimental_shadow_report.py" --logs-dir $LogsDir --json --json-out $ReportPaths.shadow_summary
  if ($LASTEXITCODE -ne 0) { throw 'experimental shadow report failed' }

  & $Python "tools\unified_experimental_report.py" --logs-dir $LogsDir --json --json-out $ReportPaths.unified
  if ($LASTEXITCODE -ne 0) { throw 'unified experimental report failed' }

  if ($ReportPaths.Contains('xgboost_audit')) {
    & $Python "tools\audit_xgboost_rejections.py" --json --json-out $ReportPaths.xgboost_audit
    if ($LASTEXITCODE -ne 0) { throw 'xgboost audit report failed' }
  }
}

Assert-MatrixSelection

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$root = (Resolve-Path (Join-Path $scriptDir '..')).Path
Set-Location $root
$logsDir = Join-Path $root 'logs'
$reportsDir = Join-Path $root 'reports'
$py = Get-PythonExe -Root $root
$psExe = Get-PowerShellExe
. (Join-Path $scriptDir 'apply_experiment_mode.ps1')

$matrixStamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss')
$modesToRun = if ($All) { $SupportedModes } else { @($Mode) }
$plans = @()
foreach ($modeName in $modesToRun) {
  $plans += Get-ModePlan -ModeName $modeName -Duration $Minutes -ScriptDir $scriptDir -ReportsDir $reportsDir -Stamp $matrixStamp
}

if ($DryRun) {
  Write-Host "[matrix] DRY RUN ONLY - no writer/executor processes will be started."
  Write-Host "[matrix] Matrix index would be: $(Join-Path $reportsDir "matrix_index_${matrixStamp}.json")"
  foreach ($plan in $plans) {
    Write-Host ""
    Write-Host "[matrix] Mode: $($plan.mode)"
    Write-Host "  command: $($plan.command)"
    Write-Host "  forced paper flags: LIVE_TRADING=false PAPER_TRADING=true LIVE_MODE=false EXEC_PAPER=true PLACE_REAL_ORDERS=false"
    Write-Host "  refuses real-order/live mode: true"
    Write-Host "  reports:"
    foreach ($key in $plan.report_paths.Keys) {
      Write-Host ("    {0}: {1}" -f $key, $plan.report_paths[$key])
    }
  }
  exit 0
}

New-Item -ItemType Directory -Path $logsDir -Force | Out-Null
New-Item -ItemType Directory -Path $reportsDir -Force | Out-Null

$results = @()
foreach ($plan in $plans) {
  $modeName = [string]$plan.mode
  $started = (Get-Date).ToUniversalTime().ToString('o')
  $notes = New-Object System.Collections.Generic.List[string]
  $exitStatus = 0
  Write-Host ""
  Write-Host "[matrix] ===== Mode: $modeName ====="
  try {
    if ($plan.kind -eq 'generic') {
      Invoke-GenericPaperMode `
        -ModeName $modeName `
        -ExperimentMode $plan.experiment_mode `
        -Duration $Minutes `
        -Root $root `
        -LogsDir $logsDir `
        -Python $py `
        -Stamp $matrixStamp
    } else {
      if ($FreshLogs) {
        $notes.Add('fresh logs delegated to mode runbook where supported')
      }
      Invoke-ChildRunbook -PowerShellExe $psExe -ScriptPath $plan.script -Arguments $plan.args -Root $root
      $code = [int]$LASTEXITCODE
      if ($code -ne 0) {
        $exitStatus = $code
        throw "child runbook exited with status $code"
      }
    }
    Write-MatrixReports -Python $py -LogsDir $logsDir -ReportPaths $plan.report_paths
  } catch {
    if ($exitStatus -eq 0) {
      $exitStatus = 1
    }
    $notes.Add([string]$_.Exception.Message)
    Write-Host ("[matrix] FAIL: {0}: {1}" -f $modeName, $_.Exception.Message) -ForegroundColor Red
  }
  $finished = (Get-Date).ToUniversalTime().ToString('o')
  $results += [ordered]@{
    run_timestamp = $started
    finished_at = $finished
    mode = $modeName
    duration_minutes = $Minutes
    command_run = $plan.command
    report_paths = $plan.report_paths
    exit_status = $exitStatus
    notes = @($notes)
  }
}

$index = [ordered]@{
  matrix_timestamp = $matrixStamp
  generated_at = (Get-Date).ToUniversalTime().ToString('o')
  requested_mode = if ($All) { 'ALL' } else { $Mode }
  all = [bool]$All
  duration_minutes = $Minutes
  fresh_logs = [bool]$FreshLogs
  dry_run = [bool]$DryRun
  runs = $results
}
$indexPath = Join-Path $reportsDir "matrix_index_${matrixStamp}.json"
Set-Content -Path $indexPath -Value ($index | ConvertTo-Json -Depth 8) -Encoding UTF8
Write-Host ""
Write-Host "[matrix] Index written: $indexPath"

$failed = @($results | Where-Object { $_.exit_status -ne 0 })
if ($failed.Count -gt 0) {
  Write-Host ("[matrix] Completed with {0} failed mode(s)." -f $failed.Count) -ForegroundColor Red
  exit 1
}

Write-Host "[matrix] Completed successfully." -ForegroundColor Green
