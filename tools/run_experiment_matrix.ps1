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
  [switch]$DryRun,
  [ValidateRange(0, 300)] [int]$StaleEntryToleranceSeconds = 5
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

function ConvertTo-UtcDateTimeOffset([string]$Value) {
  if ([string]::IsNullOrWhiteSpace($Value)) { return $null }

  $parsed = [DateTimeOffset]::MinValue
  $styles = [Globalization.DateTimeStyles]::AllowWhiteSpaces -bor `
    [Globalization.DateTimeStyles]::AssumeUniversal -bor `
    [Globalization.DateTimeStyles]::AdjustToUniversal
  if ([DateTimeOffset]::TryParse(
      $Value,
      [Globalization.CultureInfo]::InvariantCulture,
      $styles,
      [ref]$parsed
    )) {
    return $parsed.ToUniversalTime()
  }
  return $null
}

function Test-MatrixStalePaperEntries {
  param(
    [string]$LogsDir,
    [DateTimeOffset]$RunStartedUtc,
    [int]$ToleranceSeconds = 5
  )

  $entryActions = @('BUY', 'SELL_SHORT')
  $cutoff = $RunStartedUtc.AddSeconds(-[double]$ToleranceSeconds)
  $contaminated = New-Object System.Collections.Generic.List[object]

  foreach ($file in Get-ChildItem -Path $LogsDir -Filter 'trades_paper_*.csv' -File -ErrorAction SilentlyContinue) {
    try {
      $rows = @(Import-Csv -LiteralPath $file.FullName)
    } catch {
      Write-Warning ("Could not inspect paper trade file {0}: {1}" -f $file.FullName, $_.Exception.Message)
      continue
    }

    foreach ($row in $rows) {
      $action = [string]$row.side
      if ([string]::IsNullOrWhiteSpace($action)) {
        $action = [string]$row.action
      }
      $action = $action.Trim().ToUpperInvariant()
      if ($entryActions -notcontains $action) { continue }

      $timestamp = [string]$row.ts
      if ([string]::IsNullOrWhiteSpace($timestamp)) {
        $timestamp = [string]$row.timestamp
      }
      $entryUtc = ConvertTo-UtcDateTimeOffset -Value $timestamp
      if ($null -eq $entryUtc) {
        Write-Warning ("Could not parse paper entry timestamp as UTC: file={0} timestamp={1}" -f $file.Name, $timestamp)
        continue
      }

      if ($entryUtc -lt $cutoff) {
        $contaminated.Add([pscustomobject]@{
          signal_id = [string]$row.signal_id
          timestamp = $timestamp
          timestamp_utc = $entryUtc.ToString('o')
          action = $action
          file = $file.FullName
        })
      }
    }
  }

  return [pscustomobject]@{
    checked = $true
    count = [int]$contaminated.Count
    signal_ids = @($contaminated | ForEach-Object { $_.signal_id })
    entries = @($contaminated | ForEach-Object { $_ })
  }
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

function Get-ReplayPaths([string]$ReportsDir, [string]$ModeName, [string]$Stamp) {
  return [ordered]@{
    contract = Join-Path $ReportsDir "matrix_${ModeName}_${Stamp}_replay_contract.json"
    model_snapshot = Join-Path $ReportsDir "matrix_${ModeName}_${Stamp}_model_serving_snapshot.json"
    bundle = Join-Path (Join-Path $ReportsDir 'replay_bundles') "${ModeName}_${Stamp}"
  }
}

function Get-ReplayExperimentMode([string]$ModeName) {
  switch ($ModeName) {
    'advanced_risk_shadow' { return 'advanced_risk_shadow_placeholder' }
    'survival_active' { return 'survival_shadow' }
    default { return $ModeName }
  }
}

function Get-ReplayForcedEnv([string]$Python, [string]$Root, [string]$ModeName) {
  $contractMode = Get-ReplayExperimentMode -ModeName $ModeName
  $overrides = Get-ForcedPaperEnv -Python $Python -Root $Root -ModeName $contractMode
  if ($ModeName -eq 'survival_active') {
    $overrides['SURVIVAL_EXIT_ACTIVE'] = 'true'
  }
  if ($ModeName -eq 'advanced_risk_shadow') {
    $overrides['USE_ADVANCED_RISK'] = 'true'
    $overrides['ADVANCED_RISK_ACTIVE'] = 'false'
  }
  return ,$overrides
}

function Invoke-ReplayContractCapture {
  param(
    [string]$Python,
    [string]$Root,
    [string]$LogsDir,
    [string]$Identity,
    [string]$ModeName,
    [DateTimeOffset]$RunStartedUtc,
    [DateTimeOffset]$ExpectedFinishedUtc,
    [string]$ContractPath
  )

  $forced = Get-ReplayForcedEnv -Python $Python -Root $Root -ModeName $ModeName
  $forcedPath = Join-Path $LogsDir "matrix_${ModeName}_replay_forced_env.json"
  Set-Content -Path $forcedPath -Value ($forced | ConvertTo-Json -Depth 4) -Encoding UTF8
  $raw = & $Python "tools\replay_contract.py" capture `
    --identity $Identity `
    --mode $ModeName `
    --forced-env-json $forcedPath `
    --base-dir $Root `
    --run-started-utc $RunStartedUtc.ToString('o') `
    --expected-finished-at $ExpectedFinishedUtc.ToString('o') `
    --json-out $ContractPath
  if ($LASTEXITCODE -ne 0) {
    throw "replay contract safety validation failed: $raw"
  }
  $contract = Get-Content -LiteralPath $ContractPath -Raw | ConvertFrom-Json
  return [pscustomobject]@{
    path = $ContractPath
    digest = [string]$contract.contract_digest
    status = 'exact_matrix_snapshot'
  }
}

function Invoke-ReplayBundleCapture {
  param(
    [string]$Python,
    [string]$Identity,
    [string]$RunStartedUtc,
    [string]$FinishedAt,
    [string]$ContractPath,
    [string]$ModelServingSnapshotPath,
    [string]$ReportsDir,
    [string]$LogsDir,
    [string]$BundleRoot
  )

  $raw = & $Python "tools\replay_bundle.py" build `
    --identity $Identity `
    --run-started-utc $RunStartedUtc `
    --finished-at $FinishedAt `
    --contract $ContractPath `
    --model-serving-snapshot $ModelServingSnapshotPath `
    --reports-dir $ReportsDir `
    --logs-dir $LogsDir `
    --bundle-root $BundleRoot
  if ($LASTEXITCODE -ne 0) {
    throw "replay bundle capture failed: $raw"
  }
  $bundle = $raw | Select-Object -Last 1 | ConvertFrom-Json
  return [pscustomobject]@{
    path = [string]$bundle.bundle_path
    digest = [string]$bundle.bundle_digest
    status = 'exact_bundle'
  }
}

function Invoke-ModelServingSnapshotCapture {
  param(
    [string]$Python,
    [string]$Root,
    [string]$LogsDir,
    [string]$Identity,
    [string]$ModeName,
    [string]$SnapshotPath
  )

  $forced = Get-ReplayForcedEnv -Python $Python -Root $Root -ModeName $ModeName
  $forcedPath = Join-Path $LogsDir "matrix_${ModeName}_model_serving_forced_env.json"
  Set-Content -Path $forcedPath -Value ($forced | ConvertTo-Json -Depth 4) -Encoding UTF8
  $raw = & $Python "tools\model_serving_snapshot.py" `
    --identity $Identity `
    --mode $ModeName `
    --forced-env-json $forcedPath `
    --base-dir $Root `
    --json-out $SnapshotPath
  if ($LASTEXITCODE -ne 0) {
    throw "model-serving snapshot safety validation failed: $raw"
  }
  $snapshot = Get-Content -LiteralPath $SnapshotPath -Raw | ConvertFrom-Json
  return [pscustomobject]@{
    path = $SnapshotPath
    digest = [string]$snapshot.snapshot_digest
    status = 'exact_matrix_snapshot'
    model_contract_status = [string]$snapshot.training_serving_contract_status
    model_contract_guard_digest = [string]$snapshot.model_serving_guard_digest
    model_contract_critical_mismatches = @($snapshot.training_serving_critical_mismatches)
  }
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
      replay_paths = Get-ReplayPaths -ReportsDir $ReportsDir -ModeName $ModeName -Stamp $Stamp
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
    replay_paths = Get-ReplayPaths -ReportsDir $ReportsDir -ModeName $ModeName -Stamp $Stamp
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
    $drySnapshotRaw = & $py "tools/model_serving_snapshot.py" `
      --identity "dry_run_model_contract" `
      --mode ([string]$plan.mode) `
      --base-dir $root `
      --paper-safe
    if ($LASTEXITCODE -ne 0) {
      throw "dry-run model contract preflight failed to inspect"
    }
    $drySnapshot = $drySnapshotRaw | ConvertFrom-Json
    Write-Host ""
    Write-Host "[matrix] Mode: $($plan.mode)"
    Write-Host "  command: $($plan.command)"
    Write-Host "  forced paper flags: LIVE_TRADING=false PAPER_TRADING=true LIVE_MODE=false EXEC_PAPER=true PLACE_REAL_ORDERS=false"
    Write-Host "  refuses real-order/live mode: true"
    Write-Host "  stale paper entry guard: enabled (tolerance=$StaleEntryToleranceSeconds second(s))"
    Write-Host "  replay contract: $($plan.replay_paths.contract)"
    Write-Host "  model-serving snapshot: $($plan.replay_paths.model_snapshot)"
    Write-Host "  model contract status: $($drySnapshot.training_serving_contract_status)"
    Write-Host "  model contract guard digest: $($drySnapshot.model_serving_guard_digest)"
    if ($drySnapshot.training_serving_contract_status -ne 'pass') {
      Write-Host "  model_contract_preflight_failed: $(@($drySnapshot.training_serving_critical_mismatches) -join '; ')"
    }
    Write-Host "  replay bundle: $($plan.replay_paths.bundle)"
    Write-Host "  replay capture creates files in non-dry-run mode only: true"
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
  $runStartedUtc = [DateTimeOffset]::UtcNow
  $started = $runStartedUtc.ToString('o')
  $expectedFinishedUtc = $runStartedUtc.AddMinutes([double]$Minutes)
  $finished = $null
  $replayIdentity = "${modeName}:${matrixStamp}"
  $replayContractPath = [string]$plan.replay_paths.contract
  $replayContractDigest = $null
  $replayContractStatus = 'not_captured'
  $modelServingSnapshotPath = [string]$plan.replay_paths.model_snapshot
  $modelServingSnapshotDigest = $null
  $modelServingSnapshotStatus = 'not_captured'
  $modelContractStatus = 'unverified'
  $modelContractGuardDigest = $null
  $modelContractCriticalMismatches = @()
  $replayBundlePath = [string]$plan.replay_paths.bundle
  $replayBundleDigest = $null
  $replayBundleStatus = 'not_captured'
  $notes = New-Object System.Collections.Generic.List[string]
  $exitStatus = 0
  $staleEntryGuardChecked = $false
  $staleEntryCount = 0
  $staleEntrySignalIds = @()
  Write-Host ""
  Write-Host "[matrix] ===== Mode: $modeName ====="
  try {
    try {
      $contractCapture = Invoke-ReplayContractCapture `
        -Python $py `
        -Root $root `
        -LogsDir $logsDir `
        -Identity $replayIdentity `
        -ModeName $modeName `
        -RunStartedUtc $runStartedUtc `
        -ExpectedFinishedUtc $expectedFinishedUtc `
        -ContractPath $replayContractPath
      $replayContractPath = [string]$contractCapture.path
      $replayContractDigest = [string]$contractCapture.digest
      $replayContractStatus = [string]$contractCapture.status
      Write-Host ("[matrix] replay_contract_status={0} digest={1}" -f `
        $replayContractStatus, $replayContractDigest)
    } catch {
      $replayContractStatus = 'failed'
      throw
    }

    try {
      $snapshotCapture = Invoke-ModelServingSnapshotCapture `
        -Python $py `
        -Root $root `
        -LogsDir $logsDir `
        -Identity $replayIdentity `
        -ModeName $modeName `
        -SnapshotPath $modelServingSnapshotPath
      $modelServingSnapshotPath = [string]$snapshotCapture.path
      $modelServingSnapshotDigest = [string]$snapshotCapture.digest
      $modelServingSnapshotStatus = [string]$snapshotCapture.status
      $modelContractStatus = [string]$snapshotCapture.model_contract_status
      $modelContractGuardDigest = [string]$snapshotCapture.model_contract_guard_digest
      $modelContractCriticalMismatches = @($snapshotCapture.model_contract_critical_mismatches)
      Write-Host ("[matrix] model_serving_snapshot_status={0} digest={1}" -f `
        $modelServingSnapshotStatus, $modelServingSnapshotDigest)
      Write-Host ("[matrix] model_contract_status={0} guard_digest={1}" -f `
        $modelContractStatus, $modelContractGuardDigest)
    } catch {
      $modelServingSnapshotStatus = 'failed'
      $notes.Add('model_serving_snapshot_capture_failed')
      throw
    }
    if ($modelContractStatus -ne 'pass') {
      $notes.Add('model_contract_preflight_failed')
      throw ("model contract preflight failed: {0}" -f ($modelContractCriticalMismatches -join '; '))
    }

    if ($FreshLogs) {
      Archive-MatrixLogs -LogsDir $logsDir -ModeName $modeName -Stamp $matrixStamp
      $notes.Add('fresh logs archived at matrix level')
    }

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
      Invoke-ChildRunbook -PowerShellExe $psExe -ScriptPath $plan.script -Arguments $plan.args -Root $root
      $code = [int]$LASTEXITCODE
      if ($code -ne 0) {
        $exitStatus = $code
        throw "child runbook exited with status $code"
      }
    }

    $staleGuard = Test-MatrixStalePaperEntries `
      -LogsDir $logsDir `
      -RunStartedUtc $runStartedUtc `
      -ToleranceSeconds $StaleEntryToleranceSeconds
    $staleEntryGuardChecked = [bool]$staleGuard.checked
    $staleEntryCount = [int]$staleGuard.count
    $staleEntrySignalIds = @($staleGuard.signal_ids)
    Write-Host ("[matrix] stale_entry_guard_checked=true stale_entry_count={0}" -f $staleEntryCount)

    if ($staleEntryCount -gt 0) {
      foreach ($entry in $staleGuard.entries) {
        $signalId = if ([string]::IsNullOrWhiteSpace([string]$entry.signal_id)) { '<missing>' } else { [string]$entry.signal_id }
        Write-Host ("[matrix] CONTAMINATED signal_id={0} timestamp={1} file={2}" -f `
          $signalId, $entry.timestamp, $entry.file) -ForegroundColor Red
      }
      $notes.Add('stale_signal_replay_or_prestart_entry_detected')
      $exitStatus = 1
    }

    Write-MatrixReports -Python $py -LogsDir $logsDir -ReportPaths $plan.report_paths
    $finished = (Get-Date).ToUniversalTime().ToString('o')

    try {
      $bundleCapture = Invoke-ReplayBundleCapture `
        -Python $py `
        -Identity $replayIdentity `
        -RunStartedUtc $started `
        -FinishedAt $finished `
        -ContractPath $replayContractPath `
        -ModelServingSnapshotPath $modelServingSnapshotPath `
        -ReportsDir $reportsDir `
        -LogsDir $logsDir `
        -BundleRoot (Join-Path $reportsDir 'replay_bundles')
      $replayBundlePath = [string]$bundleCapture.path
      $replayBundleDigest = [string]$bundleCapture.digest
      $replayBundleStatus = [string]$bundleCapture.status
      Write-Host ("[matrix] replay_bundle_status={0} digest={1}" -f `
        $replayBundleStatus, $replayBundleDigest)
    } catch {
      $replayBundleStatus = 'failed'
      $notes.Add('replay_bundle_capture_failed')
      $exitStatus = 1
      throw
    }

    if ($staleEntryCount -gt 0) {
      throw "stale paper entries detected before mode run start: $staleEntryCount"
    }
  } catch {
    if ($exitStatus -eq 0) {
      $exitStatus = 1
    }
    $notes.Add([string]$_.Exception.Message)
    Write-Host ("[matrix] FAIL: {0}: {1}" -f $modeName, $_.Exception.Message) -ForegroundColor Red
  }
  if ([string]::IsNullOrWhiteSpace([string]$finished)) {
    $finished = (Get-Date).ToUniversalTime().ToString('o')
  }
  $evidenceValid = [bool](
    ($exitStatus -eq 0) -and
    $staleEntryGuardChecked -and
    ($staleEntryCount -eq 0) -and
    ($replayContractStatus -eq 'exact_matrix_snapshot') -and
    ($modelServingSnapshotStatus -eq 'exact_matrix_snapshot') -and
    ($modelContractStatus -eq 'pass') -and
    ($replayBundleStatus -eq 'exact_bundle')
  )
  Write-Host ("[matrix] {0}: run_started_utc={1} stale_entry_count={2} evidence_valid={3}" -f `
    $modeName, $started, $staleEntryCount, $evidenceValid.ToString().ToLowerInvariant())
  $results += [ordered]@{
    run_timestamp = $started
    run_started_utc = $started
    finished_at = $finished
    mode = $modeName
    duration_minutes = $Minutes
    command_run = $plan.command
    report_paths = $plan.report_paths
    exit_status = $exitStatus
    stale_entry_guard_checked = [bool]$staleEntryGuardChecked
    stale_entry_count = [int]$staleEntryCount
    stale_entry_signal_ids = @($staleEntrySignalIds)
    evidence_valid = $evidenceValid
    replay_contract_path = $replayContractPath
    replay_contract_digest = $replayContractDigest
    replay_contract_status = $replayContractStatus
    model_serving_snapshot_path = $modelServingSnapshotPath
    model_serving_snapshot_digest = $modelServingSnapshotDigest
    model_serving_snapshot_status = $modelServingSnapshotStatus
    model_contract_status = $modelContractStatus
    model_contract_guard_digest = $modelContractGuardDigest
    model_contract_critical_mismatches = @($modelContractCriticalMismatches)
    replay_bundle_path = $replayBundlePath
    replay_bundle_digest = $replayBundleDigest
    replay_bundle_status = $replayBundleStatus
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
  stale_entry_tolerance_seconds = $StaleEntryToleranceSeconds
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
