<# 
Usage:
  .\tools\run_xgboost_shadow_outcome_paper_test.ps1 -Minutes 5
  .\tools\run_xgboost_shadow_outcome_paper_test.ps1 -Minutes 30
  .\tools\run_xgboost_shadow_outcome_paper_test.ps1 -Minutes 60 -FreshShadowLog -FreshPaperLogs
#>

param(
  [ValidateSet(5, 30, 60)] [int]$Minutes = 30,
  [string]$Artifact = "model_artifacts/xgboost_signal.joblib",
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
     ($_.CommandLine -match 'xgboost_(lineage|shadow_outcome)_(writer|executor)_launcher\.py') -or
     ($_.CommandLine -match 'xgboost_blocking_writer_launcher\.py'))
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

function Get-CsvGlobRowCount([string]$Dir, [string]$Pattern) {
  $total = 0
  foreach ($file in Get-ChildItem -Path $Dir -Filter $Pattern -File -ErrorAction SilentlyContinue) {
    $total += Get-CsvDataRowCount -PathValue $file.FullName
  }
  return $total
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
  Write-Host "[xgboost-shadow-outcome] Archived $PathValue to $dest"
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
  Write-Host "[xgboost-shadow-outcome] REFUSING: artifact not found: $artifactFull" -ForegroundColor Red
  exit 1
}

$experimentMode = 'xgboost_shadow_outcome'
$forcedPaperEnv = Get-ExperimentModeOverrides -Python $py -Root $root -Mode $experimentMode
$forcedPaperEnv['XGBOOST_SIGNAL_ARTIFACT'] = $artifactFull
$forcedPaperEnv['EXEC_RESTORE_STATE'] = 'false'
Set-ExperimentModeEnvironment -Overrides $forcedPaperEnv
Set-Item -Path "Env:CONFIRM_LIVE_TRADING" -Value ""
$forcedEnvPath = Join-Path $logsDir 'xgboost_shadow_outcome_mode_env.json'
Set-Content -Path $forcedEnvPath -Value ($forcedPaperEnv | ConvertTo-Json -Depth 3) -Encoding UTF8

$existing = Get-ScopedLiveProcess -RootDir $root
if ($existing) {
  Write-Host "[xgboost-shadow-outcome] REFUSING: live writer/executor process is already running under this repo." -ForegroundColor Red
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

$preflightPath = Join-Path $logsDir 'xgboost_shadow_outcome_preflight.py'
Set-Content -Path $preflightPath -Value $preflightCode -Encoding UTF8
$preflightRaw = & $py $preflightPath $root $forcedEnvPath
if ($LASTEXITCODE -ne 0) {
  Write-Host "[xgboost-shadow-outcome] REFUSING: mode preflight failed." -ForegroundColor Red
  exit 1
}
$preflight = $preflightRaw | ConvertFrom-Json
if ($preflight.refuse) {
  Write-Host "[xgboost-shadow-outcome] REFUSING: live/mainnet mode detected." -ForegroundColor Red
  foreach ($reason in $preflight.reasons) {
    Write-Host "  - $reason" -ForegroundColor Yellow
  }
  Write-Host ("  effective: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
    $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading)
  exit 1
}

Write-Host ("[xgboost-shadow-outcome] Mode preflight OK: exchange={0} env={1} guardrail_mode={2} place_real_orders={3} hl_testnet={4} paper_trading={5}" -f `
  $preflight.exchange, $preflight.environment, $preflight.guardrail_mode, $preflight.place_real_orders, $preflight.hl_testnet, $preflight.paper_trading) -ForegroundColor Green

if (-not $SkipVerifier) {
  Write-Host "[xgboost-shadow-outcome] Verifying missing artifact path safety..."
  & $py "tools\verify_xgboost_signal.py" --artifact $artifactFull --missing-artifact-check
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[xgboost-shadow-outcome] REFUSING: missing artifact verifier failed." -ForegroundColor Red
    exit 1
  }

  Write-Host "[xgboost-shadow-outcome] Running deterministic XGBoost verifier..."
  & $py "tools\verify_xgboost_signal.py" --artifact $artifactFull
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[xgboost-shadow-outcome] REFUSING: verifier failed." -ForegroundColor Red
    exit 1
  }
}

$stamp = (Get-Date).ToUniversalTime().ToString('yyyyMMddHHmmss')
$archiveDir = Join-Path $logsDir "xgboost_shadow_outcome_archive_$stamp"
$shadowLog = Join-Path $logsDir 'xgboost_signal_shadow.csv'
$signalsLog = Join-Path $logsDir 'live_signals.csv'
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

$shadowRowsBefore = Get-CsvDataRowCount -PathValue $shadowLog
$signalRowsBefore = Get-CsvDataRowCount -PathValue $signalsLog
$paperRowsBefore = Get-CsvGlobRowCount -Dir $logsDir -Pattern 'trades_paper_*.csv'
$closedMasterRowsBefore = Get-CsvDataRowCount -PathValue $closedMaster
$closedDatedRowsBefore = Get-CsvGlobRowCount -Dir $logsDir -Pattern 'trades_closed_*.csv'

$writerOut = Join-Path $logsDir 'xgboost_shadow_outcome_paper_writer.out'
$writerErr = Join-Path $logsDir 'xgboost_shadow_outcome_paper_writer.err'
$executorOut = Join-Path $logsDir 'xgboost_shadow_outcome_paper_executor.out'
$executorErr = Join-Path $logsDir 'xgboost_shadow_outcome_paper_executor.err'
$writerLock = Join-Path $logsDir 'live_writer.lock'
$executorLock = Join-Path $logsDir 'live_executor.lock'
$writerLauncherPath = Join-Path $logsDir 'xgboost_shadow_outcome_writer_launcher.py'
$executorLauncherPath = Join-Path $logsDir 'xgboost_shadow_outcome_executor_launcher.py'
$lineageCheckPath = Join-Path $logsDir 'xgboost_shadow_outcome_check.py'
$auditJson = Join-Path $reportsDir 'xgboost_shadow_outcome_paper_audit.json'

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

$lineageCheckCode = @'
import csv
import glob
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
logs = root / "logs"
shadow_before = int(sys.argv[2])
signals_before = int(sys.argv[3])
paper_before = int(sys.argv[4])
closed_before = int(sys.argv[5])
closed_dated_before = int(sys.argv[6])

def rows(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return [], []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)

def glob_rows(pattern):
    out = []
    fields = set()
    for name in sorted(glob.glob(str(logs / pattern))):
        header, data = rows(name)
        fields.update(header)
        out.extend(data)
    return fields, out

def truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

shadow_header, shadow_rows = rows(logs / "xgboost_signal_shadow.csv")
signals_header, signal_rows = rows(logs / "live_signals.csv")
paper_fields, paper_rows = glob_rows("trades_paper_*.csv")
closed_header, closed_rows = rows(logs / "trades_closed.csv")
closed_dated_fields, closed_dated_rows = glob_rows("trades_closed_*.csv")

shadow_new = shadow_rows[shadow_before:]
signals_new = signal_rows[signals_before:]
paper_new = paper_rows[paper_before:]
closed_new = closed_rows[closed_before:]
closed_dated_new = closed_dated_rows[closed_dated_before:]

def count_id(data):
    return sum(1 for row in data if (row.get("signal_id") or row.get("decision_id") or "").strip())

print(json.dumps({
    "shadow_new_rows": len(shadow_new),
    "shadow_signal_id_count": count_id(shadow_new),
    "shadow_has_signal_id_column": "signal_id" in shadow_header or "decision_id" in shadow_header,
    "shadow_actually_rejected_count": sum(1 for row in shadow_new if truthy(row.get("actually_rejected"))),
    "shadow_would_reject_count": sum(1 for row in shadow_new if truthy(row.get("would_reject"))),
    "shadow_would_confirm_count": sum(1 for row in shadow_new if truthy(row.get("would_confirm"))),
    "live_signal_new_rows": len(signals_new),
    "live_signal_id_count": count_id(signals_new),
    "live_has_signal_id_column": "signal_id" in signals_header or "decision_id" in signals_header,
    "paper_new_rows": len(paper_new),
    "paper_signal_id_count": count_id(paper_new),
    "paper_has_signal_id_column": "signal_id" in paper_fields or "decision_id" in paper_fields,
    "closed_new_rows": len(closed_new) + len(closed_dated_new),
    "closed_signal_id_count": count_id(closed_new) + count_id(closed_dated_new),
    "closed_master_new_rows": len(closed_new),
    "closed_master_signal_id_count": count_id(closed_new),
    "closed_master_has_signal_id_column": "signal_id" in closed_header or "decision_id" in closed_header,
    "closed_dated_new_rows": len(closed_dated_new),
    "closed_dated_signal_id_count": count_id(closed_dated_new),
    "closed_dated_has_signal_id_column": "signal_id" in closed_dated_fields or "decision_id" in closed_dated_fields,
}))
'@

Set-Content -Path $writerLauncherPath -Value $writerLauncherCode -Encoding UTF8
Set-Content -Path $executorLauncherPath -Value $executorLauncherCode -Encoding UTF8
Set-Content -Path $lineageCheckPath -Value $lineageCheckCode -Encoding UTF8

Write-Host "[xgboost-shadow-outcome] Starting live_writer and live_executor in paper mode."
Write-Host "[xgboost-shadow-outcome] Experiment mode: $experimentMode"
Write-Host "[xgboost-shadow-outcome] Forced flags:"
foreach ($name in $forcedPaperEnv.Keys) {
  Write-Host ("  {0}={1}" -f $name, $forcedPaperEnv[$name])
}
Write-Host ("[xgboost-shadow-outcome] Duration: {0} minutes" -f $Minutes)

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
        Write-Host "[xgboost-shadow-outcome] REFUSING: $label exited early. Check $errPath" -ForegroundColor Red
        exit 1
      }
    }
    $remaining = [Math]::Ceiling(($endAt - (Get-Date)).TotalMinutes)
    Write-Host ("[xgboost-shadow-outcome] writer PID={0}; executor PID={1}; about {2} minute(s) remaining" -f `
      $writer.Id, $executor.Id, [Math]::Max(0, $remaining))
  }
}
finally {
  Stop-OwnedProcess -Proc $executor -LockPath $executorLock
  Stop-OwnedProcess -Proc $writer -LockPath $writerLock
}

$lineageRaw = & $py $lineageCheckPath $root $shadowRowsBefore $signalRowsBefore $paperRowsBefore $closedMasterRowsBefore $closedDatedRowsBefore
if ($LASTEXITCODE -ne 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: lineage log inspection failed." -ForegroundColor Red
  exit 1
}
$lineage = $lineageRaw | ConvertFrom-Json

if ($lineage.shadow_new_rows -le 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: xgboost_signal_shadow.csv got no new rows." -ForegroundColor Red
  Write-Host "Check $writerOut and $writerErr"
  exit 1
}
if (-not $lineage.shadow_has_signal_id_column -or $lineage.shadow_signal_id_count -le 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: new XGBoost shadow rows did not include signal_id." -ForegroundColor Red
  exit 1
}
if ($lineage.shadow_actually_rejected_count -ne 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: actually_rejected_count should be 0 because XGBOOST_SIGNAL_BLOCKING=false." -ForegroundColor Red
  exit 1
}
if ($lineage.live_signal_new_rows -le 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: live_signals.csv got no new rows." -ForegroundColor Red
  Write-Host "Check $writerOut and $writerErr"
  exit 1
}
if (-not $lineage.live_has_signal_id_column -or $lineage.live_signal_id_count -le 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: new live_signals rows did not include signal_id." -ForegroundColor Red
  exit 1
}
if ($lineage.paper_new_rows -gt 0) {
  if (-not $lineage.paper_has_signal_id_column -or $lineage.paper_signal_id_count -le 0) {
    Write-Host "[xgboost-shadow-outcome] FAIL: paper trade rows occurred but no signal_id was logged." -ForegroundColor Red
    exit 1
  }
  Write-Host ("[xgboost-shadow-outcome] Paper trades OK: {0} new row(s), {1} with signal_id." -f `
    $lineage.paper_new_rows, $lineage.paper_signal_id_count) -ForegroundColor Green
} else {
  Write-Host "[xgboost-shadow-outcome] No paper trades occurred during this window; this is not a failure." -ForegroundColor Yellow
}
if ($lineage.closed_new_rows -gt 0) {
  if ($lineage.closed_master_new_rows -gt 0 -and (-not $lineage.closed_master_has_signal_id_column -or $lineage.closed_master_signal_id_count -le 0)) {
    Write-Host "[xgboost-shadow-outcome] FAIL: aggregate closed trade rows occurred but no signal_id was logged." -ForegroundColor Red
    exit 1
  }
  if ($lineage.closed_dated_new_rows -gt 0 -and (-not $lineage.closed_dated_has_signal_id_column -or $lineage.closed_dated_signal_id_count -le 0)) {
    Write-Host "[xgboost-shadow-outcome] FAIL: dated closed trade rows occurred but no signal_id was logged." -ForegroundColor Red
    exit 1
  }
  Write-Host ("[xgboost-shadow-outcome] Closed trades OK: master={0}/{1} with signal_id, dated={2}/{3} with signal_id." -f `
    $lineage.closed_master_signal_id_count, $lineage.closed_master_new_rows, `
    $lineage.closed_dated_signal_id_count, $lineage.closed_dated_new_rows) -ForegroundColor Green
} else {
  Write-Host "[xgboost-shadow-outcome] No trades closed during this window; this is not a failure." -ForegroundColor Yellow
}

Write-Host ("[xgboost-shadow-outcome] Shadow outcome rows OK: shadow_new_rows={0}, would_confirm_count={1}, would_reject_count={2}, actually_rejected_count={3}" -f `
  $lineage.shadow_new_rows, $lineage.shadow_would_confirm_count, $lineage.shadow_would_reject_count, $lineage.shadow_actually_rejected_count) -ForegroundColor Green
Write-Host "[xgboost-shadow-outcome] would_reject_count may be greater than 0; shadow-only mode logs decisions but does not block trades."

Write-Host "[xgboost-shadow-outcome] Running XGBoost rejection audit..."
& $py "tools\audit_xgboost_rejections.py" --json --json-out $auditJson
if ($LASTEXITCODE -ne 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: audit generation failed." -ForegroundColor Red
  exit 1
}

$audit = Get-Content $auditJson -Raw | ConvertFrom-Json
if ($FreshShadowLog -and $audit.actually_rejected_count -ne 0) {
  Write-Host "[xgboost-shadow-outcome] FAIL: audit actually_rejected_count should be 0 for a fresh shadow-only run." -ForegroundColor Red
  exit 1
}
if ((-not $FreshShadowLog) -and $audit.actually_rejected_count -gt 0) {
  Write-Host "[xgboost-shadow-outcome] Audit includes pre-existing shadow rows with actually_rejected_count > 0; new rows were verified as shadow-only." -ForegroundColor Yellow
}
$join = $audit.trade_outcome_join
Write-Host ("[xgboost-shadow-outcome] Audit join: method={0}, would_confirm_matched_count={1}, would_reject_matched_count={2}" -f `
  $join.join_method, $join.would_confirm_matched_count, $join.would_reject_matched_count)
Write-Host ("[xgboost-shadow-outcome] Audit PnL: would_confirm_average_pnl={0}, would_reject_average_pnl={1}" -f `
  $join.would_confirm_average_pnl, $join.would_reject_average_pnl)

Write-Host ""
Write-Host "[xgboost-shadow-outcome] Done."
Write-Host "  shadow_log: $shadowLog"
Write-Host "  live_signals: $signalsLog"
Write-Host "  audit_json: $auditJson"
Write-Host "  writer_stdout: $writerOut"
Write-Host "  writer_stderr: $writerErr"
Write-Host "  executor_stdout: $executorOut"
Write-Host "  executor_stderr: $executorErr"
