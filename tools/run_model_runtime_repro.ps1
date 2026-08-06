[CmdletBinding()]
param(
    [switch]$Bootstrap,
    [switch]$Compare,
    [switch]$DryRun,
    [string]$Bundle = ".\reports\model_alignment_bundles\history_5m_final"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Requirements = Join-Path $RepoRoot "requirements\model_repro_sklearn180.txt"
$ReproRoot = Join-Path $RepoRoot ".venv-repro-sklearn180"
$ReproPython = Join-Path $ReproRoot "Scripts\python.exe"
$ContractFile = Join-Path $ReproRoot ".phase23-contract.json"
$Report = Join-Path $RepoRoot "reports\model_runtime_reproducibility.json"
$BundlePath = if ([System.IO.Path]::IsPathRooted($Bundle)) {
    [System.IO.Path]::GetFullPath($Bundle)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $RepoRoot $Bundle))
}
$MainPython = if (Test-Path (Join-Path $RepoRoot ".venv\Scripts\python.exe")) {
    Join-Path $RepoRoot ".venv\Scripts\python.exe"
} else {
    (Get-Command python -ErrorAction Stop).Source
}

if (-not $Bootstrap -and -not $Compare -and -not $DryRun) {
    throw "Specify -Bootstrap, -Compare, or -DryRun."
}

$RequirementDigest = (Get-FileHash -Algorithm SHA256 -LiteralPath $Requirements).Hash.ToLowerInvariant()

if ($DryRun) {
    Write-Output "Phase 23 dry run: no environment or report will be created."
    Write-Output "Main Python: $MainPython"
    Write-Output "Reproduction environment: $ReproRoot"
    Write-Output "Requirements SHA256: $RequirementDigest"
    Write-Output "Bundle: $BundlePath"
    exit 0
}

function Get-PythonContract([string]$PythonPath) {
    $Code = @'
import hashlib, json, platform, subprocess, sys
versions = {"python": platform.python_version()}
for name, module in (("numpy", "numpy"), ("scipy", "scipy"), ("joblib", "joblib"), ("scikit_learn", "sklearn"), ("threadpoolctl", "threadpoolctl")):
    try:
        value = __import__(module)
        versions[name] = str(value.__version__)
    except Exception:
        versions[name] = None
try:
    import torch
    versions["torch_installed"] = True
except Exception:
    versions["torch_installed"] = False
try:
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze", "--all"], text=True)
    lines = sorted(line.strip() for line in frozen.splitlines() if line.strip())
    versions["pip_freeze_digest"] = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
except Exception:
    versions["pip_freeze_digest"] = None
print(json.dumps(versions, sort_keys=True))
'@
    # Send code over stdin. Windows PowerShell 5.1 can otherwise strip quotes
    # from multiline native-command arguments passed through `python -c`.
    $Output = $Code | & $PythonPath -
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to inventory Python environment contract."
    }
    return ($Output | ConvertFrom-Json)
}

$MainContract = Get-PythonContract $MainPython
$MainMajorMinor = ($MainContract.python -split '\.')[0..1] -join '.'

function Assert-ReproductionContract($Contract) {
    $ReproMajorMinor = ($Contract.python -split '\.')[0..1] -join '.'
    if ($ReproMajorMinor -ne $MainMajorMinor) {
        throw "Python major/minor mismatch between main and reproduction environments."
    }
    if ($Contract.scikit_learn -ne "1.8.0" -or
        $Contract.numpy -ne "2.3.3" -or
        $Contract.scipy -ne "1.16.2" -or
        $Contract.joblib -ne "1.5.2" -or
        $Contract.threadpoolctl -ne "3.6.0" -or
        $Contract.torch_installed -or
        [string]::IsNullOrWhiteSpace($Contract.pip_freeze_digest)) {
        throw "Reproduction environment package contract differs; refusing reuse."
    }
}

if ($Bootstrap) {
    if (Test-Path $ReproRoot) {
        if (-not (Test-Path $ReproPython)) {
            throw "Existing reproduction path is not a valid virtual environment; refusing reuse."
        }
        $Existing = Get-PythonContract $ReproPython
        $Marker = if (Test-Path $ContractFile) { Get-Content $ContractFile -Raw | ConvertFrom-Json } else { $null }
        Assert-ReproductionContract $Existing
        if ($null -ne $Marker -and (
            $Marker.requirements_digest -ne $RequirementDigest -or
            $Marker.python_major_minor -ne $MainMajorMinor -or
            $Marker.pip_freeze_digest -ne $Existing.pip_freeze_digest)) {
            throw "Existing reproduction environment contract differs; refusing reuse."
        }
        if ($null -eq $Marker) {
            # Recover safely from an interruption after installation but before
            # marker creation.  Every executable contract field was validated
            # above; no package operation is performed during this recovery.
            @{
                requirements_digest=$RequirementDigest
                python_major_minor=$MainMajorMinor
                pip_freeze_digest=$Existing.pip_freeze_digest
            } |
                ConvertTo-Json | Set-Content -LiteralPath $ContractFile -Encoding utf8
        }
    } else {
        $LogRoot = Join-Path $RepoRoot "reports"
        New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
        $StdoutLog = Join-Path $LogRoot "model_runtime_repro_bootstrap_stdout.log"
        $StderrLog = Join-Path $LogRoot "model_runtime_repro_bootstrap_stderr.log"
        $PreviousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $MainPython -m venv $ReproRoot 1> $StdoutLog 2> $StderrLog
        $VenvExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($VenvExitCode -ne 0) { throw "Unable to create reproduction environment." }
        $ErrorActionPreference = "Continue"
        & $ReproPython -m pip install --disable-pip-version-check --requirement $Requirements 1>> $StdoutLog 2>> $StderrLog
        $PipExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousErrorActionPreference
        if ($PipExitCode -ne 0) { throw "Reproduction dependency installation failed; inspect captured logs." }
        $Installed = Get-PythonContract $ReproPython
        Assert-ReproductionContract $Installed
        @{
            requirements_digest=$RequirementDigest
            python_major_minor=$MainMajorMinor
            pip_freeze_digest=$Installed.pip_freeze_digest
        } |
            ConvertTo-Json | Set-Content -LiteralPath $ContractFile -Encoding utf8
    }
}

if ($Compare) {
    if (-not (Test-Path $ReproPython)) { throw "Reproduction environment is missing; run -Bootstrap first." }
    $ReproContract = Get-PythonContract $ReproPython
    Assert-ReproductionContract $ReproContract
    $Marker = if (Test-Path $ContractFile) { Get-Content $ContractFile -Raw | ConvertFrom-Json } else { $null }
    if ($null -eq $Marker -or
        $Marker.requirements_digest -ne $RequirementDigest -or
        $Marker.python_major_minor -ne $MainMajorMinor -or
        $Marker.pip_freeze_digest -ne $ReproContract.pip_freeze_digest) {
        throw "Existing reproduction environment contract differs; refusing reuse."
    }
    $LogRoot = Join-Path $RepoRoot "reports"
    New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
    $CompareStdout = Join-Path $LogRoot "model_runtime_repro_compare_stdout.log"
    $CompareStderr = Join-Path $LogRoot "model_runtime_repro_compare_stderr.log"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $MainPython (Join-Path $RepoRoot "tools\model_runtime_repro.py") `
        --bundle $BundlePath --current-python $MainPython --repro-python $ReproPython --json-out $Report `
        1> $CompareStdout 2> $CompareStderr
    $CompareExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($CompareExitCode -ne 0) { throw "Runtime comparison failed; inspect captured logs." }
}
