[CmdletBinding()]
param(
    [switch]$Bootstrap,
    [switch]$Compare,
    [switch]$DryRun,
    [ValidateSet(
        "observed_main", "declared_sklearn_only", "sklearn_only_180",
        "numpy_only_233", "scipy_only_162", "joblib_only_152",
        "serialized_full_stack", "numpy_233_sklearn_180",
        "scipy_162_sklearn_180", "joblib_152_sklearn_180",
        "numpy_233_scipy_162", "numpy_233_joblib_152"
    )]
    [string]$Stack,
    [string]$Bundle = ".\reports\model_alignment_bundles\history_5m_final",
    [string]$Matrix = ".\research\runtime_stack_matrix.json",
    [string]$WorkingDir = ".\reports\runtime_stack_work",
    [string]$JsonOut = ".\reports\runtime_stack_isolation.json"
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentRoot = Join-Path $repoRoot ".venv-runtime-isolation"
$mainVenv = Join-Path $repoRoot ".venv"
$mainPython = (Get-Command python -ErrorAction Stop).Source
$matrixPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Matrix))
$bundlePath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Bundle))
$workingPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $WorkingDir))
$reportPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $JsonOut))
$isolationTool = Join-Path $repoRoot "tools\runtime_stack_isolation.py"

if (-not ($Bootstrap -or $Compare -or $DryRun)) {
    throw "Specify -Bootstrap, -Compare, or -DryRun."
}
if ($DryRun -and ($Bootstrap -or $Compare)) {
    throw "-DryRun cannot be combined with -Bootstrap or -Compare."
}
if ($mainPython.StartsWith($mainVenv, [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "main_interpreter_source=project_.venv_read_only"
}

$matrixObject = Get-Content -LiteralPath $matrixPath -Raw | ConvertFrom-Json
$primaryStacks = @(
    "declared_sklearn_only", "sklearn_only_180", "numpy_only_233",
    "scipy_only_162", "joblib_only_152", "serialized_full_stack"
)
$targetStacks = if ($Stack) { @($Stack) } else { $primaryStacks }
$targetStacks = @($targetStacks | Where-Object { $_ -ne "observed_main" })

function Write-SafetyContract {
    Write-Host "writer_started=false"
    Write-Host "executor_started=false"
    Write-Host "matrix_started=false"
    Write-Host "exchange_initialized=false"
    Write-Host "orders_allowed=false"
    Write-Host "main_environment_modified=false"
}

function Get-StackPython([string]$StackId) {
    return Join-Path (Join-Path $environmentRoot $StackId) "Scripts\python.exe"
}

function Write-StackIntent([string]$StackId) {
    $definition = $matrixObject.stacks.$StackId
    $pythonPath = Get-StackPython $StackId
    Write-Host "stack=$StackId"
    Write-Host "environment=$([System.IO.Path]::GetDirectoryName([System.IO.Path]::GetDirectoryName($pythonPath)))"
    Write-Host "python_major_minor=$($matrixObject.python_major_minor)"
    foreach ($package in $matrixObject.packages) {
        Write-Host "$package==$($definition.package_versions.$package)"
    }
}

function Invoke-Captured([string]$Label, [scriptblock]$Command) {
    # Native tools legitimately emit deserialization/model warnings on stderr.
    # Capture both streams without promoting stderr text to a PowerShell
    # terminating error; the native exit code remains authoritative.
    $previousErrorAction = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $captured = & $Command 2>&1
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorAction
    }
    foreach ($line in $captured) { Write-Host "$line" }
    if ($exitCode -ne 0) {
        throw "$Label failed with exit code $exitCode"
    }
}

function Initialize-Stack([string]$StackId) {
    $definition = $matrixObject.stacks.$StackId
    if ($null -eq $definition) { throw "Unknown stack: $StackId" }
    $stackDir = Join-Path $environmentRoot $StackId
    $stackPython = Get-StackPython $StackId
    $marker = Join-Path $stackDir ".runtime-stack-manifest.json"
    Write-StackIntent $StackId
    if (Test-Path -LiteralPath $stackPython -PathType Leaf) {
        if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) {
            throw "Refusing to reuse $StackId without its environment manifest."
        }
        Invoke-Captured "environment reuse validation" {
            & $mainPython $isolationTool --matrix $matrixPath --validate-environment $StackId
        }
        Write-Host "reuse_status=validated"
        return
    }
    if (Test-Path -LiteralPath $stackDir) {
        throw "Refusing partial or mismatched environment directory: $stackDir"
    }
    if (-not (Test-Path -LiteralPath $environmentRoot)) {
        New-Item -ItemType Directory -Path $environmentRoot | Out-Null
    }
    $mainVersion = & $mainPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($mainVersion.Trim() -ne $matrixObject.python_major_minor) {
        throw "Main Python major/minor $mainVersion does not match matrix $($matrixObject.python_major_minor)."
    }
    Invoke-Captured "venv creation" { & $mainPython -m venv $stackDir }
    $specifications = @()
    foreach ($package in $matrixObject.packages) {
        $specifications += "$package==$($definition.package_versions.$package)"
    }
    Invoke-Captured "isolated numerical package install" {
        & $stackPython -m pip install --disable-pip-version-check --no-input --no-deps $specifications
    }
    Invoke-Captured "environment manifest recording" {
        & $mainPython $isolationTool --matrix $matrixPath --record-environment $StackId --environment-python $stackPython
    }
    Write-Host "bootstrap_status=created_and_validated"
}

function Invoke-Comparison([string[]]$SelectedStacks) {
    $arguments = @(
        $isolationTool, "--bundle", $bundlePath, "--matrix", $matrixPath,
        "--working-dir", $workingPath, "--json-out", $reportPath,
        "--current-python", $mainPython
    )
    foreach ($selected in $SelectedStacks) {
        $arguments += @("--stack", $selected)
    }
    Invoke-Captured "runtime stack comparison" { & $mainPython @arguments }
}

if ($DryRun) {
    Write-Host "dry_run=true"
    Write-Host "environment_root=$environmentRoot"
    foreach ($stackId in $targetStacks) { Write-StackIntent $stackId }
    Write-Host "isolation_report=$reportPath"
    Write-Host "attribution_report=$(Join-Path $repoRoot 'reports\runtime_stack_attribution.json')"
    Write-Host "decision_report=$(Join-Path $repoRoot 'reports\runtime_stack_decision.json')"
    Write-SafetyContract
    exit 0
}

if ($Bootstrap) {
    foreach ($stackId in $targetStacks) { Initialize-Stack $stackId }
}

if ($Compare) {
    $comparisonSelection = if ($Stack) { @($Stack) } else { @() }
    Invoke-Comparison $comparisonSelection
    if ($Bootstrap -and -not $Stack) {
        $initialReport = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $interactionStacks = @($initialReport.overall_decision.interaction_stacks_required)
        if ($interactionStacks.Count -gt 0) {
            foreach ($stackId in $interactionStacks) { Initialize-Stack $stackId }
            Invoke-Comparison @()
        }
    }
}

Write-SafetyContract
