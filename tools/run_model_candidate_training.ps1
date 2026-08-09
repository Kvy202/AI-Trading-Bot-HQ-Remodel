[CmdletBinding()]
param(
    [ValidateSet("lstm", "tcn", "tx")]
    [string]$Model = "lstm",

    [switch]$DryRun,
    [switch]$Bootstrap,
    [switch]$CaptureDataset,
    [switch]$BuildDataset,
    [switch]$BalanceProbe,
    [switch]$BalanceFreeze,
    [switch]$Train,
    [switch]$Evaluate,
    [switch]$LegacyRepairGate,
    [switch]$CaptureConfirmation,
    [switch]$ConfirmationGate,

    [string]$Dataset,
    [string]$Candidate,
    [string]$Confirmation,
    [string]$Venue = "bitget"
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProjectPython = "python"
$TrainingPython = Join-Path $Repository ".venv-model-training\canonical\Scripts\python.exe"

function Show-SafetyBanner {
    Write-Output "PHASE24_CANDIDATE_RESEARCH_ONLY"
    Write-Output "writer_started=false"
    Write-Output "executor_started=false"
    Write-Output "matrix_started=false"
    Write-Output "exchange_execution_initialized=false"
    Write-Output "orders_allowed=false"
    Write-Output "incumbent_overwrite_allowed=false"
    Write-Output "promotion_allowed=false"
    Write-Output "live_activation_allowed=false"
    Write-Output "project_python=$ProjectPython"
    Write-Output "training_python=$TrainingPython"
}

function Invoke-CheckedPython {
    param(
        [Parameter(Mandatory = $true)][string]$Interpreter,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Output ("interpreter={0} arguments={1}" -f $Interpreter, ($Arguments -join " "))
    & $Interpreter @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Phase 24 command failed with exit code $LASTEXITCODE"
    }
}

function Require-PathArgument {
    param([string]$Value, [string]$Name)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "-$Name is required for this operation"
    }
}

Show-SafetyBanner

if ($DryRun) {
    Write-Output "dry_run=true"
    Write-Output "mutation_performed=false"
    Write-Output "bootstrap_planned=project Python -> tools/model_training_environment.py --bootstrap"
    Write-Output "capture_planned=project Python -> tools/model_training_dataset.py capture"
    Write-Output "build_planned=project Python -> tools/model_training_dataset.py build (scaler worker uses training Python)"
    Write-Output "balance_probe_planned=project Python -> tools/model_loss_balance_probe.py synthetic_only"
    Write-Output "balance_freeze_planned=training Python -> tools/model_candidate_loss_balance.py training_sequences_only"
    Write-Output "balance_required=true"
    Write-Output "balance_freeze_status=pending"
    Write-Output "validation_access_allowed=false"
    Write-Output "training_allowed=false"
    Write-Output "train_planned=training Python -> tools/model_candidate_train.py --model $Model"
    Write-Output "objective_contract_required=reports/model_objective_contract.json verdict=candidate_objective_contract_resolved_multitask_training_required"
    Write-Output "objective_planned=parent objective plus architecture-specific frozen balance formulation"
    Write-Output "evaluate_planned=training Python -> tools/model_candidate_evaluate.py"
    Write-Output "legacy_gate_planned=training Python -> tools/model_candidate_health_gate.py --gate legacy-repair"
    Write-Output "confirmation_capture_planned=project Python -> tools/model_training_dataset.py capture-confirmation"
    Write-Output "confirmation_gate_planned=training Python -> tools/model_candidate_health_gate.py --gate confirmation"
    exit 0
}

$OperationCount = @(
    $Bootstrap, $CaptureDataset, $BuildDataset, $BalanceProbe, $BalanceFreeze, $Train, $Evaluate,
    $LegacyRepairGate, $CaptureConfirmation, $ConfirmationGate
).Where({ $_ }).Count
if ($OperationCount -eq 0) {
    throw "Select an operation switch or use -DryRun"
}

Push-Location $Repository
try {
    if ($Bootstrap) {
        Invoke-CheckedPython -Interpreter $ProjectPython -Arguments @(
            "tools/model_training_environment.py", "--bootstrap"
        )
    }
    if ($CaptureDataset) {
        Invoke-CheckedPython -Interpreter $ProjectPython -Arguments @(
            "tools/model_training_dataset.py", "capture", "--venue", $Venue
        )
    }
    if ($BuildDataset) {
        Require-PathArgument -Value $Dataset -Name "Dataset"
        Invoke-CheckedPython -Interpreter $ProjectPython -Arguments @(
            "tools/model_training_dataset.py", "build", "--dataset", $Dataset,
            "--training-python", $TrainingPython
        )
    }
    if ($BalanceProbe) {
        Invoke-CheckedPython -Interpreter $ProjectPython -Arguments @(
            "tools/model_loss_balance_probe.py", "--json-out", "reports/model_loss_balance_probe.json"
        )
    }
    if ($BalanceFreeze) {
        Require-PathArgument -Value $Dataset -Name "Dataset"
        Invoke-CheckedPython -Interpreter $TrainingPython -Arguments @(
            "tools/model_candidate_loss_balance.py", "--dataset", $Dataset,
            "--json-out", "reports/model_candidate_loss_balance.json",
            "--freeze-out", "reports/model_candidate_loss_balance_freeze.json"
        )
    }
    if ($Train) {
        Require-PathArgument -Value $Dataset -Name "Dataset"
        Invoke-CheckedPython -Interpreter $TrainingPython -Arguments @(
            "tools/model_candidate_train.py", "--model", $Model, "--dataset", $Dataset,
            "--objective", "resolved_candidate_objective",
            "--objective-contract", "reports/model_objective_contract.json",
            "--balance-freeze", "reports/model_candidate_loss_balance_freeze.json"
        )
    }
    if ($Evaluate) {
        Require-PathArgument -Value $Candidate -Name "Candidate"
        Invoke-CheckedPython -Interpreter $TrainingPython -Arguments @(
            "tools/model_candidate_evaluate.py", "--candidate", $Candidate
        )
    }
    if ($LegacyRepairGate) {
        Require-PathArgument -Value $Candidate -Name "Candidate"
        Invoke-CheckedPython -Interpreter $TrainingPython -Arguments @(
            "tools/model_candidate_health_gate.py", "--candidate", $Candidate,
            "--gate", "legacy-repair"
        )
    }
    if ($CaptureConfirmation) {
        Invoke-CheckedPython -Interpreter $ProjectPython -Arguments @(
            "tools/model_training_dataset.py", "capture-confirmation"
        )
    }
    if ($ConfirmationGate) {
        Require-PathArgument -Value $Candidate -Name "Candidate"
        Require-PathArgument -Value $Confirmation -Name "Confirmation"
        Invoke-CheckedPython -Interpreter $TrainingPython -Arguments @(
            "tools/model_candidate_health_gate.py", "--candidate", $Candidate,
            "--gate", "confirmation", "--confirmation", $Confirmation
        )
    }
}
finally {
    Pop-Location
}
