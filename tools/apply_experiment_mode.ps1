function Get-ExperimentModeOverrides {
  param(
    [Parameter(Mandatory = $true)] [string]$Python,
    [Parameter(Mandatory = $true)] [string]$Root,
    [Parameter(Mandatory = $true)] [string]$Mode
  )

  $toolPath = Join-Path $Root 'tools\experiment_mode.py'
  $rawLines = & $Python $toolPath --print-env $Mode
  if ($LASTEXITCODE -ne 0) {
    throw "experiment_mode.py failed for mode '$Mode'"
  }

  $overrides = [ordered]@{}
  foreach ($line in $rawLines) {
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    $idx = $line.IndexOf('=')
    if ($idx -le 0) {
      throw "invalid experiment mode line: $line"
    }
    $name = $line.Substring(0, $idx).Trim()
    $value = $line.Substring($idx + 1).Trim()
    if ($name -notmatch '^[A-Z0-9_]+$') {
      throw "invalid experiment mode key: $name"
    }
    $overrides[$name] = $value
  }

  return ,$overrides
}

function Set-ExperimentModeEnvironment {
  param(
    [Parameter(Mandatory = $true)] [System.Collections.IDictionary]$Overrides
  )

  foreach ($name in $Overrides.Keys) {
    Set-Item -Path "Env:$name" -Value ([string]$Overrides[$name])
  }
}
