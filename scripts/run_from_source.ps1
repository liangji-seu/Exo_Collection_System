[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("ExoCollector", "ExoDataStudio")]
    [string]$Application
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw @"
Python virtual environment was not found at:
$Python

Create it from the project root with:
  py -3.11 -m venv .venv
  .\.venv\Scripts\python.exe -m pip install --upgrade pip
  .\.venv\Scripts\python.exe -m pip install -e ".[dev,packaging]"
"@
}

$VersionOutput = & $Python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$VersionExitCode = $LASTEXITCODE
if ($VersionExitCode -ne 0) {
    throw "Unable to query the virtual environment Python version (exit code $VersionExitCode)."
}
$PythonVersion = [string]($VersionOutput | Select-Object -Last 1)
$PythonVersion = $PythonVersion.Trim()
if ($PythonVersion -ne "3.11") {
    throw "This project requires Python 3.11, but .venv uses Python $PythonVersion. Recreate .venv with: py -3.11 -m venv .venv"
}

$Module = switch ($Application) {
    "ExoCollector" { "exo_collection.apps.collector.main" }
    "ExoDataStudio" { "exo_collection.apps.data_studio.main" }
}
$DependencyProbe = "import PySide6, pyqtgraph, numpy, h5py, sqlalchemy, pydantic; import $Module"
& $Python -c $DependencyProbe | Out-Null
$ImportExitCode = $LASTEXITCODE
if ($ImportExitCode -ne 0) {
    throw "Unable to import the $Application entry point or one of its dependencies (exit code $ImportExitCode). Run: .\.venv\Scripts\python.exe -m pip install -e `".[dev,packaging]`""
}

Push-Location -LiteralPath $ProjectRoot
try {
    Write-Host "Starting $Application from source with Python $PythonVersion..."
    & $Python -m $Module
    $ApplicationExitCode = $LASTEXITCODE
    if ($ApplicationExitCode -ne 0) {
        throw "$Application exited with code $ApplicationExitCode."
    }
}
finally {
    Pop-Location
}
