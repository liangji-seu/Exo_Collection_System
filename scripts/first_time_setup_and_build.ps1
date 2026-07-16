[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvRoot "Scripts\python.exe"
$BuildScript = Join-Path $ProjectRoot "packaging\build_windows.ps1"
$PyProject = Join-Path $ProjectRoot "pyproject.toml"

foreach ($RequiredFile in @($BuildScript, $PyProject)) {
    if (-not (Test-Path -LiteralPath $RequiredFile -PathType Leaf)) {
        throw "Required project file was not found: $RequiredFile"
    }
}

$GitCommand = Get-Command "git.exe" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $GitCommand) {
    throw @"
Git for Windows was not found. A default release build requires a verified,
clean Git checkout so BUILD_MANIFEST.json can identify its exact source.

Install Git for Windows, clone the repository, and run First_Time_Setup.cmd
again from that clean checkout.
"@
}
$InsideWorkTree = & $GitCommand.Source -C $ProjectRoot rev-parse --is-inside-work-tree 2>$null
if ($LASTEXITCODE -ne 0 -or ([string]($InsideWorkTree | Select-Object -Last 1)).Trim() -ne "true") {
    throw @"
This directory is not a verified Git checkout:
$ProjectRoot

Clone the repository with Git and run First_Time_Setup.cmd from the clone.
"@
}

$PyLauncherCommand = Get-Command "py.exe" -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $PyLauncherCommand) {
    throw @"
Windows Python launcher (py.exe) was not found.

Install 64-bit Python 3.11 from:
  https://www.python.org/downloads/windows/

During installation, enable the Python launcher option. Then reopen the terminal
and run First_Time_Setup.cmd again. No project files were removed.
"@
}
$PyLauncher = $PyLauncherCommand.Source

Write-Host "Checking the Windows Python launcher and Python 3.11..."
$LauncherVersionOutput = & $PyLauncher -3.11 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$LauncherVersionExitCode = $LASTEXITCODE
if ($LauncherVersionExitCode -ne 0) {
    throw @"
The Windows Python launcher is installed, but Python 3.11 is unavailable
(launcher exit code $LauncherVersionExitCode).

Install 64-bit Python 3.11 from:
  https://www.python.org/downloads/windows/

Keep the Python launcher enabled, reopen the terminal, and run this script again.
The existing .venv, if any, was not deleted or changed.
"@
}
$LauncherVersion = [string]($LauncherVersionOutput | Select-Object -Last 1)
$LauncherVersion = $LauncherVersion.Trim()
if ($LauncherVersion -ne "3.11") {
    throw "The command 'py -3.11' unexpectedly selected Python $LauncherVersion. Install Python 3.11 and run this script again."
}
Write-Host "Found Python $LauncherVersion through: $PyLauncher"

if (Test-Path -LiteralPath $VenvRoot) {
    if (-not (Test-Path -LiteralPath $VenvRoot -PathType Container)) {
        throw ".venv exists but is not a directory: $VenvRoot. This script will not delete or replace it."
    }
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw @"
An existing .venv directory was found, but its Python executable is missing:
$VenvPython

This script will never delete or replace an existing .venv. Move or rename that
directory manually after checking its contents, then run this script again.
"@
    }
    Write-Host "Using the existing virtual environment: $VenvRoot"
}
else {
    Write-Host "Creating a Python 3.11 virtual environment: $VenvRoot"
    & $PyLauncher -3.11 -m venv $VenvRoot
    $CreateVenvExitCode = $LASTEXITCODE
    if ($CreateVenvExitCode -ne 0) {
        throw "Creating .venv failed with exit code $CreateVenvExitCode. The incomplete directory, if created, was left untouched."
    }
    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        throw "The virtual-environment command succeeded but did not create the expected Python executable: $VenvPython"
    }
}

$VenvVersionOutput = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$VenvVersionExitCode = $LASTEXITCODE
if ($VenvVersionExitCode -ne 0) {
    throw "Unable to run the existing .venv Python (exit code $VenvVersionExitCode). The environment was not deleted or replaced."
}
$VenvVersion = [string]($VenvVersionOutput | Select-Object -Last 1)
$VenvVersion = $VenvVersion.Trim()
if ($VenvVersion -ne "3.11") {
    throw @"
The existing .venv uses Python $VenvVersion, but this project requires Python 3.11.

This script will never delete or replace an existing .venv. Move or rename it
manually after checking its contents, then run First_Time_Setup.cmd again.
"@
}
Write-Host "Virtual environment Python version is $VenvVersion."

Push-Location -LiteralPath $ProjectRoot
try {
    Write-Host "[1/3] Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip
    $PipUpgradeExitCode = $LASTEXITCODE
    if ($PipUpgradeExitCode -ne 0) {
        $PipVersionOutput = & $VenvPython -m pip --version
        $PipVersionExitCode = $LASTEXITCODE
        if ($PipVersionExitCode -ne 0) {
            throw "pip upgrade failed with exit code $PipUpgradeExitCode, and the existing pip is unusable (exit code $PipVersionExitCode). Check the package index and configured network/proxy support. Proxy settings were not changed."
        }
        $PipVersion = [string](
            $PipVersionOutput |
                Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
                Select-Object -Last 1
        )
        $PipVersion = $PipVersion.Trim()
        Write-Warning "pip self-upgrade failed with exit code $PipUpgradeExitCode, but the existing pip is usable ($PipVersion). Continuing with the installed pip; proxy settings were not changed."
    }

    Write-Host "[2/3] Installing the project and development/packaging dependencies..."
    & $VenvPython -m pip install -e ".[dev,packaging]"
    $InstallExitCode = $LASTEXITCODE
    if ($InstallExitCode -ne 0) {
        Write-Warning "Project dependency installation failed with exit code $InstallExitCode. Checking whether this environment is already complete and linked to the current checkout..."
        $DependencyProbe = @'
import importlib
import os
from pathlib import Path
import sys

required_modules = (
    "PySide6",
    "pyqtgraph",
    "numpy",
    "h5py",
    "sqlalchemy",
    "pydantic",
    "alembic",
    "paramiko",
    "scp",
    "pytest",
    "hypothesis",
    "PyInstaller",
    "exo_collection",
)
failures = []
loaded = {}
for module_name in required_modules:
    try:
        loaded[module_name] = importlib.import_module(module_name)
    except Exception as exc:
        failures.append(
            f"{module_name}: {type(exc).__name__}: {exc}"
        )

expected_source_root = Path(os.environ["EXO_EXPECTED_SOURCE_ROOT"]).resolve()
exo_module = loaded.get("exo_collection")
if exo_module is not None:
    module_file_value = getattr(exo_module, "__file__", None)
    if not module_file_value:
        failures.append("exo_collection: __file__ is unavailable")
    else:
        module_file = Path(module_file_value).resolve()
        if not module_file.is_relative_to(expected_source_root):
            failures.append(
                "exo_collection: imported from "
                f"{module_file}, expected a path under {expected_source_root}"
            )

if failures:
    print("Existing-environment dependency probe failed:", file=sys.stderr)
    for failure in failures:
        print(f"  - {failure}", file=sys.stderr)
    raise SystemExit(1)

print(
    "Existing-environment dependency probe passed; "
    f"exo_collection is loaded from {Path(exo_module.__file__).resolve()}"
)
'@
        # Windows PowerShell 5.1 can corrupt a multi-line native-command
        # argument containing quotes. Base64 keeps the Python probe as one
        # ASCII argument, while the expected Unicode path travels via env.
        $DependencyProbeBytes = [System.Text.Encoding]::UTF8.GetBytes($DependencyProbe)
        $EncodedDependencyProbe = [Convert]::ToBase64String($DependencyProbeBytes)
        $ProbeBootstrap = "import base64,sys;exec(compile(base64.b64decode(sys.argv[1]),'<exo_dependency_probe>','exec'))"
        $ExpectedSourceRootWasSet = Test-Path -LiteralPath "Env:\EXO_EXPECTED_SOURCE_ROOT"
        $PreviousExpectedSourceRoot = $env:EXO_EXPECTED_SOURCE_ROOT
        $env:EXO_EXPECTED_SOURCE_ROOT = Join-Path $ProjectRoot "src"
        try {
            & $VenvPython -c $ProbeBootstrap $EncodedDependencyProbe
            $DependencyProbeExitCode = $LASTEXITCODE
        }
        finally {
            if ($ExpectedSourceRootWasSet) {
                $env:EXO_EXPECTED_SOURCE_ROOT = $PreviousExpectedSourceRoot
            }
            else {
                Remove-Item -LiteralPath "Env:\EXO_EXPECTED_SOURCE_ROOT" -ErrorAction SilentlyContinue
            }
        }
        if ($DependencyProbeExitCode -ne 0) {
            throw @"
Project installation failed with exit code $InstallExitCode, and the existing
environment is missing a required dependency or is not linked to this checkout
(probe exit code $DependencyProbeExitCode). Review the probe details above.

Check access to the configured package index and confirm that the inherited
network/proxy configuration has the required protocol support. This script did
not clear, bypass, or otherwise change any proxy setting.
"@
        }
        Write-Warning "pip install failed, but every required dependency is importable and exo_collection points to the current $ProjectRoot\src. Reusing the complete existing environment and continuing; proxy settings were not changed."
    }

    Write-Host "[3/3] Running tests, building both applications, smoke-checking, and creating the release bundle..."
    & $BuildScript
    if ($LASTEXITCODE -ne 0) {
        throw "Windows release build failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "First-time setup and Windows release bundle completed successfully."
