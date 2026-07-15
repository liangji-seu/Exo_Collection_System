Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$CollectorSpec = Join-Path $PSScriptRoot "collector.spec"
$DataStudioSpec = Join-Path $PSScriptRoot "data_studio.spec"
$CollectorExe = Join-Path $ProjectRoot "dist\ExoCollector.exe"
$DataStudioExe = Join-Path $ProjectRoot "dist\ExoDataStudio.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw @"
Python virtual environment was not found at:
$Python

Create the required Python 3.11 environment from the project root:
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

$PyInstallerOutput = & $Python -c "import PyInstaller; print(PyInstaller.__version__)"
$PyInstallerExitCode = $LASTEXITCODE
if ($PyInstallerExitCode -ne 0) {
    throw 'PyInstaller is not installed. Run: .\.venv\Scripts\python.exe -m pip install -e ".[dev,packaging]"'
}
$PyInstallerVersion = [string]($PyInstallerOutput | Select-Object -Last 1)
$PyInstallerVersion = $PyInstallerVersion.Trim()

foreach ($Spec in @($CollectorSpec, $DataStudioSpec)) {
    if (-not (Test-Path -LiteralPath $Spec -PathType Leaf)) {
        throw "PyInstaller spec file was not found: $Spec"
    }
}

Write-Host "Building Windows applications with Python $PythonVersion and PyInstaller $PyInstallerVersion..."
Push-Location -LiteralPath $ProjectRoot
try {
    Write-Host "[1/2] Building ExoCollector.exe"
    & $Python -m PyInstaller --noconfirm --clean $CollectorSpec
    $CollectorBuildExitCode = $LASTEXITCODE
    if ($CollectorBuildExitCode -ne 0) {
        throw "ExoCollector build failed with exit code $CollectorBuildExitCode."
    }

    Write-Host "[2/2] Building ExoDataStudio.exe"
    & $Python -m PyInstaller --noconfirm --clean $DataStudioSpec
    $DataStudioBuildExitCode = $LASTEXITCODE
    if ($DataStudioBuildExitCode -ne 0) {
        throw "ExoDataStudio build failed with exit code $DataStudioBuildExitCode."
    }
}
finally {
    Pop-Location
}

foreach ($Executable in @($CollectorExe, $DataStudioExe)) {
    if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
        throw "Packaging completed without the expected executable: $Executable"
    }
}

Write-Host "Build completed successfully:"
Write-Host "  $CollectorExe"
Write-Host "  $DataStudioExe"
