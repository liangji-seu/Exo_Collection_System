$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python 3.11 virtual environment not found. Follow README.md first."
}

Push-Location $ProjectRoot
try {
    & $Python -m PyInstaller --noconfirm --clean packaging\collector.spec
    & $Python -m PyInstaller --noconfirm --clean packaging\data_studio.spec
}
finally {
    Pop-Location
}

Write-Host "Built dist\ExoCollector.exe and dist\ExoDataStudio.exe"

