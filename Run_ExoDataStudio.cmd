@echo off
setlocal

set "PROJECT_ROOT=%~dp0"
set "APP=%PROJECT_ROOT%dist\ExoDataStudio.exe"

if not exist "%APP%" (
    echo [ERROR] ExoDataStudio.exe was not found:
    echo         "%APP%"
    echo.
    echo Run Build_Windows.cmd first.
    pause
    exit /b 1
)

start "" /D "%PROJECT_ROOT%" "%APP%"
if errorlevel 1 (
    echo [ERROR] Failed to start ExoDataStudio.exe.
    pause
    exit /b 1
)

exit /b 0
