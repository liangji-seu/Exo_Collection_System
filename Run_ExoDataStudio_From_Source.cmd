@echo off
setlocal

set "LAUNCHER=%~dp0scripts\run_from_source.ps1"
if not exist "%LAUNCHER%" (
    echo [ERROR] Source launcher was not found:
    echo         "%LAUNCHER%"
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Application ExoDataStudio
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] ExoDataStudio source launch failed. See the message above.
    pause
)

endlocal & exit /b %EXIT_CODE%
