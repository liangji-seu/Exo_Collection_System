@echo off
setlocal

set "BUILD_SCRIPT=%~dp0packaging\build_windows.ps1"
if not exist "%BUILD_SCRIPT%" (
    echo [ERROR] Build script was not found:
    echo         "%BUILD_SCRIPT%"
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%BUILD_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Windows packaging failed. See the message above.
    pause
)

endlocal & exit /b %EXIT_CODE%
