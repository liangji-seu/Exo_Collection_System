@echo off
setlocal

set "SETUP_SCRIPT=%~dp0scripts\first_time_setup_and_build.ps1"
if not exist "%SETUP_SCRIPT%" (
    echo [ERROR] First-time setup script was not found:
    echo         "%SETUP_SCRIPT%"
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] First-time setup or Windows packaging failed.
    echo Review the message above, fix the reported issue, and run this script again.
    pause
    endlocal & exit /b %EXIT_CODE%
)

echo First-time setup and Windows packaging completed successfully.
echo You can now run Run_ExoCollector.cmd or Run_ExoDataStudio.cmd.
pause
endlocal & exit /b 0
