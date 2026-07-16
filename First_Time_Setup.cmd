@echo off
setlocal

set "REMAINING=%~dp0"
if "%REMAINING:~-1%"=="\" set "REMAINING=%REMAINING:~0,-1%"
:__exo_chk_nul
for /f "tokens=1* delims=\" %%a in ("%REMAINING%") do (
    if /i "%%a"=="NUL" goto :__exo_nul_blocked
    if /i "%%a"=="CON" goto :__exo_nul_blocked
    if /i "%%a"=="PRN" goto :__exo_nul_blocked
    if /i "%%a"=="AUX" goto :__exo_nul_blocked
    set "REMAINING=%%b"
    if not "%%b"=="" goto :__exo_chk_nul
)
goto :__exo_nul_ok
:__exo_nul_blocked
echo [ERROR] The project directory path contains a Windows reserved device name (NUL/CON/PRN/AUX).
echo Move the project to a directory without reserved names before running this script.
pause
exit /b 1
:__exo_nul_ok

set "SETUP_SCRIPT=%~dp0scripts\first_time_setup_and_build.ps1"
if not exist "%SETUP_SCRIPT%" (
    echo [ERROR] First-time setup script was not found:
    echo         "%SETUP_SCRIPT%"
    pause
    exit /b 1
)

echo Exo Collection System - first-time Windows setup and release build
echo.
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SETUP_SCRIPT%"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%EXIT_CODE%"=="0" (
    echo [ERROR] First-time setup or Windows release build failed.
    echo Review the message above, fix the reported issue, and run this script again.
    pause
    endlocal & exit /b %EXIT_CODE%
)

echo Setup, tests, packaging, smoke checks, and bundle creation completed.
echo The distributable ZIP is in the release folder.
echo For daily use in this checkout, run Run_ExoCollector.cmd or Run_ExoDataStudio.cmd.
pause
endlocal & exit /b 0
