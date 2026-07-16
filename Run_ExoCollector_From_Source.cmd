@echo off
setlocal

rem --- Guard against Windows reserved device names (NUL/CON/PRN/AUX) in the
rem     script path because they can cause silent corruption of the working directory.
rem     Segment-by-segment check; no findstr or pipeline so that the guard
rem     itself never fails on Unicode or multi-byte characters in the path.
set "PROJECT_ROOT=%~dp0"
set "REMAINING=%PROJECT_ROOT%"
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

set "LAUNCHER=%~dp0scripts\run_from_source.ps1"
if not exist "%LAUNCHER%" (
    echo [ERROR] Source launcher was not found:
    echo         "%LAUNCHER%"
    pause
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%LAUNCHER%" -Application ExoCollector
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] ExoCollector source launch failed. See the message above.
    pause
)

endlocal & exit /b %EXIT_CODE%
