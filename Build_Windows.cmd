@echo off
setlocal

rem --- Guard against Windows reserved device names (NUL/CON/PRN/AUX) in the
rem     script path because they can cause silent corruption of the working directory.
rem     Segment-by-segment check; no findstr or pipeline so that the guard
rem     itself never fails on Unicode or multi-byte characters in the path.
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
if "%EXIT_CODE%"=="0" (
    echo.
    echo Windows release build completed. See the release folder for the distributable ZIP.
)

endlocal & exit /b %EXIT_CODE%
