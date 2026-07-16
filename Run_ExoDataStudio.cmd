@echo off
setlocal

set "PROJECT_ROOT=%~dp0"

rem --- Guard against Windows reserved device names (NUL/CON/PRN/AUX) in the
rem     script path because they can cause silent corruption of the working directory.
rem     Segment-by-segment check; no findstr or pipeline so that the guard
rem     itself never fails on Unicode or multi-byte characters in the path.
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
