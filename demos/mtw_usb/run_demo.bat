@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if defined MTW_DEMO_PYTHON (
  set "MTW_PY=%MTW_DEMO_PYTHON%"
) else if exist "E:\miniconda\envs\EXO\python.exe" (
  set "MTW_PY=E:\miniconda\envs\EXO\python.exe"
) else (
  set "MTW_PY=python"
)
"%MTW_PY%" -u run_demo.py %*
set "MTW_RESULT=%ERRORLEVEL%"
echo.
echo Exit code: %MTW_RESULT%
pause
exit /b %MTW_RESULT%
