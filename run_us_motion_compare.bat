@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "EXO_PY=E:\miniconda\envs\EXO\python.exe"
if exist "%EXO_PY%" (
  "%EXO_PY%" "%SCRIPT_DIR%run_us_motion_compare.py" %*
) else (
  python "%SCRIPT_DIR%run_us_motion_compare.py" %*
)
if errorlevel 1 (
  echo.
  echo [错误] 超声原始区域移动对比工具启动失败。
  pause
)
endlocal
