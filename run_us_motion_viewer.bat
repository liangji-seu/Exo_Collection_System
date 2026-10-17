@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "EXO_PY=E:\miniconda\envs\EXO\python.exe"

if exist "%EXO_PY%" (
  "%EXO_PY%" "%SCRIPT_DIR%run_us_motion_viewer.py" %*
) else (
  python "%SCRIPT_DIR%run_us_motion_viewer.py" %*
)

if errorlevel 1 (
  echo.
  echo [错误] 超声区域移动播放器启动失败。请查看上方信息。
  echo 程序需要 PySide6、pyqtgraph、numpy 和 scipy，不会自动安装依赖。
  pause
)
endlocal
