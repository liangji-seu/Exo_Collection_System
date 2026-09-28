@echo off
set "SCRIPT_DIR=%~dp0"
set "EXO_PY=E:\miniconda\envs\EXO\python.exe"
if exist "%EXO_PY%" (
  "%EXO_PY%" "%SCRIPT_DIR%run_us_feature_viewer.py" %*
) else (
  python "%SCRIPT_DIR%run_us_feature_viewer.py" %*
)
