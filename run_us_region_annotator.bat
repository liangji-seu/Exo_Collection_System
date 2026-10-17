@echo off
set "EXO_PYTHON=E:\miniconda\envs\EXO\python.exe"
if exist "%EXO_PYTHON%" (
  "%EXO_PYTHON%" "%~dp0run_us_region_annotator.py" %*
) else (
  python "%~dp0run_us_region_annotator.py" %*
)
