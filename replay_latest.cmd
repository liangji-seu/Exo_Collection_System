@echo off
cd /d "%~dp0"
"E:\miniconda\envs\EXO\python.exe" "%~dp0run_online_replay.py"
if errorlevel 1 pause
