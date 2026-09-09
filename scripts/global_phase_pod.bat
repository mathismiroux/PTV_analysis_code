@echo off
python "%~dp0global_phase_pod.py" %*
exit /b %errorlevel%
