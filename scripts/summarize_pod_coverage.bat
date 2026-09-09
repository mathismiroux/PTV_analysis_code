@echo off
python "%~dp0summarize_pod_coverage.py" %*
exit /b %errorlevel%
