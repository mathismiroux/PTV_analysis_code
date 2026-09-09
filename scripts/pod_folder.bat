@echo off
REM Default: local POD plus energy profiles. Use --analysis energy to add profiles to existing PODs.
python "%~dp0pod_folder.py" %*
exit /b %errorlevel%
