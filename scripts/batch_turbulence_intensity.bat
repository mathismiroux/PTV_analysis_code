@echo off
python "%~dp0batch_turbulence_intensity.py" %*
exit /b %errorlevel%
