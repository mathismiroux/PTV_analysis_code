@echo off
python "%~dp0pod_spectra_folder.py" %*
exit /b %errorlevel%
