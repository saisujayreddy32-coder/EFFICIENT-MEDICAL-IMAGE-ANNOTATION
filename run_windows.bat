@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  call setup_windows.bat
  if errorlevel 1 exit /b 1
)
call start_windows.bat
