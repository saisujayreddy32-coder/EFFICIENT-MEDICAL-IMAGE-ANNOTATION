@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Project environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -c "import flask, numpy, torch" >nul 2>&1
if errorlevel 1 (
  echo Required packages are missing. Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Starting MedAnnotate AI at http://127.0.0.1:8000
python run.py
pause
