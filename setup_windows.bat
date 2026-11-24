@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo MedAnnotate AI - Python 3.14.6 environment setup
echo ============================================================
py -3.14 -c "import sys, struct; print('Python:', sys.version); print('64-bit:', struct.calcsize('P')*8 == 64)" >nul 2>&1
if errorlevel 1 (
  echo ERROR: Python 3.14 was not found. Install Python 3.14 64-bit first.
  echo Run: py --list
  pause
  exit /b 1
)
if exist .venv rmdir /s /q .venv
py -3.14 -m venv .venv
if errorlevel 1 goto failed
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto failed

echo Installing NumPy binary wheel for CPython 3.14...
python -m pip install --only-binary=:all: "numpy>=2.3.4,<2.6"
if errorlevel 1 goto failed

echo Installing CPU PyTorch wheel. This avoids requiring CUDA or Visual Studio Build Tools.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed

echo Installing application dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo Verifying installed packages...
python -c "import sys, numpy, flask, torch; print('Python:',sys.version.split()[0]); print('NumPy:',numpy.__version__); print('Flask:',flask.__version__ if hasattr(flask,'__version__') else 'installed'); print('PyTorch:',torch.__version__); print('Compute device:','CUDA' if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 goto failed

echo.
echo Setup completed successfully. Run start_windows.bat to launch the application.
pause
exit /b 0
:failed
echo.
echo SETUP FAILED. Read the error above. Do not run the application until setup succeeds.
pause
exit /b 1
