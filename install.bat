@echo off
REM 🚀 TRAFFIC BEV PROJECT - INSTALLATION SCRIPT FOR WINDOWS

echo.
echo ========================================================
echo   Traffic BEV Project - Installation
echo ========================================================
echo.

REM 1. Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)
python --version
echo.

REM 2. Create virtual environment (optional)
if "%1"=="--venv" (
    echo [2/5] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Virtual environment activated
    echo.
)

REM 3. Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM 4. Install requirements
echo [4/5] Installing packages from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error during installation
    pause
    exit /b 1
)
echo.

REM 5. Run setup check
echo [5/5] Running setup check...
python setup_check.py
echo.

echo ========================================================
echo Installation Complete!
echo ========================================================
echo.
echo Next steps:
echo   1. Place video file in 'data/' folder
echo   2. Run: python main.py
echo.
echo Documentation: See SETUP.md
echo.
pause
