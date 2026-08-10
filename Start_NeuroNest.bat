@echo off
title NeuroNest App Launcher
echo ===================================================
echo     Starting NeuroNest Air-Gapped AI Engine...
echo ===================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
echo [ERROR] Python is not installed or not in your PATH!
echo Please install Python and try again.
pause
exit /b
)

:: 2. Check if the virtual environment exists. If not, build it!
if not exist ".venv\Scripts\activate.bat" (
echo [SETUP] First time setup detected. Building your private environment...
python -m venv .venv

echo [SETUP] Installing required AI libraries...
call .venv\Scripts\activate.bat
pip install -r requirements.txt


) else (
echo [SYSTEM] Environment found.
)

echo.
echo [SYSTEM] Waking up the Python Backend...

:: 3. Start the API server in a NEW persistent window so you can see its logs
start "NeuroNest API Server" cmd /k "call .venv\Scripts\activate.bat && python api_server.py"

echo [SYSTEM] Backend is booting up. Please wait 4 seconds...
timeout /t 4 /nobreak > nul

echo.
echo [SYSTEM] Launching 3D Interface...
:: 4. Open the HTML file in the default web browser
start index.html

echo.
echo [SUCCESS] System is live!
echo Note: A second black window has opened running your Python server.
echo Leave that window open while you chat, and close it when you are done!
timeout /t 5 > nul