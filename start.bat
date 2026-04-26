@echo off
title AI Chat - RAG Memory System

echo.
echo  ==========================================
echo   AI Chat - RAG Memory System
echo  ==========================================
echo.

if not exist "%~dp0.env" (
    echo  [WARN] .env not found!
    echo  Copy .env.example to .env and set ANTHROPIC_API_KEY
    echo.
    pause
    exit /b 1
)

python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install Python 3.10+
    pause
    exit /b 1
)

cd /d "%~dp0backend"

echo  [1/2] Installing dependencies...
python -m pip install -r requirements.txt -q
if errorlevel 1 (
    echo  [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

echo  [2/2] Starting server...
echo.
echo   URL:  http://localhost:8000
echo   API:  http://localhost:8000/docs
echo.
echo  Press Ctrl+C to stop.
echo.

set PYTHONUNBUFFERED=1
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

pause
