@echo off
setlocal

cd /d "%~dp0"

echo [1/4] Checking Python virtual environment...
if not exist ".venv311\Scripts\python.exe" (
    echo [ERROR] .venv311 not found.
    echo Run: py -3.11 -m venv .venv311
    pause
    exit /b 1
)

echo [2/4] Checking Redis service...
sc.exe query Redis | findstr /I "RUNNING" >nul
if errorlevel 1 (
    echo Redis is not running. Trying to start Redis...
    net start Redis >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Failed to start Redis.
        echo Please start Redis manually and run this script again.
        pause
        exit /b 1
    )
)
echo Redis is running.

echo [3/4] Starting Celery Worker...
start "Celery Worker" cmd /k ".\.venv311\Scripts\activate.bat && python -m celery -A app.celery_app worker --loglevel=info --pool=solo"

timeout /t 2 /nobreak >nul

echo [4/4] Starting FastAPI Web...
start "FastAPI Web" cmd /k ".\.venv311\Scripts\activate.bat && uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"

timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8000

echo.
echo Started successfully:
echo - Celery Worker window
echo - FastAPI Web window
echo - Browser: http://127.0.0.1:8000
echo.
echo Close the two terminal windows to stop services.
pause
