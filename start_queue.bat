@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

rem ============================================================
rem  HuaLi_garbage_system Windows launcher
rem
rem  Modes:
rem    start_queue.bat              -> setup + web + celery (PyO3 mode)
rem    start_queue.bat all          -> same as default
rem    start_queue.bat web          -> setup + web only
rem    start_queue.bat worker       -> setup + celery only
rem    start_queue.bat setup        -> create venv + install deps + build PyO3
rem    start_queue.bat rust-http    -> setup + optional Rust HTTP fallback service
rem    start_queue.bat check        -> import checks only
rem    start_queue.bat help         -> show help
rem ============================================================

set "MODE=%~1"
if not defined MODE set "MODE=all"
if /I "%MODE%"=="default" set "MODE=all"

set "ROOT_DIR=%~dp0"
set "LOG_DIR=%ROOT_DIR%logs"
set "VENV_DIR=.venv"
set "VENV_PYTHON=%ROOT_DIR%.venv\Scripts\python.exe"
set "RUST_HTTP_EXE=%ROOT_DIR%rust\target\release\huali_garbage_server.exe"
set "WEB_URL=http://127.0.0.1:8000"
set "REDIS_HOST=127.0.0.1"
set "REDIS_PORT=6379"
set "RUST_SERVICE_URL=http://127.0.0.1:50051"
set "UPLOADS_DIR=%ROOT_DIR%..\HuaLi_garbage_runtime\uploads"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

if /I "%MODE%"=="help" goto :help

echo ============================================================
echo   HuaLi_garbage_system launcher
echo   Mode: %MODE%
echo   Root: %ROOT_DIR%
echo ============================================================
echo.

call :ensure_venv || goto :fail
call :ensure_python || goto :fail

if /I "%MODE%"=="setup" (
    call :install_python_deps || goto :fail
    call :build_pyo3_extension || goto :fail
    call :verify_runtime || goto :fail
    echo.
    echo [OK] Setup completed.
    pause
    exit /b 0
)

if /I "%MODE%"=="check" (
    call :verify_runtime || goto :fail
    echo.
    echo [OK] Runtime imports succeeded.
    pause
    exit /b 0
)

call :install_python_deps_if_needed || goto :fail
call :build_pyo3_extension_if_needed || goto :fail
call :verify_runtime || goto :fail
call :check_redis || goto :fail

if /I "%MODE%"=="web" (
    call :start_web || goto :fail
    echo.
    echo [OK] Web started.
    pause
    exit /b 0
)

if /I "%MODE%"=="worker" (
    call :start_worker || goto :fail
    echo.
    echo [OK] Worker started.
    pause
    exit /b 0
)

if /I "%MODE%"=="rust-http" (
    call :build_rust_http_binary || goto :fail
    call :start_rust_http || goto :fail
    echo.
    echo [OK] Rust HTTP fallback service started.
    pause
    exit /b 0
)

rem default/all
call :start_worker || goto :fail
call :start_web || goto :fail
echo.
echo [OK] Started successfully.
echo   - Web:    %WEB_URL%
echo   - Redis:  redis://%REDIS_HOST%:%REDIS_PORT%/0
echo   - Uploads:%UPLOADS_DIR%
echo   - Logs:   %LOG_DIR%
echo.
echo Tips:
echo   - Default runtime uses PyO3 in-process Rust extension.
echo   - To run Rust HTTP fallback service separately: start_queue.bat rust-http
echo.
pause
exit /b 0

:help
echo Usage:
echo   start_queue.bat [all^|web^|worker^|setup^|check^|rust-http^|help]
echo.
echo Modes:
echo   all       Setup, build PyO3, then start FastAPI + Celery worker
echo   web       Setup, build PyO3, then start FastAPI only
echo   worker    Setup, build PyO3, then start Celery worker only
echo   setup     Create venv, install Python deps, build PyO3 extension
echo   check     Verify imports for FastAPI/Celery/PyO3 extension
echo   rust-http Build/start optional Rust HTTP fallback service
echo.
pause
exit /b 0

:ensure_venv
if exist "%VENV_PYTHON%" exit /b 0
echo [INFO] Creating virtual environment...
where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -m venv "%VENV_DIR%" >nul 2>&1
)
if not exist "%VENV_PYTHON%" (
    python -m venv "%VENV_DIR%" >nul 2>&1
)
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)
exit /b 0

:ensure_python
if exist "%VENV_PYTHON%" exit /b 0
echo [ERROR] Python executable not found: %VENV_PYTHON%
exit /b 1

:install_python_deps_if_needed
call "%VENV_PYTHON%" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2, maturin" >nul 2>&1
if not errorlevel 1 exit /b 0
call :install_python_deps || exit /b 1
exit /b 0

:install_python_deps
echo [INFO] Installing Python dependencies...
call "%VENV_PYTHON%" -m pip install --upgrade pip >"%LOG_DIR%\pip_upgrade.log" 2>&1
if errorlevel 1 (
    echo [ERROR] pip upgrade failed. See logs\pip_upgrade.log
    exit /b 1
)
call "%VENV_PYTHON%" -m pip install -r requirements.txt >"%LOG_DIR%\pip_install.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. See logs\pip_install.log
    exit /b 1
)
exit /b 0

:build_pyo3_extension_if_needed
call "%VENV_PYTHON%" -c "import huali_garbage_core" >nul 2>&1
if not errorlevel 1 exit /b 0
call :build_pyo3_extension || exit /b 1
exit /b 0

:build_pyo3_extension
echo [INFO] Building Rust PyO3 extension with maturin...
call "%VENV_PYTHON%" -m maturin develop --manifest-path rust\Cargo.toml --release >"%LOG_DIR%\maturin_build.log" 2>&1
if errorlevel 1 (
    echo [ERROR] PyO3 build failed. See logs\maturin_build.log
    exit /b 1
)
exit /b 0

:build_rust_http_binary
if exist "%RUST_HTTP_EXE%" exit /b 0
echo [INFO] Building Rust HTTP fallback binary...
cargo build --release --manifest-path "%ROOT_DIR%rust\Cargo.toml" >"%LOG_DIR%\rust_http_build.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Rust HTTP build failed. See logs\rust_http_build.log
    exit /b 1
)
if not exist "%RUST_HTTP_EXE%" (
    echo [ERROR] Rust HTTP executable not found after build.
    exit /b 1
)
exit /b 0

:verify_runtime
echo [INFO] Verifying runtime imports...
call "%VENV_PYTHON%" -c "import fastapi, uvicorn, celery, redis, sqlalchemy, cv2, huali_garbage_core; import app.main; print('ok')" >"%LOG_DIR%\runtime_check.log" 2>&1
if errorlevel 1 (
    echo [ERROR] Runtime verification failed. See logs\runtime_check.log
    exit /b 1
)
exit /b 0

:check_redis
echo [INFO] Checking Redis connectivity...
call "%VENV_PYTHON%" -c "import socket; s=socket.create_connection(('%REDIS_HOST%', %REDIS_PORT%), timeout=1); s.close(); print('ok')" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Redis is not reachable at redis://%REDIS_HOST%:%REDIS_PORT%/0
    echo         Please start Redis first.
    exit /b 1
)
exit /b 0

:start_web
echo [INFO] Starting FastAPI web...
start "FastAPI Web" cmd /k "cd /d \"%ROOT_DIR%\" && set PYTHONUTF8=1 && set UPLOADS_DIR=%UPLOADS_DIR% && \"%VENV_PYTHON%\" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload"
exit /b 0

:start_worker
echo [INFO] Starting Celery worker...
start "Celery Worker" cmd /k "cd /d \"%ROOT_DIR%\" && set PYTHONUTF8=1 && set UPLOADS_DIR=%UPLOADS_DIR% && \"%VENV_PYTHON%\" -m celery -A app.celery_app worker --loglevel=info --pool=solo"
exit /b 0

:start_rust_http
echo [INFO] Starting Rust HTTP fallback service...
start "Rust HTTP Service" cmd /k "cd /d \"%ROOT_DIR%\" && set RUST_SERVICE_HOST=127.0.0.1 && set RUST_SERVICE_PORT=50051 && \"%RUST_HTTP_EXE%\""
exit /b 0

:fail
echo.
echo [ERROR] Startup failed.
echo         Please inspect logs under %LOG_DIR%
pause
exit /b 1
