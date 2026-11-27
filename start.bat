@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo First-time setup detected! Running one-command installer...
    call "%~dp0install.bat" --no-pause
    if errorlevel 1 (
        echo.
        echo [ERROR] Auto-install failed. Fix the error above and try again.
        pause
        exit /b 1
    )
)

echo Starting Local ClipCraft by WPM on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
