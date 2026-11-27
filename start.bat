@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Run: uv venv --python 3.12 .venv ^&^& uv pip install --python .venv\Scripts\python.exe -r requirements.txt
    pause
    exit /b 1
)

echo Starting ClipCraft Local on http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
