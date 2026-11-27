@echo off
setlocal
cd /d "%~dp0"

set "NO_PAUSE_ARG="
if /i "%~1"=="--no-pause" set "NO_PAUSE_ARG=-NoPause"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %NO_PAUSE_ARG%
set "EXIT_CODE=%errorlevel%"

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Installation failed with code %EXIT_CODE%.
    if not defined NO_PAUSE_ARG pause
    exit /b %EXIT_CODE%
)

echo %cmdcmdline% | find /i "/c" >nul
if %errorlevel%==0 (
    if not defined NO_PAUSE_ARG pause
)
exit /b 0
