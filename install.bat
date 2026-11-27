: << '__POSIX_SH__'
@echo off
setlocal
cd /d "%~dp0"
set "CLIPCRAFT_ROOT=%~dp0"
set "NO_PAUSE=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$c=[IO.File]::ReadAllText('%~f0',[Text.Encoding]::UTF8); $m1='# __PS1_'+'START__'; $m2='# __PS1_'+'END__'; $s=$c.IndexOf($m1); $e=$c.IndexOf($m2); if($s -lt 0 -or $e -le $s){exit 2}; & ([scriptblock]::Create($c.Substring($s,$e-$s)))"
set "EXIT_CODE=%errorlevel%"
if %EXIT_CODE% neq 0 goto :win_fail
echo %cmdcmdline% | find /i "/c" >nul
if not %errorlevel%==0 exit /b 0
if /i "%NO_PAUSE%"=="--no-pause" exit /b 0
pause
exit /b 0

:win_fail
echo.
echo [ERROR] Installation failed with code %EXIT_CODE%.
if /i not "%NO_PAUSE%"=="--no-pause" pause
exit /b %EXIT_CODE%
__POSIX_SH__

# ==============================================================================
# macOS & Linux Universal Setup (executed by sh / bash / zsh)
# ==============================================================================
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================================="
echo "  Local ClipCraft by WPM - One-Command Setup (macOS/Linux)"
echo "=========================================================="
echo ""

OS_NAME="$(uname -s)"

find_compat_python() {
  for cand in ".venv/bin/python" python3.12 python3.13 python3.11 python3; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then
      if "$cand" -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] <= (3,13) else 1)" >/dev/null 2>&1; then
        command -v "$cand" 2>/dev/null || printf '%s\n' "$cand"
        return 0
      fi
    fi
  done
  return 1
}

run_priv() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "[ERROR] Need root/sudo privileges to install system packages: $*"
    return 1
  fi
}

echo "[1/4] Checking system tools (FFmpeg, Node.js, Python 3.11-3.13)..."
NEED_FFMPEG=0
NEED_NODE=0
NEED_PY=0
command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1 || NEED_FFMPEG=1
command -v node >/dev/null 2>&1 || NEED_NODE=1
PY_BIN="$(find_compat_python || true)"
[ -z "$PY_BIN" ] && NEED_PY=1

if [ "$NEED_FFMPEG" -eq 1 ] || [ "$NEED_NODE" -eq 1 ] || [ "$NEED_PY" -eq 1 ]; then
  if [ "$OS_NAME" = "Darwin" ]; then
    if command -v brew >/dev/null 2>&1; then
      PKGS=""
      [ "$NEED_FFMPEG" -eq 1 ] && PKGS="$PKGS ffmpeg"
      [ "$NEED_NODE" -eq 1 ] && PKGS="$PKGS node"
      [ "$NEED_PY" -eq 1 ] && PKGS="$PKGS python@3.12"
      echo "Installing missing tools via Homebrew:$PKGS ..."
      brew install $PKGS
    else
      echo "[!] Homebrew not found. Install missing tools first:"
      echo "    brew install ffmpeg node python@3.12"
      exit 1
    fi
  elif [ "$OS_NAME" = "Linux" ]; then
    if command -v apt-get >/dev/null 2>&1; then
      PKGS=""
      [ "$NEED_FFMPEG" -eq 1 ] && PKGS="$PKGS ffmpeg"
      [ "$NEED_NODE" -eq 1 ] && PKGS="$PKGS nodejs npm"
      [ "$NEED_PY" -eq 1 ] && PKGS="$PKGS python3 python3-venv python3-pip"
      echo "Installing missing tools via apt:$PKGS ..."
      run_priv apt-get update
      run_priv apt-get install -y $PKGS python3-venv
    elif command -v dnf >/dev/null 2>&1; then
      PKGS=""
      [ "$NEED_FFMPEG" -eq 1 ] && PKGS="$PKGS ffmpeg"
      [ "$NEED_NODE" -eq 1 ] && PKGS="$PKGS nodejs"
      [ "$NEED_PY" -eq 1 ] && PKGS="$PKGS python3.12"
      echo "Installing missing tools via dnf:$PKGS ..."
      run_priv dnf install -y $PKGS
    elif command -v pacman >/dev/null 2>&1; then
      PKGS=""
      [ "$NEED_FFMPEG" -eq 1 ] && PKGS="$PKGS ffmpeg"
      [ "$NEED_NODE" -eq 1 ] && PKGS="$PKGS nodejs npm"
      [ "$NEED_PY" -eq 1 ] && PKGS="$PKGS python"
      echo "Installing missing tools via pacman:$PKGS ..."
      run_priv pacman -Sy --noconfirm $PKGS
    else
      echo "[!] Unsupported Linux package manager. Install ffmpeg, nodejs, and python3.12 manually."
      exit 1
    fi
  fi
fi

echo "[2/4] Verifying Python & creating .venv..."
if [ ! -x ".venv/bin/python" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.12 --seed .venv
  else
    PY_BIN="$(find_compat_python || command -v python3 || true)"
    if [ -z "$PY_BIN" ]; then
      echo "[ERROR] Compatible Python (3.11-3.13) not found."
      exit 1
    fi
    "$PY_BIN" -m venv .venv
  fi
fi

echo "[3/4] Installing all packages from requirements.txt..."
if command -v uv >/dev/null 2>&1; then
  uv pip install --python .venv/bin/python -r requirements.txt
else
  .venv/bin/python -m ensurepip --upgrade >/dev/null 2>&1 || true
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "[4/4] Verifying imports & caching face tracker model..."
PYTHONPATH="$SCRIPT_DIR" .venv/bin/python -c "import fastapi, uvicorn, yt_dlp, faster_whisper, numpy, scipy, pysubs2, cv2, websockets; from app.face_tracker import ensure_yunet_model; p = ensure_yunet_model(); print('Verified! YuNet cached:', bool(p))"

echo ""
echo "=========================================================="
echo "  ALL SET! Start ClipCraft with: .venv/bin/python run.py  "
echo "  Then open: http://127.0.0.1:8000                        "
echo "=========================================================="
echo ""
exit 0

# __PS1_POSIX_SKIP__
: << '__PS1_POSIX_SKIP__'
# __PS1_START__
$ErrorActionPreference = 'Stop'
$Root = ($env:CLIPCRAFT_ROOT).TrimEnd('\', '/')
if (-not $Root) { $Root = (Get-Location).Path }
Set-Location -Path $Root

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Yellow
Write-Host "  Local ClipCraft by WPM - One-Command Setup (Windows)" -ForegroundColor Yellow
Write-Host "==========================================================" -ForegroundColor Yellow
Write-Host ""

function Refresh-SessionPath {
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = @()
    if ($machinePath) { $parts += $machinePath }
    if ($userPath) { $parts += $userPath }
    if ($parts.Count -gt 0) {
        $env:Path = ($parts -join ';')
    }
}

function Find-FFmpegDir {
    $cmd = Get-Command ffmpeg -ErrorAction SilentlyContinue
    if ($cmd) { return Split-Path -Parent $cmd.Source }
    $wingetPkg = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages'
    if (Test-Path $wingetPkg) {
        $hit = Get-ChildItem -Path $wingetPkg -Filter 'ffmpeg.exe' -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($hit) { return $hit.DirectoryName }
    }
    return $null
}

function Find-NodeExe {
    $cmd = Get-Command node -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($cand in @('C:\Program Files\nodejs\node.exe', 'C:\Program Files (x86)\nodejs\node.exe')) {
        if (Test-Path $cand) { return $cand }
    }
    return $null
}

function Get-PythonVersionTuple([string]$ExePath) {
    try {
        $ver = & $ExePath -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
        if ($ver -match '^(\d+)\.(\d+)$') {
            return @([int]$Matches[1], [int]$Matches[2])
        }
    } catch { }
    return $null
}

function Find-CompatiblePython {
    $candidates = @()
    $venvPy = Join-Path $Root '.venv\Scripts\python.exe'
    if (Test-Path $venvPy) { $candidates += $venvPy }

    foreach ($tag in @('-3.12', '-3.13', '-3.11')) {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            try {
                $resolved = (& py $tag -c "import sys; print(sys.executable)" 2>$null)
                if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path $resolved.Trim())) {
                    $candidates += $resolved.Trim()
                }
            } catch { }
        }
    }

    foreach ($dirName in @('Python312', 'Python313', 'Python311')) {
        $p = Join-Path $env:LOCALAPPDATA "Programs\Python\$dirName\python.exe"
        if (Test-Path $p) { $candidates += $p }
    }

    $pathPy = Get-Command python -ErrorAction SilentlyContinue
    if ($pathPy) { $candidates += $pathPy.Source }

    foreach ($cand in $candidates) {
        $vt = Get-PythonVersionTuple $cand
        if ($vt -and $vt[0] -eq 3 -and $vt[1] -ge 11 -and $vt[1] -le 13) {
            return $cand
        }
    }
    return $null
}

function Install-WingetPackage([string]$PackageId, [string]$FriendlyName) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "$FriendlyName is missing and Windows 'winget' is unavailable. Please install $FriendlyName manually, then re-run install.bat."
    }
    Write-Host "Installing $FriendlyName via winget..." -ForegroundColor Cyan
    & winget install -e --id $PackageId --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install $FriendlyName automatically (exit $LASTEXITCODE)."
    }
    Refresh-SessionPath
}

Write-Host "[1/4] Checking FFmpeg..." -ForegroundColor White
$ffDir = Find-FFmpegDir
if (-not $ffDir) {
    Install-WingetPackage 'Gyan.FFmpeg' 'FFmpeg'
    $ffDir = Find-FFmpegDir
}
if ($ffDir) {
    Write-Host "  OK FFmpeg found: $ffDir" -ForegroundColor Green
} else {
    Write-Host "  WARN FFmpeg not detected yet. Restart terminal or install from https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

Write-Host "[2/4] Checking Node.js..." -ForegroundColor White
$nodeExe = Find-NodeExe
if (-not $nodeExe) {
    Install-WingetPackage 'OpenJS.NodeJS.LTS' 'Node.js LTS'
    $nodeExe = Find-NodeExe
}
if ($nodeExe) {
    Write-Host "  OK Node.js found: $nodeExe" -ForegroundColor Green
} else {
    Write-Host "  WARN Node.js not detected; YouTube downloads work best with Node.js LTS installed." -ForegroundColor Yellow
}

Write-Host "[3/4] Checking Python (3.11-3.13) & virtual environment..." -ForegroundColor White
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
$venvPy = Join-Path $Root '.venv\Scripts\python.exe'
$compatPy = Find-CompatiblePython

if (-not (Test-Path $venvPy)) {
    if (-not $compatPy -and $uvCmd) {
        Write-Host "  Creating .venv with Python 3.12 via uv..." -ForegroundColor Cyan
        & uv venv --python 3.12 --seed .venv
        if ($LASTEXITCODE -ne 0) { throw "uv failed to create .venv." }
    } else {
        if (-not $compatPy) {
            Install-WingetPackage 'Python.Python.3.12' 'Python 3.12'
            $compatPy = Find-CompatiblePython
        }
        if (-not $compatPy) {
            throw "Python 3.12 could not be located. Download Python 3.12 from https://www.python.org/downloads/ (check 'Add Python to PATH') and run install.bat again."
        }
        Write-Host "  Creating .venv using $compatPy ..." -ForegroundColor Cyan
        if ($uvCmd) {
            & uv venv --python "$compatPy" --seed .venv
        } else {
            & $compatPy -m venv .venv
        }
        if ($LASTEXITCODE -ne 0) { throw "Failed to create .venv virtual environment." }
    }
}
Write-Host "  OK Virtual environment ready: .venv" -ForegroundColor Green

Write-Host "[4/4] Installing all requirements from requirements.txt..." -ForegroundColor White
if ($uvCmd) {
    & uv pip install --python "$venvPy" -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "Failed installing requirements with uv." }
} else {
    & $venvPy -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & $venvPy -m ensurepip --upgrade
    }
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $Root 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "Failed installing requirements with pip." }
}

Write-Host "Verifying setup & caching lightweight face tracker..." -ForegroundColor White
$env:PYTHONPATH = $Root
& $venvPy -c "import fastapi, uvicorn, yt_dlp, faster_whisper, numpy, scipy, pysubs2, cv2, websockets; from app.face_tracker import ensure_yunet_model; p = ensure_yunet_model(); print('Verified! YuNet cached:', bool(p))"
if ($LASTEXITCODE -ne 0) { throw "Verification check failed." }

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  ALL SET! Double-click start.bat to launch ClipCraft!    " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
# __PS1_END__
__PS1_POSIX_SKIP__
