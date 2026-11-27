param(
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

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
    if ($cmd) {
        return Split-Path -Parent $cmd.Source
    }
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
    $venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
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

# 1. FFmpeg (required for cutting/rendering clips)
Write-Host "[1/4] Checking FFmpeg..." -ForegroundColor White
$ffDir = Find-FFmpegDir
if (-not $ffDir) {
    Install-WingetPackage 'Gyan.FFmpeg' 'FFmpeg'
    $ffDir = Find-FFmpegDir
}
if ($ffDir) {
    Write-Host "  OK FFmpeg found: $ffDir" -ForegroundColor Green
} else {
    Write-Host "  WARN FFmpeg not detected on PATH yet. Restart terminal or install from https://ffmpeg.org/download.html" -ForegroundColor Yellow
}

# 2. Node.js LTS (recommended for YouTube JS solver in yt-dlp)
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

# 3. Compatible Python (3.11 - 3.13) + .venv creation
Write-Host "[3/4] Checking Python (3.11-3.13) & virtual environment..." -ForegroundColor White
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
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
            throw "Python 3.12 could not be located after install. Download Python 3.12 from https://www.python.org/downloads/ (check 'Add Python to PATH') and run install.bat again."
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

# 4. Install Python requirements
Write-Host "[4/4] Installing all requirements from requirements.txt..." -ForegroundColor White
if ($uvCmd) {
    & uv pip install --python "$venvPy" -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "Failed installing requirements with uv." }
} else {
    & $venvPy -m pip --version 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        & $venvPy -m ensurepip --upgrade
    }
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw "Failed installing requirements with pip." }
}

# Verify core imports & warm lightweight YuNet model cache (~227 KB)
Write-Host "Verifying setup & caching lightweight face tracker..." -ForegroundColor White
& $venvPy -c "import fastapi, uvicorn, yt_dlp, faster_whisper, numpy, scipy, pysubs2, cv2, websockets; from app.face_tracker import ensure_yunet_model; p = ensure_yunet_model(); print('Verified! YuNet cached:', bool(p))"
if ($LASTEXITCODE -ne 0) { throw "Verification check failed." }

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  ALL SET! Double-click start.bat to launch ClipCraft!    " -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host ""
