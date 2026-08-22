<#
.SYNOPSIS
  Set up chartgen on a fresh machine. Normally launched by install.bat.

.DESCRIPTION
  Finds or installs Python, detects whether the machine has an NVIDIA GPU, picks
  the matching torch build, creates the venv and verifies the result.

  The .venv is never copied between machines — it hardcodes absolute interpreter
  and project paths. Only the ~1 MB of source travels; this rebuilds the rest.

.EXAMPLE
  .\setup.ps1                    # auto-detect GPU
  .\setup.ps1 -Cpu               # force the CPU build
  .\setup.ps1 -Training          # also install the training stack
  .\setup.ps1 -CudaTag cu128     # if your driver is new enough to prefer it
#>
param(
    [switch]$Cpu,
    [switch]$Cuda,
    [switch]$Training,
    # cu126 is the conservative default: cu128/cu129 exist for torch 2.9.1 but
    # want newer drivers. cu124 has no 2.9.1 build at all.
    [string]$CudaTag = "cu126",
    [switch]$NonInteractive
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$PYTHON_VERSION = "3.13.1"

function Say($text, $colour = "Gray") { Write-Host $text -ForegroundColor $colour }

Say "chartgen setup" "Cyan"
Say ("-" * 50)

# ---------------------------------------------------------------- 1. Python
function Find-Python {
    foreach ($candidate in @("py -3.13", "py -3.12", "py -3.14", "py -3.11", "py -3.10", "python")) {
        $parts = $candidate.Split(" ")
        try {
            $version = & $parts[0] $parts[1..$parts.Length] -c "import sys;print('%d.%d'%sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match '^3\.(1[0-4])$') {
                # tkinter is not optional here: the UI is built on it, and some
                # trimmed installs omit tcl/tk entirely.
                & $parts[0] $parts[1..$parts.Length] -c "import tkinter" 2>$null | Out-Null
                if ($LASTEXITCODE -eq 0) { return $candidate }
                Say "  $candidate has no tkinter, skipping" "DarkYellow"
            }
        } catch { }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Say "No suitable Python found (need 3.10-3.14 with tkinter)." "Yellow"
    $answer = if ($NonInteractive) { "y" } else {
        Read-Host "Download and install Python $PYTHON_VERSION now? [y/N]"
    }
    if ($answer -notmatch '^[Yy]') {
        Say "Install Python from https://www.python.org/downloads/ then re-run." "Red"
        exit 1
    }
    $installer = Join-Path $env:TEMP "python-$PYTHON_VERSION-amd64.exe"
    Say "downloading Python $PYTHON_VERSION ..."
    Invoke-WebRequest "https://www.python.org/ftp/python/$PYTHON_VERSION/python-$PYTHON_VERSION-amd64.exe" `
        -OutFile $installer -UseBasicParsing
    Say "installing (per-user, adds to PATH, includes tcl/tk) ..."
    Start-Process $installer -Wait -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_tcltk=1",
        "Include_pip=1", "Include_test=0"
    )
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "Machine")
    $python = Find-Python
    if (-not $python) {
        Say "Python installed but not visible yet. Close this window, open a new one, re-run." "Red"
        exit 1
    }
}
Say "python       $python" "Green"

# ------------------------------------------------------- 2. vendored sources
# These ship inside the archive. git is only needed if they are missing.
$repos = @{
    "vendor/audio2chart"        = "https://github.com/3podi/audio2chart.git"
    "vendor/EasyChartGenerator" = "https://github.com/Eerovil/EasyChartGenerator.git"
    "vendor/ChartFormats"       = "https://github.com/TheNathannator/GuitarGame_ChartFormats.git"
}
foreach ($path in $repos.Keys) {
    if (Test-Path $path) { continue }
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Say "$path is missing and git is not installed." "Red"
        Say "Re-copy the archive (it should contain vendor/) or install git." "Red"
        exit 1
    }
    Say "cloning $path"
    git clone --depth 1 $repos[$path] $path
}
Say "vendor       present" "Green"

# ------------------------------------------------------------------ 3. GPU
$hasNvidia = $false
try {
    $hasNvidia = [bool](Get-CimInstance Win32_VideoController |
        Where-Object { $_.Name -match "NVIDIA" })
} catch { }

if ($Cpu) { $useCuda = $false }
elseif ($Cuda) { $useCuda = $true }
else { $useCuda = $hasNvidia }

$index = if ($useCuda) { "https://download.pytorch.org/whl/$CudaTag" }
         else { "https://download.pytorch.org/whl/cpu" }
Say ("gpu          " + $(if ($hasNvidia) { "NVIDIA detected" } else { "none detected" }) +
     " -> " + $(if ($useCuda) { "CUDA ($CudaTag)" } else { "CPU build" })) "Green"

# --------------------------------------------------------------- 4. install
$parts = $python.Split(" ")
if (-not (Test-Path ".venv")) {
    Say "`ncreating .venv ..."
    & $parts[0] $parts[1..$parts.Length] -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip --quiet

Say "`ninstalling torch (this is the big one, a few minutes) ..."
& $py -m pip install torch==2.9.1 torchaudio==2.9.1 --index-url $index
if ($LASTEXITCODE -ne 0) {
    Say "torch install failed. If this is a CUDA driver mismatch, retry with:" "Red"
    Say "  .\setup.ps1 -CudaTag cu128     (or -Cpu to skip GPU entirely)" "Red"
    exit 1
}

Say "`ninstalling everything else ..."
& $py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Say "dependency install failed" "Red"; exit 1 }

if ($Training) {
    Say "`ninstalling training stack ..."
    & $py -m pip install "pytorch-lightning==2.5.6" "lightning==2.5.5" `
        "hydra-core==1.3.2" "omegaconf==2.3.0" "torchmetrics==1.8.2" "wandb==0.22.2"
}

# ----------------------------------------------------------------- 5. verify
Say "`n--- verifying ---" "Cyan"
& $py -c @"
import sys, torch, tkinter
print('python      ', sys.version.split()[0])
print('torch       ', torch.__version__)
print('tkinter      ok (the UI needs it)')
if torch.cuda.is_available():
    print('gpu         ', torch.cuda.get_device_name(0))
else:
    print('gpu          none - generating charts works, training does not')
"@
if ($LASTEXITCODE -ne 0) { Say "verification failed" "Red"; exit 1 }

& $py -m tests.test_chartgen
if ($LASTEXITCODE -ne 0) { Say "self-checks failed" "Red"; exit 1 }

Say "`nReady. Double-click chartgen.bat to open the app." "Green"
Say "First chart download the model (~1 GB) automatically." "DarkGray"
