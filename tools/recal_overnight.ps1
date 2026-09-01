# Launch the overnight recalibration detached from any terminal or Claude
# session, logging to work\recal_overnight.log. Survives the console closing.
#   powershell -ExecutionPolicy Bypass -File tools\recal_overnight.ps1
$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "tools\recal_overnight.py"
$log = Join-Path $root "work\recal_overnight.log"
New-Item -ItemType Directory -Force (Join-Path $root "work") | Out-Null
Start-Process -FilePath $py `
    -ArgumentList @("-u", $script, "--taps-limit", "800") `
    -WorkingDirectory $root `
    -RedirectStandardOutput $log `
    -RedirectStandardError (Join-Path $root "work\recal_overnight.err") `
    -WindowStyle Hidden
Write-Host "launched; tail with: Get-Content -Wait $log"
