<#
.SYNOPSIS
  Zip the project for moving to another machine.

.DESCRIPTION
  Excludes .venv (1 GB, and hardcodes absolute paths so it cannot be copied),
  plus generated output and caches. What is left is about 1 MB.

  vendor/ IS included: it is gitignored, so cloning the repo elsewhere would not
  bring it, and audio2chart has no license — keep this archive to yourself.

.EXAMPLE
  .\package.ps1
  .\package.ps1 -Destination D:\chartgen.zip
#>
param(
    [string]$Destination = "$PSScriptRoot\..\chartgen-transfer.zip",
    [switch]$IncludeOutput
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$skip = @('\.venv\\', '__pycache__', '\.pyc$', '\\work\\', '\.git\\')
if (-not $IncludeOutput) { $skip += '\\out\\' }
$pattern = ($skip -join '|')

$files = Get-ChildItem . -Recurse -File |
    Where-Object { $_.FullName -notmatch $pattern }

$staging = Join-Path $env:TEMP "chartgen-pkg-$PID"
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
foreach ($file in $files) {
    $relative = $file.FullName.Substring($PSScriptRoot.Length).TrimStart('\')
    $target = Join-Path $staging $relative
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    Copy-Item $file.FullName $target
}

if (Test-Path $Destination) { Remove-Item $Destination -Force }
Compress-Archive -Path "$staging\*" -DestinationPath $Destination
Remove-Item $staging -Recurse -Force

$size = (Get-Item $Destination).Length / 1MB
Write-Host ("packaged {0} files -> {1} ({2:N2} MB)" -f $files.Count, $Destination, $size)
Write-Host ""
Write-Host "On the other machine:"
Write-Host "  1. unzip it anywhere"
Write-Host "  2. double-click install.bat   (installs Python if missing, picks"
Write-Host "                                 CUDA or CPU torch automatically)"
Write-Host "  3. double-click chartgen.bat"
Write-Host ""
Write-Host "install.bat needs internet: ~3 GB of packages, plus ~1 GB of model"
Write-Host "on the first chart. For the training stack, run install.bat -Training."
