@echo off
rem Double-click this first, once, on a new machine.
rem
rem Goes through cmd rather than letting you double-click setup.ps1 directly:
rem PowerShell blocks unsigned scripts by default, so a double-clicked .ps1
rem silently does nothing. -ExecutionPolicy Bypass applies to this run only and
rem changes no machine settings.

cd /d "%~dp0"
echo.
echo  chartgen installer
echo  ------------------
echo  Sets up Python, PyTorch and dependencies. Needs an internet connection.
echo  Expect several minutes and roughly 3 GB of downloads.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*

echo.
if errorlevel 1 (
    echo  Setup FAILED - see the messages above.
) else (
    echo  Setup finished. You can now double-click chartgen.bat
)
echo.
pause
