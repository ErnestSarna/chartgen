@echo off
rem Double-click to open the chartgen UI.
cd /d "%~dp0"

if not exist ".venv\Scripts\pythonw.exe" (
    echo.
    echo  Not set up yet on this machine.
    echo  Double-click install.bat first, then come back here.
    echo.
    pause
    exit /b 1
)

rem pythonw.exe runs with no console window. If the app cannot even start, that
rem would be invisible, so check that it imports before launching detached.
".venv\Scripts\python.exe" -c "import chartgen.app" 2>nul
if errorlevel 1 (
    echo.
    echo  The app failed to load. Details:
    echo.
    ".venv\Scripts\python.exe" -c "import chartgen.app"
    echo.
    echo  Try re-running install.bat.
    echo.
    pause
    exit /b 1
)

start "" ".venv\Scripts\pythonw.exe" -m chartgen.app
