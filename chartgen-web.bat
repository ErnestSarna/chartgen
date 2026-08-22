@echo off
rem chartgen web UI: chart songs from your phone/laptop over Tailscale.
rem Serves on port 8471; open http://<this-pc's-tailscale-name>:8471
cd /d "%~dp0"
set PYTHONPATH=vendor\audio2chart;vendor\EasyChartGenerator\EasyChartGenerator
set HF_HUB_DISABLE_PROGRESS_BARS=1
.venv\Scripts\python.exe -m chartgen.web %*
pause
