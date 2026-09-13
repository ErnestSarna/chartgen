@echo off
rem chartgen web UI: chart songs from your phone/laptop, get the zip back.
rem
rem Listens on http://127.0.0.1:8471 - the right thing behind a Cloudflare
rem Tunnel on this PC (see docs/remote-access.md). To reach it over Tailscale
rem or the LAN instead, run:  chartgen-web.bat --host 0.0.0.0
rem
rem Set CHARTGEN_WEB_OWNERS to the email(s) Cloudflare Access signs you in
rem with; everyone else charts as a guest whose songs are handed over and
rem deleted rather than kept in the library.
cd /d "%~dp0"
set HF_HUB_DISABLE_PROGRESS_BARS=1
.venv\Scripts\python.exe -m chartgen.web %*
pause
