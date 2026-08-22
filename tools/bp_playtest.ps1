# Basic Pitch playtest batch: same songs as the champion versions, named
# "(BP)" so both engines sit side by side in the CH song list for the A/B.
$ErrorActionPreference = "Continue"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
Set-Location "C:\Users\ernes\Desktop\chartgen-transfer"
$env:PYTHONPATH = "vendor\audio2chart;vendor\EasyChartGenerator\EasyChartGenerator"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".\.venv\Scripts\python.exe"
$songs = "C:\Users\ernes\Documents\Clone Hero\Songs"

"start $(Get-Date)" | Out-File runs\bp_playtest.log
& $py -m chartgen "$songs\Alan Walker - Faded\song.opus" --engine basicpitch --artist "Alan Walker" --name "Faded (BP)" -o $songs *>> runs\bp_playtest.log
& $py -m chartgen "$songs\Calvin Harris - Summer (OHM)\song.opus" --engine basicpitch --artist "Calvin Harris" --name "Summer (BP)" -o $songs *>> runs\bp_playtest.log
& $py -m chartgen "$songs\Tom Petty And The Heartbreakers - Mary Jane's Last Dance\song.opus" --engine basicpitch --artist "Tom Petty" --name "Mary Janes Last Dance (BP)" -o $songs *>> runs\bp_playtest.log
"done-all $(Get-Date)" | Out-File -Append runs\bp_playtest.log
