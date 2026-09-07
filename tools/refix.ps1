# Regenerate the two playtest-flagged songs with the solo and fret-collapse fixes.
$ErrorActionPreference = "Continue"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
Set-Location "C:\Users\ernes\Desktop\chartgen-transfer"
$env:PYTHONPATH = "vendor\EasyChartGenerator\EasyChartGenerator"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".\.venv\Scripts\python.exe"
$dl = "C:\Users\ernes\Documents\Clone Hero\Songs\_downloads"
$songs = "C:\Users\ernes\Documents\Clone Hero\Songs"

"start $(Get-Date)" | Out-File runs\refix.log
& $py -m chartgen "$dl\QOJaol9zpNw.wav" --artist "Desed" --name "I Am Who I Thought I Was" -o $songs *>> runs\refix.log
& $py -m chartgen "$dl\eiYVcFFeYBA.wav" --artist "Arden Jones" --name "mr. sunshine" -o $songs *>> runs\refix.log
"done-all $(Get-Date)" | Out-File -Append runs\refix.log
