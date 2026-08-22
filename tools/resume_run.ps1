# Remaining queue after the pause on 2026-08-09: eval of the retrained model
# (skip-generate resumes where it stopped) and the pitch probe. Appends to the
# same night_run.log the monitors know.
$ErrorActionPreference = "Continue"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
Set-Location "C:\Users\ernes\Desktop\chartgen-transfer"
$env:PYTHONPATH = "vendor\audio2chart;vendor\EasyChartGenerator\EasyChartGenerator"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".\.venv\Scripts\python.exe"
$log = "runs\night_run.log"

function Step($name, $block) {
    "### STEP $name : start $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
    & $block *>> $log
    "### STEP $name : exit $LASTEXITCODE $(Get-Date -Format HH:mm:ss)" | Out-File -Append -Encoding utf8 $log
}

"### resume $(Get-Date)" | Out-File -Append -Encoding utf8 $log
Step "eval-ft" {
    & $py tools\evaluate.py --val data\val.json --model runs\pitch-m2\export --conditioner runs\pitch-m2\export\conditioner.pt --fret-mode model --outdir out\eval-ft --limit 10 --skip-generate
}
Step "probe-ft" {
    & $py tools\pitch_probe.py --model runs\pitch-m2\export --conditioner runs\pitch-m2\export\conditioner.pt
}
"### night run done $(Get-Date)" | Out-File -Append -Encoding utf8 $log
