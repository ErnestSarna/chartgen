# GPU work queue: evals of the pretrained model, retrain on the enlarged
# corpus, then eval + probe of the new model. Launched detached; everything
# logs to runs\night_run.log. Written as a file because inline -Command
# quoting through Start-Process silently mangles nested quotes.
$ErrorActionPreference = "Continue"
# PS 5.1 appends redirected output as UTF-16LE by default, which turns the log
# into a mixed-encoding file that text tools read as binary.
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

"### night run start $(Get-Date)" | Out-File -Encoding utf8 $log

Step "rebuild-dataset" {
    & $py tools\build_dataset.py --root "C:\Users\ernes\Documents\Clone Hero\Songs" --out data --workers 5
}
Step "split" {
    & $py -c "import sys; sys.path.insert(0, 'vendor/audio2chart'); from dataloader.utils_dataloader import split_json_entries_by_audio_raw; split_json_entries_by_audio_raw('data/audio_dataset_with_raw.json', 'data/train.json', 'data/val.json', val_ratio=0.15, random_seed=42)"
}
# 10 songs per config: enough for medians, ~2.5x faster than the full 27.
# --skip-generate reuses charts already produced by an interrupted run.
Step "eval-pitch" {
    & $py tools\evaluate.py --val data\val.json --outdir out\eval-pitch --limit 10 --skip-generate
}
Step "eval-model" {
    & $py tools\evaluate.py --val data\val.json --fret-mode model --outdir out\eval-model --limit 10 --skip-generate
}
Step "retrain-177" {
    & $py tools\finetune.py --data data --out runs\pitch-m2 --workers 0
}
Step "eval-ft" {
    & $py tools\evaluate.py --val data\val.json --model runs\pitch-m2\export --conditioner runs\pitch-m2\export\conditioner.pt --fret-mode model --outdir out\eval-ft --limit 10 --skip-generate
}
Step "probe-ft" {
    & $py tools\pitch_probe.py --model runs\pitch-m2\export --conditioner runs\pitch-m2\export\conditioner.pt
}
"### night run done $(Get-Date)" | Out-File -Append -Encoding utf8 $log
