# Playtest batch: four songs in the default config (pitch frets, onset
# density gate, HOPOs on), plus Faded a second time charted by the retrained
# pitch-conditioned model for a direct A/B. Everything lands in out\playtest.
$ErrorActionPreference = "Continue"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
Set-Location "C:\Users\ernes\Desktop\chartgen-transfer"
$env:PYTHONPATH = "vendor\audio2chart;vendor\EasyChartGenerator\EasyChartGenerator"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".\.venv\Scripts\python.exe"
$log = "runs\playtest.log"
$songs = "C:\Users\ernes\Documents\Clone Hero\Songs"

"### playtest batch start $(Get-Date)" | Out-File -Encoding utf8 $log

& $py -m chartgen "$songs\Alan Walker - Faded\song.opus" --artist "Alan Walker" --name "Faded [chartgen]" -o out\playtest --seed 21 *>> $log
& $py -m chartgen "$songs\Calvin Harris - Summer (OHM)\song.opus" --artist "Calvin Harris" --name "Summer [chartgen]" -o out\playtest --seed 21 *>> $log
& $py -m chartgen "$songs\Billy Squier - The Stroke\song.opus" --artist "Billy Squier" --name "The Stroke [chartgen]" -o out\playtest --seed 21 *>> $log
& $py -m chartgen "$songs\Tom Petty And The Heartbreakers - Mary Jane's Last Dance\song.opus" --artist "Tom Petty" --name "Mary Janes Last Dance [chartgen]" -o out\playtest --seed 21 *>> $log
# pitch-m, not pitch-m2: the retrain's best checkpoint landed at epoch 1 and
# probed pitch-blind (falling -0.03); the first fine-tune probed +0.42.
& $py -m chartgen "$songs\Alan Walker - Faded\song.opus" --artist "Alan Walker" --name "Faded [chartgen ft-model]" -o out\playtest --seed 21 --model runs\pitch-m\export --conditioner runs\pitch-m\export\conditioner.pt --fret-mode model *>> $log

"### playtest batch done $(Get-Date)" | Out-File -Append -Encoding utf8 $log
