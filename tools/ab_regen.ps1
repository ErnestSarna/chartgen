# Regenerate the A/B playtest set with every calibration fix, both engines.
$ErrorActionPreference = "Continue"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
Set-Location "C:\Users\ernes\Desktop\chartgen-transfer"
$env:PYTHONPATH = "vendor\audio2chart;vendor\EasyChartGenerator\EasyChartGenerator"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".\.venv\Scripts\python.exe"
$songs = "C:\Users\ernes\Documents\Clone Hero\Songs"
$log = "runs\ab_regen.log"

"start $(Get-Date)" | Out-File -Encoding utf8 $log

$tracks = @(
  @{ Path = "$songs\Alan Walker - Faded\song.opus"; Artist = "Alan Walker"; Name = "Faded" },
  @{ Path = "$songs\Calvin Harris - Summer (OHM)\song.opus"; Artist = "Calvin Harris"; Name = "Summer" },
  @{ Path = "$songs\Tom Petty And The Heartbreakers - Mary Jane's Last Dance\song.opus"; Artist = "Tom Petty"; Name = "Mary Janes Last Dance" }
)

foreach ($t in $tracks) {
  & $py -m chartgen $t.Path --artist $t.Artist --name $t.Name -o $songs --seed 21 *>> $log
  & $py -m chartgen $t.Path --engine basicpitch --artist $t.Artist --name "$($t.Name) (BP)" -o $songs --seed 21 *>> $log
}
"done-all $(Get-Date)" | Out-File -Append -Encoding utf8 $log
