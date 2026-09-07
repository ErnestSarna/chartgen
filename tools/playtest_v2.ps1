# Regenerate the four default-config playtest charts with the Hard density
# cap and chord-jump clamp, then update the copies in the CH songs folder.
$ErrorActionPreference = "Continue"
$PSDefaultParameterValues["Out-File:Encoding"] = "utf8"
Set-Location "C:\Users\ernes\Desktop\chartgen-transfer"
$env:PYTHONPATH = "vendor\EasyChartGenerator\EasyChartGenerator"
$env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
$py = ".\.venv\Scripts\python.exe"
$songs = "C:\Users\ernes\Documents\Clone Hero\Songs"
$log = "runs\playtest2.log"

"start $(Get-Date)" | Out-File -Encoding utf8 $log
& $py -m chartgen "$songs\Alan Walker - Faded\song.opus" --artist "Alan Walker" --name "Faded [chartgen]" -o out\playtest --seed 21 *>> $log
& $py -m chartgen "$songs\Calvin Harris - Summer (OHM)\song.opus" --artist "Calvin Harris" --name "Summer [chartgen]" -o out\playtest --seed 21 *>> $log
& $py -m chartgen data\stroke_mix.wav --artist "Billy Squier" --name "The Stroke [chartgen]" -o out\playtest --seed 21 *>> $log
& $py -m chartgen "$songs\Tom Petty And The Heartbreakers - Mary Jane's Last Dance\song.opus" --artist "Tom Petty" --name "Mary Janes Last Dance [chartgen]" -o out\playtest --seed 21 *>> $log
# -LiteralPath everywhere: the [chartgen] in folder names is a wildcard
# character class to PowerShell, and plain Copy-Item silently matches nothing.
Get-ChildItem out\playtest -Directory | Where-Object Name -notmatch 'ft-model' |
    ForEach-Object {
        $dest = Join-Path $songs $_.Name
        New-Item -ItemType Directory -Force $dest | Out-Null
        foreach ($f in Get-ChildItem -LiteralPath $_.FullName -File) {
            Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
        }
    }
"done $(Get-Date)" | Out-File -Append -Encoding utf8 $log
