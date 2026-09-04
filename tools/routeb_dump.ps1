# Route B overnight dump: per-stem onset labels + features over the seeded
# 300-song sample (work/routeb_songs.txt). Detached and resumable: rerun
# this file and finished songs are skipped. Log: work/routeb_dump.log
$root = "C:\Users\ernes\Desktop\chartgen-transfer"
Set-Location $root
& "$root\.venv\Scripts\python.exe" "$root\tools\prominence_pilot.py" `
    --list "$root\work\routeb_songs.txt" -o "$root\work\routeb_dump.json" `
    2>&1 | Where-Object { $_ -notmatch "Warning|deprecat|warnings.warn|Predicting|sr_native|audio_original" } |
    Out-File -FilePath "$root\work\routeb_dump.log" -Encoding utf8 -Append
