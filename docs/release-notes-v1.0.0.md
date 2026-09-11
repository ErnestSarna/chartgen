## chartgen 1.0.0

First public release. Paste a YouTube link or drop an audio file; get a
complete Clone Hero song folder.

**Install:** download the source zip below, unzip anywhere, double-click
`install.bat`, then `chartgen.bat`. Windows 10/11; an NVIDIA GPU is strongly
recommended (about 90 s per song on an RTX 3060 Ti, ~20 min on CPU). See the
[README](https://github.com/ErnestSarna/chartgen#readme) for details.

**What you get per song**
- `notes.chart` with Expert, Hard, Medium and Easy, reduced the way human
  charters do it and calibrated against an 800-chart hand-made library
- Sustains from real note lengths, HOPOs, star power, solo markers, tap
  sections, numbered practice sections
- Synced lyrics: LRCLIB lookup word-aligned to the vocal stem, or
  faster-whisper transcription when nobody has synced the song
- `song.ini` with a 0–6 difficulty rating and a preview time, plus
  `album.png` from the file's tags or the YouTube thumbnail

**How it works:** six-stem BS-RoFormer separation, Basic Pitch transcription
on the mix and the followed instrument's stem, a real tempo map, and a chain
of chart-shaping passes (texture, playability, riff unification, section
reuse). Everything runs locally; only artist/title strings go to lrclib.net
for the lyric lookup (turn lyrics off to avoid that).

**Three front ends:** desktop app (`chartgen.bat`), CLI (`python -m
chartgen`), and a phone-friendly web UI for your private network
(`chartgen-web.bat`).

MIT licensed. Not affiliated with Clone Hero. Chart music you have the
right to use, for your own play.
