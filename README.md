# chartgen

Turn any song into a playable Clone Hero chart. Paste a YouTube link or drop
an audio file; chartgen listens to the recording and writes a complete song
folder — Expert through Easy, synced lyrics, star power, solos, taps, album
art — ready to drop into your Songs directory.

Everything runs on your own PC. No audio is ever uploaded.

**Website:** https://ernestsarna.github.io/chartgen/

## What it does

- **Any input.** YouTube links, whole playlists, or local MP3/FLAC/WAV/OGG/
  Opus/M4A files. Batches queue up and chart one after another; songs you
  have already charted are skipped.
- **Real transcription, not a guess.** Notes come from
  [Basic Pitch](https://github.com/spotify/basic-pitch) (exact pitches, real
  polyphony, real note durations) over a six-stem
  [BS-RoFormer](https://github.com/openmirlab/bs-roformer-infer) separation,
  so the chart follows the actual guitar, synth or piano part instead of
  whatever is loudest.
- **Full difficulty ladder.** Expert, Hard, Medium, Easy — each reduced the way
  human charters do it, calibrated against an 800-chart library of hand-made
  charts. A 0–6 `diff_guitar` rating is written for the song list.
- **The details that make a chart feel charted.** Sustains from real note
  lengths, HOPOs, star power phrases, numbered practice sections, tap
  sections on soft passages, solo markers, and riffs that repeat consistently
  every time the song does.
- **Lyrics that scroll in time.** Looked up on [LRCLIB](https://lrclib.net)
  first and word-aligned to the vocal stem; when nobody has synced the song
  yet, the vocals are transcribed with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper) instead.
- **Three ways to use it.** A desktop app, a command line, and a
  phone-friendly web page for queuing songs from the couch over your own
  network.

Guitar (5-fret) only. Songs must be at least 30 seconds.

## Install (Windows)

1. Download the latest release zip from the
   [Releases page](https://github.com/ErnestSarna/chartgen/releases) (or
   clone the repo) and unzip it anywhere.
2. **Double-click `install.bat`.** It finds or installs Python, detects
   whether you have an NVIDIA GPU, installs the matching PyTorch build and
   every dependency, then runs the self-checks. Expect several minutes and
   roughly 3 GB of downloads.
3. **Double-click `chartgen.bat`.**

The first song you chart downloads the separation and lyric models
(about 1.5 GB) automatically; after that everything is local.

**Requirements:** Windows 10/11, an internet connection for setup and for
YouTube links. Python 3.10–3.14 with tkinter (the installer offers to install
3.13 if none is found). An NVIDIA GPU is strongly recommended — a 4-minute
song takes about 90 seconds on an RTX 3060 Ti, or about 20 minutes CPU-only.

Installer flags: `install.bat -Cpu` forces the CPU build, `install.bat
-CudaTag cu128` if your driver prefers a newer CUDA build (the default is
cu126).

## Using it

**Desktop app** (`chartgen.bat`): paste a link or browse for files, point
*Save to* at your Clone Hero songs folder once (it's remembered), hit
*Generate chart*. Progress and the finished chart's stats show in the window;
*Open song folder* takes you straight to the result. Then rescan songs in
Clone Hero.

Options worth knowing: *Max difficulty* caps how hard Expert gets; *Note
grid* switches between 16ths (default), 8ths (sparser) and triplets; the
toggles turn sustains, star power, sections, HOPOs, lyrics, solos, open notes
and taps on or off. Everything defaults to on.

**Command line**, same pipeline:

```
python -m chartgen "https://www.youtube.com/watch?v=..." -o "C:\Clone Hero\Songs"
python -m chartgen song.mp3 other.flac "https://youtu.be/..." -o out
python -m chartgen --help
```

Artist and title come from the video title or the file's tags. Override them
for a single song with `--artist` and `--name`.

**Web UI** (`chartgen-web.bat`): serves a page on port 8471 where any device
on your network can paste links, upload files, watch the queue and download
finished songs as zips. It has **no login** on purpose — reach it over
Tailscale or another private network, and never port-forward it to the open
internet.

## Output

```
<Save to>/Artist - Title [chartgen]/
    notes.chart     all four difficulties, star power, sections, solos, lyrics
    song.ini        name, artist, charter, length, diff_guitar, preview time
    song.opus       the audio (mp3/ogg/opus inputs are copied as-is)
    album.png       from the file's embedded art or the YouTube thumbnail
```

Every generated song is titled "… [chartgen]" so it is always distinguishable
from a human chart in the song list.

## Privacy and network use

- Audio never leaves your machine. Separation, transcription and lyric
  alignment all run locally.
- With lyrics on (the default), the **artist and title** of each song are
  sent to lrclib.net to look up synced lyrics. Use `--no-lyrics` or the
  "Listen to the vocals" lyric source to avoid that.
- YouTube links are fetched with yt-dlp. Downloads stay in
  `<Save to>/_downloads/`.
- Model weights download once on first use (BS-RoFormer from the UVR public
  model mirror, faster-whisper from Hugging Face, the forced aligner from
  PyTorch's CDN). Stems and transcriptions are cached under
  `%APPDATA%\chartgen\cache` (up to 6 GB, oldest evicted; `CHARTGEN_CACHE_DIR`
  and `CHARTGEN_STEM_CACHE_GB` override).

## Licensing

chartgen is MIT licensed. It builds on Basic Pitch (Apache-2.0),
bs-roformer-infer and Demucs (MIT), faster-whisper (MIT), yt-dlp (Unlicense),
librosa (ISC) and PyTorch (BSD); the full pinned list is in
`requirements.txt`. Model weights are downloaded from their own sources at
first run and are not redistributed here.

Generated charts are derivative of copyrighted recordings. This is a tool for
charting music you have the right to use, for your own play — keep it local.

## For developers

```
tests/          python -m tests.test_chartgen   (also run by the installer)
docs/history.md the full engineering log: every measurement and design decision
docs/index.html the website (GitHub Pages serves docs/)
tools/          calibration studies against the human chart library
```

Manual setup, if you would rather not use the installer: create a venv,
install `torch==2.9.1 torchaudio==2.9.1` from the right PyTorch index, `pip
install -r requirements.txt`, then `pip install basic-pitch==0.4.0 --no-deps`
(its wheel asks for a TensorFlow that does not exist on modern Python; the
ONNX backend is what runs).
