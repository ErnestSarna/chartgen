"""Guitar-stem evidence via BS-RoFormer SW, and the chord-texture pass.

The signal that unblocked this file: htdemucs has no guitar stem and its
6-stem variant is unusable (guitar 5.2 SDR, poisons "other"), so "is a
strummed guitar playing chords here" was undecidable - Mary Jane and Faded
measured identically on every chart-level signal (33%/33% raw chords,
79%/82% same-shape runs), which killed the first chord-riff gate. BS-RoFormer
SW (guitar 9.05 SDR) separates them 33% vs 0.0% guitar-stem energy with
94-98% vs 0% active windows on the local library test (2026-08-31), at
~34-48s per song on the RTX 3060 Ti.

The chord-texture pass this enables attacks the largest measured fidelity
gap in the project: chord-heavy songs where humans chart 50-100% chord
positions and transcription supplies 8-30% (24-song max-chord study).
Demotion is not the thief - SUPPLY is: the mix transcription hears one
voice where the guitar plays two. So where the isolated guitar stem's own
transcription shows a second comparable voice at a tick the chart left as
a single note, the second lane is added - evidence-backed, never invented.

Separation runs as a subprocess of the bs-roformer-infer CLI (MIT, weights
auto-cached under ~/.cache/bs-roformer-infer). Everything degrades
gracefully: no package, no GPU headroom, or no guitar in the song means no
changes, same as every stems consumer.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

OPEN = 7
SW_SR = 44100

# Run the (expensive) separation only on chord-textured candidates; a song
# whose transcription is nearly chord-free has nothing to chordify.
TRIGGER_CHORD_SHARE = 0.20
# Below this guitar-stem energy share the song has no real guitar (library
# test: guitar songs 32-33%, synth songs 0.0% - the gap is enormous).
MIN_GUITAR_SHARE = 0.10
# A second voice must be comparable to the first, same ratio as the mix path.
COMPARABLE = 0.65
# Human punk tops out around 67% chord positions; never chordify past this.
MAX_CHORD_SHARE = 0.55
# Interval -> added-lane offset, per RBN chord feel: small intervals are
# "small" adjacent chords, fourths/fifths are the 1-3 power-chord default.
POWER_CHORD_SEMITONES = 5


def separate_guitar(audio_path: str, progress=lambda m: None):
    """Mono guitar stem at 44.1kHz via the SW model, or None on any failure."""
    try:
        import librosa
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="chartgen_sw_") as tmp:
            tmp = Path(tmp)
            (tmp / "in").mkdir()
            y, _ = librosa.load(str(audio_path), sr=SW_SR, mono=False)
            if y.ndim == 1:
                y = np.stack([y, y])
            sf.write(str(tmp / "in" / "song.wav"), y.T, SW_SR)
            progress("      separating guitar stem (BS-RoFormer SW)")
            exe = Path(sys.executable).parent / "bs-roformer-infer.exe"
            cmd = [str(exe) if exe.is_file() else "bs-roformer-infer",
                   "--input_folder", str(tmp / "in"),
                   "--store_dir", str(tmp / "out")]
            run = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=900)
            if run.returncode != 0:
                progress(f"      guitar stem unavailable "
                         f"({run.stderr.strip().splitlines()[-1][:80] if run.stderr.strip() else 'separator failed'})")
                return None
            out = tmp / "out" / "song_guitar.wav"
            if not out.is_file():
                return None
            g, sr = sf.read(str(out), dtype="float32", always_2d=True)
            return g.mean(axis=1)
    except Exception as error:  # never fail a chart over a bonus signal
        progress(f"      guitar stem unavailable "
                 f"({type(error).__name__}: {error})")
        return None


def guitar_share(guitar, mix_energy_stems=None) -> float:
    """Share proxy: guitar RMS energy vs a reference. Without other stems
    on hand, the ABSOLUTE activity test from the library study is used:
    share of one-second windows with real energy, scaled by mean level.
    The library separation was stark (94-98% active vs 0%), so any
    reasonable statistic clears the gap; this one needs no second model."""
    if guitar is None or not len(guitar):
        return 0.0
    n = len(guitar) // SW_SR
    if n < 8:
        return 0.0
    wins = guitar[:n * SW_SR].reshape(n, SW_SR)
    rms = np.sqrt((wins ** 2).mean(axis=1))
    peak = float(np.percentile(rms, 95))
    if peak < 1e-4:
        return 0.0
    return float((rms > 0.1 * peak).mean() * min(1.0, peak / 0.02))


def second_voices(guitar_events, tempo,
                  min_pitch: int = 40, min_amplitude: float = 0.20):
    """tick -> (root_pitch, second_pitch) where the guitar stem carries two
    comparable voices. Loudest two only - triples remain the separate,
    evidence-gated feature."""
    out = {}
    by_tick: dict[int, list] = {}
    for s, e, p, a in guitar_events:
        if p >= min_pitch and a >= min_amplitude:
            by_tick.setdefault(tempo.quantize(s, subdiv=4), []).append((a, p))
    for t, group in by_tick.items():
        group = sorted(group, reverse=True)
        comparable = [g for g in group if g[0] >= group[0][0] * COMPARABLE]
        if len(comparable) >= 2:
            pitches = sorted(p for _, p in comparable[:2])
            out[t] = (pitches[0], pitches[1])
    return out


def chordify(notes, voices, resolution: int):
    """Add the guitar's second voice to single notes the mix heard alone.

    Only at ticks where the isolated guitar stem transcribes two comparable
    voices; the added lane's direction and distance follow the interval
    (small interval = adjacent lane, fourth/fifth+ = the 1-3 power-chord
    shape). Chords the chart already has are left alone, opens are left
    alone, and the song-wide chord share is capped at the punk-band
    ceiling so no evidence glut can flood a chart.
    """
    if not notes or not voices:
        return notes, 0
    by_tick: dict[int, list] = {}
    for n in notes:
        by_tick.setdefault(n[0], []).append(n)
    ticks = sorted(by_tick)
    n_pos = len(ticks)
    chords_now = sum(
        1 for t in ticks
        if len([x for x in by_tick[t] if x[1] != OPEN]) >= 2)
    budget = int(MAX_CHORD_SHARE * n_pos) - chords_now
    if budget <= 0:
        return notes, 0

    added = []
    for t in ticks:
        if budget <= 0:
            break
        if t not in voices:
            continue
        group = by_tick[t]
        fretted = [x for x in group if x[1] != OPEN]
        if len(fretted) != 1 or any(x[1] == OPEN for x in group):
            continue
        lane = fretted[0][1]
        lo, hi = voices[t]
        offset = 2 if hi - lo >= POWER_CHORD_SEMITONES else 1
        new_lane = lane + offset if lane + offset <= 4 else lane - offset
        if not 0 <= new_lane <= 4 or new_lane == lane:
            continue
        added.append((t, new_lane, fretted[0][2]))
        budget -= 1
    if not added:
        return notes, 0
    return sorted(set(notes + added)), len(added)
