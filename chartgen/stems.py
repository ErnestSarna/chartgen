"""Source separation shared by lyrics, solos, taps, rescue and guitar texture.

Two backends behind one contract. Callers always receive mono float32 at
`SR` under the four Demucs-era names (drums, bass, other, vocals), so no
consumer needs to know which model ran.

- "sw"     BS-RoFormer SW, six stems. Better on every stem we consume
           (vocals 11.3 vs 8.3 SDR, bass 14.6 vs 12.1, drums 14.1 vs 11.2)
           and it adds a real guitar stem (9.05), which is what made
           "is a strummed guitar playing chords here" answerable at all:
           on the local library, guitar songs measure 32-33% guitar-stem
           energy against 0.0% for synth songs, where every chart-level
           signal had measured identical. Runs ~34-48s per song on an
           RTX 3060 Ti via the bs-roformer-infer CLI (MIT code, ~700MB
           weights auto-cached under ~/.cache/bs-roformer-infer).
- "demucs" htdemucs, four stems, ~8s per song. THE DEFAULT, and the
           reference every current threshold was calibrated against.

Why Demucs is still the default, from the 12-song A/B against human-chart
ground truth (tools-era harness, 2026-08-31): SW wins where its extra
quality is measurable - vocal-stem singing detection 0.99 vs 0.96 AUC
against human lyric events, stem-rescue precision 0.28 vs 0.13 against
human note times - but shows NO measurable gain on the two consumers that
carry calibrated thresholds (solo lead separation 0.57 vs 0.60 AUC, tap
section separation 0.42 vs 0.52, both noisy and near chance for either
backend), while costing 37s against 8s. Taps v3 in particular thresholds
ABSOLUTE stem shares, and a separator swap moves absolute energy
distributions, so switching it blind would be exactly the untested
threshold change this project keeps getting burned by. SW therefore runs
where it is uniquely enabling (the guitar stem) and stays available by
name everywhere else; promoting it to default is gated on recalibrating
taps and solos, not on more opinion.

THE COMPATIBILITY POINT: SW carves guitar and piano OUT of "other", so its
"other" is not the same thing Demucs called "other" - a solo detector
z-scoring "other" energy would silently stop seeing lead guitar. So the
shim rebuilds a Demucs-equivalent residual, other = other+guitar+piano,
and exposes the parts separately as extra keys. Same semantics, better
components; consumers that want the guitar ask for it by name.

Separation failing must never fail a chart - callers treat None as "no
stems, use the old path".
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_SEPARATOR = None
_CACHE: dict[tuple, dict] = {}
_CACHE_LIMIT = 2

SR = 44100  # both backends are native 44.1k
DEMUCS_STEMS = ("drums", "bass", "other", "vocals")
SW_EXTRA = ("guitar", "piano")
# Set CHARTGEN_STEMS=demucs|sw to pin a backend (the A/B harness uses this).
BACKEND_ENV = "CHARTGEN_STEMS"


def _demucs():
    global _SEPARATOR
    if _SEPARATOR is None:
        import torch
        import demucs.api

        _SEPARATOR = demucs.api.Separator(
            model="htdemucs",
            device="cuda" if torch.cuda.is_available() else "cpu")
    return _SEPARATOR


def _separate_demucs(audio_path: str, progress) -> dict | None:
    try:
        progress("      separating stems (Demucs)")
        _, stems = _demucs().separate_audio_file(audio_path)
    except Exception as error:
        progress(f"      stem separation unavailable "
                 f"({type(error).__name__}: {error})")
        return None
    return {name: wav.mean(dim=0).cpu().numpy().astype(np.float32)
            for name, wav in stems.items()}


def _separate_sw(audio_path: str, progress) -> dict | None:
    """Six SW stems, remapped onto the Demucs contract (see module docs)."""
    try:
        import librosa
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="chartgen_sw_") as tmp:
            tmp = Path(tmp)
            (tmp / "in").mkdir()
            y, _ = librosa.load(str(audio_path), sr=SR, mono=False)
            if y.ndim == 1:
                y = np.stack([y, y])
            sf.write(str(tmp / "in" / "song.wav"), y.T, SR)
            progress("      separating stems (BS-RoFormer SW)")
            exe = Path(sys.executable).parent / "bs-roformer-infer.exe"
            run = subprocess.run(
                [str(exe) if exe.is_file() else "bs-roformer-infer",
                 "--input_folder", str(tmp / "in"),
                 "--store_dir", str(tmp / "out")],
                capture_output=True, text=True, timeout=1800)
            if run.returncode != 0:
                tail = (run.stderr.strip().splitlines() or ["separator failed"])[-1]
                progress(f"      SW separation failed ({tail[:80]})")
                return None
            raw = {}
            for name in DEMUCS_STEMS + SW_EXTRA:
                path = tmp / "out" / f"song_{name}.wav"
                if not path.is_file():
                    progress(f"      SW output missing {name}; using Demucs")
                    return None
                data, _ = sf.read(str(path), dtype="float32", always_2d=True)
                raw[name] = data.mean(axis=1)
    except Exception as error:
        progress(f"      SW separation unavailable "
                 f"({type(error).__name__}: {error})")
        return None

    n = min(len(v) for v in raw.values())
    out = {name: raw[name][:n] for name in DEMUCS_STEMS}
    # Demucs-equivalent residual: guitar and piano belong in "other" for
    # every consumer calibrated before SW existed.
    out["other"] = raw["other"][:n] + raw["guitar"][:n] + raw["piano"][:n]
    out["guitar"] = raw["guitar"][:n]
    out["piano"] = raw["piano"][:n]
    out["sw_other"] = raw["other"][:n]  # the narrow residual, for new work
    return out


def separate(audio_path: str, progress=lambda m: None,
             backend: str | None = None) -> dict | None:
    """{'vocals': mono float32 @44100, 'other': ..., ...} or None on failure.

    With SW available the dict also carries 'guitar', 'piano' and
    'sw_other'. backend: 'sw', 'demucs', or None for the env/default.
    """
    backend = backend or os.environ.get(BACKEND_ENV) or "demucs"
    key = (str(audio_path), backend)
    if key in _CACHE:
        return _CACHE[key]

    mono = None
    if backend == "sw":
        mono = _separate_sw(audio_path, progress)
        if mono is None:
            progress("      falling back to Demucs stems")
    if mono is None:
        mono = _separate_demucs(audio_path, progress)
    if mono is None:
        return None

    while len(_CACHE) >= _CACHE_LIMIT:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[key] = mono
    return mono


def activity(stem: np.ndarray, sr: int = SR, frame_s: float = 0.05):
    """(times, rms) envelope of a stem, for singing/lead detection."""
    hop = max(1, int(frame_s * sr))
    n = len(stem) // hop
    frames = stem[:n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    times = (np.arange(n) + 0.5) * frame_s
    return times, rms


def singing_share(vocal: np.ndarray, sr: int, t0: float, t1: float,
                  floor_ratio: float = 0.1) -> float:
    """Share of [t0, t1] where the vocal stem is audibly singing.

    The threshold is relative to the song's own loud singing (p90 of the
    vocal envelope), so a breathy verse still counts and stem bleed in an
    instrumental does not.
    """
    times, rms = activity(vocal, sr)
    loud = float(np.percentile(rms, 90)) or 1.0
    window = (times >= t0) & (times < t1)
    if not window.any():
        return 0.0
    return float(np.mean(rms[window] > floor_ratio * loud))
