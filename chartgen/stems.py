"""Source separation (Demucs htdemucs) shared by lyrics and solos.

One separation per song (~13s on the GPU for a 4-5 minute track) yields
four stems: drums, bass, other, vocals. Two consumers:

- lyrics: the vocal stem is what the words are actually sung on. LRCLIB
  gives the TEXT per line; forced alignment against the vocal stem gives
  every word its real time, replacing the synthetic per-line spread.
- solos: the vocal stem's silence is a true "nobody is singing" signal
  (replacing lyric-word counting), and the "other" stem carries the lead
  instrument the full mix buries.

The model weights are research-licensed (not MIT like the code), which is
fine for local use; a distributable build would swap the separator.
Separation failing must never fail a chart - callers treat None as "no
stems, use the old path".
"""
import numpy as np

_SEPARATOR = None
_CACHE: dict[str, dict] = {}
_CACHE_LIMIT = 2

SR = 44100  # htdemucs native rate


def _separator():
    global _SEPARATOR
    if _SEPARATOR is None:
        import torch
        import demucs.api

        _SEPARATOR = demucs.api.Separator(
            model="htdemucs",
            device="cuda" if torch.cuda.is_available() else "cpu")
    return _SEPARATOR


def separate(audio_path: str, progress=lambda m: None) -> dict | None:
    """{'vocals': mono float32 @44100, 'other': ..., ...} or None on failure."""
    key = str(audio_path)
    if key in _CACHE:
        return _CACHE[key]
    try:
        progress("      separating stems (Demucs)")
        _, stems = _separator().separate_audio_file(key)
    except Exception as error:
        progress(f"      stem separation unavailable "
                 f"({type(error).__name__}: {error})")
        return None
    mono = {name: wav.mean(dim=0).cpu().numpy().astype(np.float32)
            for name, wav in stems.items()}
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
