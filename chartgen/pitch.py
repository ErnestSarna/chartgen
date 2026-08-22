"""Per-frame pitch features, aligned to the model's token grid.

`tools/pitch_probe.py` showed the model's fret choice does not track pitch: a
descending two-octave scale produced a correlation of -0.04. Encodec codes are
optimised for reconstruction, not for exposing pitch, so the information the
model needs to pick a fret never reaches it.

This produces that missing signal as a conditioning stream: one feature vector
per model frame, on the same grid as the chart tokens, so it can be concatenated
to the audio embeddings during training.

Same features are useful for a non-ML fret mapping, so nothing here is specific
to the training path.
"""
import numpy as np

# Standard tuning low E (82.4 Hz) up four octaves covers essentially all
# lead-guitar range; going lower just pulls in bass and kick.
FMIN_HZ = 82.41
OCTAVES = 4
BINS_PER_OCTAVE = 12
N_BINS = OCTAVES * BINS_PER_OCTAVE


def _resample_time(matrix: np.ndarray, src_times: np.ndarray, dst_times: np.ndarray):
    """Interpolate a [bins, frames] matrix onto a new time axis.

    Avoids constraining librosa's hop_length to values CQT accepts (it must be
    divisible by 2**(octaves-1)); compute at a natural hop, then land on the grid.
    """
    return np.stack([np.interp(dst_times, src_times, row) for row in matrix])


def pitch_features(y, sr, grid_ms: int, harmonic_only: bool = True):
    """(salience, f0_norm, voiced) for each grid_ms frame.

    salience: [frames, N_BINS] semitone-resolution CQT magnitude, per-frame
        normalised. Carries which pitches sound, including octave — chroma alone
        would discard register, which is exactly what decides a fret.
    f0_norm: [frames] dominant pitch in [0, 1] across the guitar range.
    voiced: [frames] bool, whether anything is actually sounding.

    voiced is a separate array rather than a sentinel in f0_norm. Encoding
    "silence" as 0.0 collides with "lowest pitch", which is the open low E — the
    most-played region on the instrument — and silently dropped every low note.

    ponytail: CQT on the full mix, so bass and vocals leak in. harmonic_only
    strips percussion, which is the cheap 80% of the fix; a Demucs guitar stem
    is the upgrade if the leakage turns out to matter.
    """
    import librosa

    if harmonic_only:
        y = librosa.effects.harmonic(y)

    cqt = np.abs(librosa.cqt(
        y, sr=sr, fmin=FMIN_HZ, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE
    ))
    src_times = librosa.frames_to_time(np.arange(cqt.shape[1]), sr=sr)

    duration = len(y) / sr
    dst_times = np.arange(0, duration, grid_ms / 1000.0)
    grid = _resample_time(cqt, src_times, dst_times)  # [bins, frames]

    # Per-frame normalise so loud passages do not dominate quiet ones.
    peak = grid.max(axis=0)
    voiced = peak > (np.median(peak) * 0.10)
    salience = np.where(peak > 0, grid / np.maximum(peak, 1e-9), 0.0)
    f0_norm = grid.argmax(axis=0) / max(1, N_BINS - 1)

    return (salience.T.astype(np.float32),
            f0_norm.astype(np.float32),
            voiced.astype(bool))


def frets_from_pitch(f0_norm, voiced, ticks, n_frets: int = 5,
                     low_percentile: float = 5.0, high_percentile: float = 95.0):
    """Map pitch to frets by ranking within the song's own range.

    Quantile binning rather than a fixed Hz->fret table: a bass-heavy song and a
    shred solo should both use the whole fretboard. Because the mapping is a pure
    function of pitch, a repeated note yields a repeated fret — the property the
    generated charts were missing (6% repeats where real parts are far higher).
    """
    last = len(f0_norm) - 1
    idx = np.asarray([min(i, last) for i in ticks])
    values = np.asarray(f0_norm)[idx]
    sounding = values[np.asarray(voiced)[idx]]
    if sounding.size == 0:
        return np.zeros(len(values), dtype=int)

    lo = float(np.percentile(sounding, low_percentile))
    hi = float(np.percentile(sounding, high_percentile))
    if hi <= lo:
        # A single sustained pitch: everything belongs on one fret.
        return np.zeros(len(values), dtype=int)

    scaled = (values - lo) / (hi - lo)
    return np.clip((scaled * n_frets).astype(int), 0, n_frets - 1)
