"""Predict the song.ini diff_guitar tier (0-6) for a generated chart.

Coefficients come from tools/calibrate_rating.py, fitted against the local
library's human-assigned ratings with trust weighting: reputable charters
(Harmonix/Neversoft-style) count x3 and full-four-tier charts x2, because
expert-only community drops routinely misrate (measured: a chart labeled 1
that plays like a 3). Fit quality at calibration time: 78% of weighted songs
within +/-1 tier.

Falls back to built-in coefficients if rating_coeffs.json is missing, so a
fresh checkout still rates sanely.
"""
import json
from pathlib import Path

import numpy as np

FEATURE_ORDER = ("nps", "nps_peak", "chords", "fast")
# Fallback = the 2026-08-09 fit over 166 rated songs (weighted MAE 0.92).
DEFAULT_COEF = (0.316, 0.235, -0.960, 0.518, 0.572)


def _coefficients():
    path = Path(__file__).with_name("rating_coeffs.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if tuple(data["features"]) == FEATURE_ORDER:
            return np.asarray(data["coef"], dtype=float)
    except (OSError, ValueError, KeyError):
        pass
    return np.asarray(DEFAULT_COEF, dtype=float)


def features_from_notes(notes, tempo) -> dict:
    """Difficulty-relevant features of an Expert track, in seconds."""
    res = tempo.resolution
    by_tick: dict[int, int] = {}
    for tick, _, _ in notes:
        by_tick[tick] = by_tick.get(tick, 0) + 1
    ticks = sorted(by_tick)
    times = np.array([tempo.beat_to_time(t / res) for t in ticks])
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    if duration < 1:
        return {k: 0.0 for k in FEATURE_ORDER}
    gaps = np.diff(times)
    gaps = gaps[gaps > 1e-4]

    window = 5.0
    starts = np.arange(times[0], times[-1] - window, 1.0)
    counts = (np.array([np.count_nonzero((times >= s) & (times < s + window))
                        for s in starts]) / window) if len(starts) else np.array([0.0])
    return {
        "nps": len(times) / duration,
        "nps_peak": float(np.percentile(counts, 95)),
        "chords": sum(1 for c in by_tick.values() if c > 1) / len(ticks),
        "fast": float(np.mean(gaps < 0.14)) if len(gaps) else 0.0,
    }


def rate_expert(notes, tempo) -> int:
    """0-6 tier for an Expert note list, matching library conventions."""
    feats = features_from_notes(notes, tempo)
    coef = _coefficients()
    x = np.array([feats[k] for k in FEATURE_ORDER] + [1.0])
    return int(np.clip(round(float(x @ coef)), 0, 6))
