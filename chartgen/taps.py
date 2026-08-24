"""Mark soft passages as tap notes. Experimental, off by default.

No charting standard documents when to tap (YARG: "use them sensibly", and
nothing else anywhere), and measurement shows the community has no single
rule either: against 24 human charts, tapped notes carry the same attack as
strummed ones (median -0.09z, a coin flip) but lean darker in timbre 2:1
(median -0.48z). So this implements a chosen philosophy rather than a
mimicked one: taps mark SOFT sounds - the piano lines, plucks and gentle
synth runs where a strum is the wrong physical gesture - scored on both the
user's axis (attack) and the community's measurable lean (darkness).

Because it is a philosophy and not a measured consensus, the feature ships
off by default; the constants are still swept against the human tap
positions (tools/sweep_taps.py) so "soft" means something calibrated
rather than guessed.

Taps are applied to PHRASES, never lone notes: a single tapped note in a
strummed line is imperceptible at best and confusing at worst, and human
taps arrive in runs. In .chart a tap is `N 6 0` at the note's tick; in CH
it plays without strumming and overrides HOPO state.
"""
import numpy as np

# Softness blend, swept against 24 human charts (9,047 tapped positions,
# a 33% base rate) in tools/sweep_taps.py. Darkness does the work: the
# best configs all put brightness first and the sweep gave loudness
# nothing. Attack keeps half weight - it is the philosophy's own axis and
# costs no agreement to keep (54% either way). The chosen config agrees
# with human taps 54% of the time at 38% recall, a 1.6x lift over the base
# rate; that is the honest ceiling for a rule the community itself only
# half-follows, and why the feature is off by default.
WEIGHTS = {"attack": 0.5, "bright": 1.0, "loud": 0.0}
# How soft, in per-song standard deviations, a note must be to join a
# tapped phrase. Mild on purpose: the phrase requirement below does the
# filtering that a harsher note-level bar would do worse.
SOFT_Z = 0.25
# A phrase: at least this many qualifying notes, no gap wider than this.
MIN_RUN = 4
MAX_GAP_BEATS = 1.0
# Never tap more of the chart than this. Human charts that tap at all
# put taps on a minority of positions; a mostly-tapped chart stops feeling
# like guitar.
MAX_SHARE = 0.40


def _samples(feature, times, when):
    """Strongest value within one frame either side, like density.py."""
    i = int(np.searchsorted(times, when))
    lo, hi = max(0, i - 2), min(len(feature), i + 3)
    return float(feature[lo:hi].max()) if hi > lo else 0.0


def softness_by_tick(notes, y, sr, tempo) -> dict[int, float]:
    """Per-position softness in per-song z-units; higher = softer."""
    import librosa

    res = tempo.resolution
    ticks = sorted({t for t, _, _ in notes})
    if len(ticks) < MIN_RUN:
        return {}

    onset = librosa.onset.onset_strength(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(onset, sr=sr)

    raw = {}
    for tick in ticks:
        when = tempo.beat_to_time(tick / res)
        raw[tick] = {
            "attack": _samples(onset, times, when),
            "bright": _samples(centroid, times, when),
            "loud": _samples(rms, times, when),
        }
    z = {}
    stats = {}
    for key in WEIGHTS:
        values = np.array([raw[t][key] for t in ticks], dtype=float)
        stats[key] = (float(values.mean()), float(values.std()) or 1e-9)
    total = sum(WEIGHTS.values())
    for tick in ticks:
        # Negated: high attack/brightness/level is HARD, so softness is the
        # weighted distance below the song's own norm.
        z[tick] = -sum(
            WEIGHTS[k] * (raw[tick][k] - stats[k][0]) / stats[k][1]
            for k in WEIGHTS
        ) / total
    return z


def phrases(notes, softness: dict[int, float], resolution: int,
            soft_z: float = None, min_run: int = None,
            max_gap_beats: float = None, max_share: float = None) -> set[int]:
    """Ticks to tap: maximal soft runs of MIN_RUN+ notes, share-capped.

    The parameters exist for tools/sweep_taps.py; production callers use
    the module constants.
    """
    soft_z = SOFT_Z if soft_z is None else soft_z
    min_run = MIN_RUN if min_run is None else min_run
    max_gap_beats = MAX_GAP_BEATS if max_gap_beats is None else max_gap_beats
    max_share = MAX_SHARE if max_share is None else max_share
    ticks = sorted({t for t, _, _ in notes})
    if not ticks:
        return set()
    max_gap = max_gap_beats * resolution

    runs, current = [], []
    for tick in ticks:
        soft = softness.get(tick, 0.0) >= soft_z
        if soft and (not current or tick - current[-1] <= max_gap):
            current.append(tick)
            continue
        if len(current) >= min_run:
            runs.append(current)
        current = [tick] if soft else []
    if len(current) >= min_run:
        runs.append(current)

    # Softest phrases first, then stop at the share cap: if the whole song
    # reads "soft", tapping all of it would just change instruments.
    runs.sort(key=lambda run: -float(np.median([softness[t] for t in run])))
    chosen: set[int] = set()
    budget = int(max_share * len(ticks))
    for run in runs:
        if len(chosen) + len(run) > budget:
            continue
        chosen.update(run)
    return chosen


def detect(notes, y, sr, tempo, progress=lambda m: None) -> set[int]:
    """Tap ticks for this chart, or an empty set when nothing reads soft."""
    try:
        soft = softness_by_tick(notes, y, sr, tempo)
    except Exception as error:  # decoration, never worth failing a chart
        progress(f"      tap detection skipped: {type(error).__name__}: {error}")
        return set()
    chosen = phrases(notes, soft, tempo.resolution)
    if chosen:
        share = len(chosen) / max(1, len({t for t, _, _ in notes}))
        progress(f"      taps: {len(chosen)} notes ({share:.0%}) in soft phrases")
    return chosen
