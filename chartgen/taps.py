"""Mark soft passages as tap notes. Experimental, off by default.

No charting standard documents when to tap (YARG: "use them sensibly", and
nothing else anywhere), and measurement settled that the community has no
rule at all: on 232 genre-balanced charts (76k tapped positions, 34% base
rate) the best acoustic predictor of human taps reaches 39-42% precision -
barely above chance. Human taps are charter style, not acoustics. So this
implements a chosen philosophy rather than a mimicked one: taps mark SOFT
sounds - the piano lines, plucks and gentle synth runs where a strum is
the wrong physical gesture.

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

# Softness blend. The 24-song library sweep favoured brightness; the
# 232-song genre-balanced re-sweep flipped it - attack edges out
# brightness at scale (42% vs 40% precision over a 34% base rate) and the
# two libraries disagreeing is itself the finding: no acoustic axis
# predicts human taps reliably, so the weights follow the PHILOSOPHY.
# Attack leads because "soft sounds" means soft attacks; darkness stays as
# a half-weight tiebreak; loudness earned nothing anywhere.
WEIGHTS = {"attack": 1.0, "bright": 0.5, "loud": 0.0}
# How soft, in per-song standard deviations, a note must be to join a
# tapped phrase. Mild on purpose: the phrase requirement below does the
# filtering that a harsher note-level bar would do worse.
SOFT_Z = 0.25
# A phrase: at least this many qualifying notes, no gap wider than this.
MIN_RUN = 4
MAX_GAP_BEATS = 1.0
# Never tap more of the chart than this - within MIXED sections only. A
# uniformly soft song (a piano piece) taps whole, because playtest proved
# the alternative: per-song z-scores on an all-piano song split the notes
# around the song's own average and the 40% cap truncated by rank, so
# most notes were not tapped and the boundaries fell at statistically
# arbitrary places while the piano never changed character.
MAX_SHARE = 0.40

# Absolute section texture from the Demucs drums stem, measured across
# calibration genres: acoustic/piano material sits at 0.00-0.04 drums
# share, a brushed ballad at 0.16, every full-band/electronic track at
# 0.24-0.45. Below SOFT the whole section taps; above PERC none of it
# does; between, the phrase logic decides - computed on DE-DRUMMED audio,
# because full-mix features made a soft piano note inherit the attack of
# whatever drum hit coincided with it.
SOFT_DRUMS_SHARE = 0.10
PERC_DRUMS_SHARE = 0.22


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


def drums_share_by_span(spans_s, mono, sr) -> list:
    """Drums-stem share of total stem energy per (t0, t1) span."""
    hop = int(0.05 * sr)
    envs = {}
    for name, stem in mono.items():
        n = len(stem) // hop
        envs[name] = np.sqrt((stem[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    frames = len(next(iter(envs.values())))
    out = []
    for t0, t1 in spans_s:
        a = max(0, min(frames - 1, int(t0 / 0.05)))
        b = max(a + 1, min(frames, int(t1 / 0.05)))
        total = sum(float(env[a:b].mean()) for env in envs.values())
        out.append(float(envs["drums"][a:b].mean()) / total if total > 0 else 0.0)
    return out


def detect(notes, y, sr, tempo, progress=lambda m: None,
           audio_path=None, section_marks=None) -> set[int]:
    """Tap ticks for this chart, or an empty set when nothing reads soft.

    Two-level decision. Sections classify ABSOLUTELY by their drums-stem
    share: a section with next to no percussion taps whole (a piano
    passage IS a tap texture), a percussive section never taps, and only
    genuinely mixed sections fall through to the phrase logic - which
    then runs on de-drummed audio so a note's softness is its own, not
    the coinciding drum hit's. Without stems or sections the old
    full-mix behaviour stands.
    """
    res = tempo.resolution
    ticks = sorted({t for t, _, _ in notes})
    if not ticks:
        return set()

    mono = None
    if audio_path and section_marks:
        try:
            from . import stems as stemsmod

            mono = stemsmod.separate(str(audio_path), progress)
        except Exception:
            mono = None

    if mono is None or not section_marks:
        try:
            soft = softness_by_tick(notes, y, sr, tempo)
        except Exception as error:  # decoration, never worth failing a chart
            progress(f"      tap detection skipped: {type(error).__name__}: {error}")
            return set()
        chosen = phrases(notes, soft, res)
        if chosen:
            share = len(chosen) / max(1, len(ticks))
            progress(f"      taps: {len(chosen)} notes ({share:.0%}) in soft phrases")
        return chosen

    from . import stems as stemsmod

    bounds = [t for t, _ in sorted(section_marks)] + [ticks[-1] + 1]
    spans = list(zip(bounds, bounds[1:]))
    spans_s = [(tempo.beat_to_time(a / res), tempo.beat_to_time(b / res))
               for a, b in spans]
    shares = drums_share_by_span(spans_s, mono, stemsmod.SR)

    chosen: set[int] = set()
    mixed_notes = []
    soft_sections = 0
    for (a, b), share in zip(spans, shares):
        inside = [n for n in notes if a <= n[0] < b]
        if not inside:
            continue
        if share < SOFT_DRUMS_SHARE:
            chosen.update(t for t, _, _ in inside)
            soft_sections += 1
        elif share <= PERC_DRUMS_SHARE:
            mixed_notes.extend(inside)

    if mixed_notes:
        # De-drummed audio: the melodic content without the kit, resampled
        # to the analysis rate the full-mix path used.
        try:
            import librosa

            dedrummed = (mono["vocals"] + mono["other"] + mono["bass"])
            dedrummed = librosa.resample(dedrummed, orig_sr=stemsmod.SR,
                                         target_sr=sr)
            soft = softness_by_tick(mixed_notes, dedrummed, sr, tempo)
            chosen.update(phrases(mixed_notes, soft, res))
        except Exception:
            pass

    if chosen:
        share = len(chosen) / max(1, len(ticks))
        progress(f"      taps: {len(chosen)} notes ({share:.0%}) - "
                 f"{soft_sections} soft section(s) tapped whole")
    return chosen
