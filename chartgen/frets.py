"""Reassign frets from audio pitch while keeping the model's timing.

`tools/pitch_probe.py` proved the model pitch-blind (falling-scale correlation
-0.04), and the playtest showed what that feels like: a +/-1 walk along the
fretboard with 6% fret repeats where real parts repeat constantly. The model's
*timing* is worth keeping — 85% of real onsets covered — so this swaps only the
lane decision: frets come from the CQT pitch features (`chartgen/pitch.py`,
r=0.98 against true pitch), which makes fret choice a pure function of what the
music plays. Repeated notes repeat, rising lines rise, falling lines fall.

What is preserved from the model: which ticks have notes, which are chords (the
interval shape is kept, anchored to the pitch-derived root), open notes, and
every note where the audio is not voiced (pitch is meaningless in silence, so
the model's lane is the only signal there).
"""
import numpy as np

from .pitch import frets_from_pitch

OPEN = 7


def _smooth_f0(f0_norm, voiced, width: int = 3):
    """Median-filter f0 over voiced frames only.

    CQT argmax flickers between a fundamental and its harmonics frame-to-frame;
    a 3-frame median removes single-frame octave spikes without dulling real
    note changes (which persist for many frames). Unvoiced frames are left
    untouched so they cannot drag a neighbouring note's pitch toward garbage.
    """
    f0 = np.asarray(f0_norm, dtype=np.float32).copy()
    voiced = np.asarray(voiced, dtype=bool)
    half = width // 2
    for i in np.flatnonzero(voiced):
        lo, hi = max(0, i - half), min(len(f0), i + half + 1)
        window = f0[lo:hi][voiced[lo:hi]]
        if window.size:
            f0[i] = float(np.median(window))
    return f0


def reassign_frets(
    notes: list[tuple[int, int, int]],
    tempo,
    f0_norm,
    voiced,
    grid_ms: int,
) -> list[tuple[int, int, int]]:
    """Rewrite lanes from pitch; ticks, chords, opens and sustains survive.

    Frame sampling follows tools/pitch_feature_check.py: round the onset to the
    nearest frame and take one frame of lead-in, because truncating downward
    lands on the previous note's ring-out — with a 400ms decay that is still the
    louder signal at the boundary.
    """
    if not notes:
        return notes

    res = tempo.resolution
    by_tick: dict[int, list[tuple[int, int]]] = {}
    for tick, lane, sus in notes:
        by_tick.setdefault(tick, []).append((lane, sus))

    ticks = sorted(by_tick)
    f0 = _smooth_f0(f0_norm, voiced)
    voiced = np.asarray(voiced, dtype=bool)
    last = len(f0) - 1
    frames = [
        min(last, int(round(tempo.beat_to_time(t / res) * 1000.0 / grid_ms)) + 1)
        for t in ticks
    ]
    base_frets = frets_from_pitch(f0, voiced, frames)

    res = tempo.resolution
    out: list[tuple[int, int, int]] = []
    prev_chord: tuple[int, int] | None = None  # (tick, anchor)
    for tick, frame, fret in zip(ticks, frames, base_frets):
        group = by_tick[tick]
        sus = max(s for _, s in group)
        lanes = sorted({lane for lane, _ in group})
        fretted = [l for l in lanes if l != OPEN]

        if not voiced[frame] or not fretted:
            # Silence carries no pitch, and an open note is open regardless.
            for lane in lanes:
                out.append((tick, lane, sus))
            continue

        # Keep the model's chord *shape*, anchored so its root is the pitch
        # fret. min() clamp keeps the whole shape on the neck.
        shape = [l - fretted[0] for l in fretted]
        anchor = min(int(fret), 4 - shape[-1])
        if len(fretted) > 1:
            # Playtest: chords teleporting across the neck in quick succession
            # (G+R then B+O inside a beat) are a hand shift no real chart asks
            # for at speed. Within a beat of the previous chord, the anchor may
            # move at most one lane; slower changes stay free to jump.
            if prev_chord is not None and tick - prev_chord[0] < res:
                lo, hi = prev_chord[1] - 1, prev_chord[1] + 1
                anchor = max(lo, min(hi, anchor))
                anchor = max(0, min(anchor, 4 - shape[-1]))
            prev_chord = (tick, anchor)
        for offset in shape:
            out.append((tick, anchor + offset, sus))
        if OPEN in lanes:
            out.append((tick, OPEN, sus))

    return sorted(out)


# Brightness lanes: a stretch this long stuck on this few lanes is the
# degenerate case; the centroid span gate is what protects deliberate
# monotony (palm-muted chugs barely move spectrally; a filter sweep moves
# a lot). Ratios are on raw Hz centroids, p90/p10 within the stretch.
DEGENERATE_MIN_BEATS = 8
DEGENERATE_MIN_NOTES = 6
DEGENERATE_MAX_LANES = 2
BRIGHTNESS_MIN_SPAN = 1.6
BRIGHTNESS_NOTE_WINDOW_S = 0.12


def degenerate_stretches(notes, resolution: int):
    """[(index_lo, index_hi)] over positions: long single-note runs stuck
    on <=2 distinct lanes (opens count as a lane). Chords end a run - a
    chord section has its own texture and is never 'stuck'."""
    by_tick: dict[int, list[int]] = {}
    for t, lane, _ in notes:
        by_tick.setdefault(t, []).append(lane)
    ticks = sorted(by_tick)
    out = []
    i = 0
    while i < len(ticks):
        if len(by_tick[ticks[i]]) != 1:
            i += 1
            continue
        j = i
        while j + 1 < len(ticks) and len(by_tick[ticks[j + 1]]) == 1:
            j += 1
        # maximal single-note run [i, j]; find stuck sub-stretches
        k = i
        while k <= j:
            m = k
            seen = set()
            while m <= j:
                seen_next = seen | {by_tick[ticks[m]][0]}
                if len(seen_next) > DEGENERATE_MAX_LANES:
                    break
                seen = seen_next
                m += 1
            length = m - k
            beats = (ticks[m - 1] - ticks[k]) / resolution if length > 1 else 0
            if length >= DEGENERATE_MIN_NOTES and beats >= DEGENERATE_MIN_BEATS:
                out.append((k, m - 1))
                k = m
            else:
                k += 1  # a stuck stretch may begin inside the failed span
        i = j + 1
    return out, ticks


def relane_by_brightness(notes, tempo, melodic, sr):
    """Re-lane stuck stretches from the spectral-centroid contour.

    Validated need (95-song ownership study): in bass-dominated windows
    human charts still use a median of FOUR distinct lanes - charters
    chart the filter sweep as lane movement when pitch has nothing to
    say (a growl's wub rising and falling reads as notes climbing and
    falling; that is what the ear tracks at constant f0). Our mapping is
    strictly pitch-driven, so a monotone growl riff charted one lane -
    the exact "one note repeated or opens" playtest complaint.

    Only stretches that are BOTH stuck (<=2 lanes across 8+ beats of
    single notes) AND spectrally moving (centroid p90/p10 >= 1.6 on the
    melodic stems) are touched: a palm-muted chug is stuck but static,
    and stays; a varied line is never stuck. Lanes come from quantile-
    mapped log-centroids, median-filtered so frame jitter cannot zigzag,
    and the result must actually gain variety (>=3 lanes) or the stretch
    is left alone. Ticks, sustains and counts never change; opens inside
    a re-laned stretch become fretted (they were register artifacts of
    the octave-lift, not punctuation).
    """
    import numpy as np

    if not notes or melodic is None or not len(melodic):
        return notes, 0
    stretches, ticks = degenerate_stretches(notes, tempo.resolution)
    if not stretches:
        return notes, 0
    res = tempo.resolution
    relane: dict[int, int] = {}
    changed = 0
    for lo, hi in stretches:
        cents = []
        for k in range(lo, hi + 1):
            t0 = tempo.beat_to_time(ticks[k] / res)
            a = int(t0 * sr)
            b = min(len(melodic), a + int(BRIGHTNESS_NOTE_WINDOW_S * sr))
            seg = melodic[a:b]
            if len(seg) < 256:
                cents.append(cents[-1] if cents else 0.0)
                continue
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
            power = spec.sum()
            cents.append(float((spec * freqs).sum() / power) if power > 0 else 0.0)
        cents = np.asarray(cents)
        voiced = cents[cents > 0]
        if len(voiced) < DEGENERATE_MIN_NOTES:
            continue
        p10, p90 = np.percentile(voiced, 10), np.percentile(voiced, 90)
        if p10 <= 0 or p90 / p10 < BRIGHTNESS_MIN_SPAN:
            continue  # spectrally static: a chug, not a sweep
        logc = np.log(np.maximum(cents, p10 * 0.5))
        lanes = np.clip(((logc - np.log(p10)) / (np.log(p90) - np.log(p10))
                         * 5).astype(int), 0, 4)
        # median filter width 3: frame jitter must not zigzag the hand
        smooth = lanes.copy()
        for k in range(1, len(lanes) - 1):
            smooth[k] = sorted(lanes[k - 1:k + 2])[1]
        if len(set(int(x) for x in smooth)) < 3:
            continue  # no variety gained: leave the honest monotone
        for k, lane in zip(range(lo, hi + 1), smooth):
            relane[ticks[k]] = int(lane)
        changed += 1
    if not relane:
        return notes, 0
    out = [(t, relane.get(t, lane), sus) if t in relane else (t, lane, sus)
           for t, lane, sus in notes]
    return sorted(out), changed


def smooth_fret_jumps(notes, tempo, max_step: int = 2,
                      free_gap_beats: float = 2.0):
    """Cap how far the hand travels between consecutive positions.

    Measured against human charts of the same songs, they jump three or more
    lanes on 0-1% of consecutive single notes in the LOCAL library; ours did it 9-30% of the time. (The 800-chart genre-balanced set later measured a 6.3% median - rock/metal charts jump far more than EDM - but the clamp stays: the playtests that approved this feel are the target, and it can be relaxed per-run with --max-fret-jump.)
    and made the full green-to-orange stretch on up to 8% (humans: 0.3%).
    That is the audible half of "follow whatever instrument is loudest" — the
    register flips mid-phrase and the hand teleports across the neck.

    Clamping the step while keeping its DIRECTION preserves pitch contour,
    which is what the charting guidelines actually ask for, and reproduces
    their advice to wrap a wide jump through an intermediate lane: a
    four-lane leap becomes two moves of two, so the line still arrives, just
    playably. Positions separated by a long rest are left alone — there is
    time to move the hand, and section changes should be free to relocate.
    """
    if not notes:
        return notes

    res = tempo.resolution
    by_tick: dict[int, list[tuple[int, int]]] = {}
    for tick, lane, sus in notes:
        by_tick.setdefault(tick, []).append((lane, sus))

    out: list[tuple[int, int, int]] = []
    prev_anchor = prev_tick = None
    for tick in sorted(by_tick):
        group = by_tick[tick]
        fretted = sorted(lane for lane, _ in group if lane != OPEN)
        if not fretted:
            out.extend((tick, lane, sus) for lane, sus in group)
            continue

        anchor = fretted[0]
        if prev_anchor is not None and (tick - prev_tick) / res < free_gap_beats:
            delta = anchor - prev_anchor
            if abs(delta) > max_step:
                anchor = prev_anchor + (max_step if delta > 0 else -max_step)

        # Shift the whole shape, then keep it on the neck without distorting it.
        shift = anchor - fretted[0]
        shift = max(-fretted[0], min(shift, 4 - fretted[-1]))
        for lane, sus in group:
            out.append((tick, lane if lane == OPEN else lane + shift, sus))
        prev_anchor, prev_tick = fretted[0] + shift, tick

    return sorted(out)


def step_share(notes, threshold: int = 3) -> float:
    """Share of consecutive single-note steps of at least `threshold` lanes.

    The measurement that exposed the teleporting problem, kept in the codebase
    so the pipeline can report the improvement it makes rather than claiming
    one. EDM-style charts sit at 0-1% for threshold 3; the genre-balanced median is 6.3%.
    """
    by_tick: dict[int, list[int]] = {}
    for tick, lane, _ in notes:
        by_tick.setdefault(tick, []).append(lane)
    singles = [v[0] for _, v in sorted(by_tick.items())
               if len(v) == 1 and v[0] != OPEN]
    steps = [abs(b - a) for a, b in zip(singles, singles[1:])]
    if not steps:
        return 0.0
    return sum(1 for s in steps if s >= threshold) / len(steps)
