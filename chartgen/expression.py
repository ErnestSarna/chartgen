"""Sustains, star power and section markers.

None of this can come from the model. audio2chart's token vocabulary is only the
set of lanes held at each frame (32 tokens); note duration and the is5/is6/isS
flags are separate tuple fields that training discretization drops. So the model
emits bare onsets, and everything expressive has to be derived here.

Deliberately NOT generated (see README): tap notes and forced HOPO flags. Both
are written as `N 6` / `N 5` lines, which the vendored difficulty reducer treats
as ordinary notes — it would keep or drop them inconsistently per tier. Natural
HOPOs need no markup at all; they are implicit in note spacing.
"""

def hopo_threshold(resolution: int) -> int:
    """Tick gap under which a different-fret single note is already a HOPO.

    Per the .chart spec: (65/192) * resolution, rounded down — 65 ticks at the
    classic 192 resolution, 162 at ours. Notes on the same lane as the previous
    one, and chords, never become natural HOPOs regardless of spacing.
    """
    return (65 * resolution) // 192


def add_sustains(
    notes: list[tuple[int, int, int]],
    resolution: int,
    end_tick: int,
    min_gap_beats: float = 1.0,
    release_beats: float = 0.25,
    max_beats: float = 4.0,
) -> list[tuple[int, int, int]]:
    """Sustain a note when the next one is far enough away to hold through.

    Purely gap-based. An envelope test on the mixed audio was the obvious
    alternative and is worse than useless: drums, bass and vocals hold the RMS up
    whether or not the guitar is still ringing, so it would sustain everything.

    ponytail: needs an isolated guitar stem (Demucs) to actually know a note is
    held. Until then, note spacing is the honest signal.
    """
    min_gap = min_gap_beats * resolution
    release = int(release_beats * resolution)
    cap = int(max_beats * resolution)
    floor = resolution // 2  # shorter than an 8th reads as a tap, not a sustain

    ticks = sorted({t for t, _, _ in notes})
    next_tick = dict(zip(ticks, ticks[1:]))

    out = []
    for tick, lane, _ in notes:
        gap = next_tick.get(tick, end_tick) - tick
        length = 0
        if gap >= min_gap:
            length = min(gap - release, cap)
            if length < floor:
                length = 0
        out.append((tick, lane, int(max(0, length))))
    return out


def add_sustains_all_tiers(
    tiers: dict[str, list[tuple[int, int, int]]],
    resolution: int,
    end_tick: int,
    expert_tier: str = "ExpertSingle",
    **kwargs,
) -> dict[str, list[tuple[int, int, int]]]:
    """Sustain from Expert spacing, then propagate the same lengths downward.

    Expert spacing is the best proxy available for what the music is doing. If
    the song is playing 16ths, Expert has no room to sustain and neither should
    the lower tiers, even though reduction left them wide gaps to fill.

    Deriving each tier from its own spacing was the first attempt and it read
    badly: it made 39 of Easy's 40 notes sustains, turning a busy song into a
    slow one. Propagating is also always safe, because a reduced tier is a subset
    of Expert's ticks, so an inherited sustain can never reach its next note.
    """
    expert = add_sustains(tiers[expert_tier], resolution, end_tick, **kwargs)
    by_tick = {tick: length for tick, _, length in expert}
    return {
        name: expert if name == expert_tier
        else [(tick, lane, by_tick.get(tick, 0)) for tick, lane, _ in notes]
        for name, notes in tiers.items()
    }


def star_power_phrases(
    notes: list[tuple[int, int, int]],
    resolution: int,
    duration_s: float,
    beats_per_bar: int = 4,
    phrase_bars: int = 2,
    min_gap_bars: int = 6,
    min_notes: int = 4,
) -> list[tuple[int, int]]:
    """Pick bar-aligned [(start_tick, length_ticks)] star power phrases.

    Phrases go on the densest bars, spaced out so activations are spread across
    the song rather than bunched. Roughly one per 25s, which lands in the usual
    range for a hand-made chart.
    """
    bar = beats_per_bar * resolution
    ticks = sorted(t for t, _, _ in notes)
    if not ticks:
        return []

    length = phrase_bars * bar
    candidates = []
    for start in range(0, ticks[-1] + 1, bar):
        count = sum(1 for t in ticks if start <= t < start + length)
        if count >= min_notes:
            candidates.append((count, start))

    target = max(2, int(duration_s / 25))
    chosen: list[tuple[int, int]] = []
    # Densest first; tie-break on position so the result is deterministic.
    for _, start in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if all(abs(start - s) >= min_gap_bars * bar for s, _ in chosen):
            chosen.append((start, length))
            if len(chosen) >= target:
                break
    return sorted(chosen)


def solos(
    expert: list[tuple[int, int, int]],
    section_marks: list[tuple[int, str]],
    lyric_events: list[tuple[int, str]],
    resolution: int,
    max_solos: int = 2,
    min_bars: int = 2,
    max_bars: int = 24,
    min_words: int = 20,
    density_ratio: float = 1.2,
) -> list[tuple[int, int]]:
    """[(start_tick, end_tick)] solo phrases: instrumental breaks that play
    busier than the song's average, in a song that otherwise has vocals.

    The vocal requirement is what makes this precise rather than a guess: a
    solo is defined by the singer stepping back, and the lyric transcription
    says exactly where that happens. Fully instrumental songs get no solos —
    with no vocals anywhere, every section would qualify and the marker would
    mean nothing. Section boundaries come from the chroma segmentation, so a
    solo spans musically coherent bars.
    """
    words = sorted(t for t, e in lyric_events if e.startswith("lyric"))
    if len(words) < min_words or not section_marks or not expert:
        return []

    ticks = sorted({t for t, _, _ in expert})
    last = ticks[-1]
    overall = len(ticks) / max(1.0, last / resolution)  # notes per beat

    bounds = [t for t, _ in section_marks] + [last + 1]
    candidates = []
    for start, end in zip(bounds, bounds[1:]):
        beats = (end - start) / resolution
        if not (min_bars * 4 <= beats <= max_bars * 4):
            continue
        if any(start <= w < end for w in words):
            continue  # someone is singing; not a solo
        # A solo is a break BETWEEN vocal parts. An instrumental outro has no
        # words after it and an intro none before — neither is a solo
        # (playtested: a fade-out got marked and it read as plain wrong).
        if not any(w < start for w in words) or not any(w >= end for w in words):
            continue
        inside = [t for t in ticks if start <= t < end]
        if len(inside) < 16:
            continue
        density = len(inside) / beats
        if density >= density_ratio * overall:
            # Tight bounds to the notes actually played in the section.
            candidates.append((density, inside[0], inside[-1] + 1))

    candidates.sort(reverse=True)
    return sorted((s, e) for _, s, e in candidates[:max_solos])


def sections(y, sr, tempo, beats_per_bar: int = 4) -> list[tuple[int, str]]:
    """Bar-aligned practice-mode section markers from audio structure.

    Numbered, not named: chroma segmentation finds *where* the song changes but
    says nothing about whether a part is a chorus or a bridge. Guessing labels
    would be fiction, and wrong names are worse than none for navigation.
    """
    import librosa
    import numpy as np

    duration = librosa.get_duration(y=y, sr=sr)
    k = max(2, min(12, int(duration / 25)))
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        bounds = librosa.segment.agglomerative(chroma, k)
        times = librosa.frames_to_time(bounds, sr=sr)
    except Exception:
        return []  # segmentation is a nice-to-have; never fail the whole chart

    bar = beats_per_bar * tempo.resolution
    out: list[tuple[int, str]] = []
    for t in np.sort(times):
        tick = max(0, int(round(tempo.time_to_beat(float(t)) * tempo.resolution / bar)) * bar)
        if not out or tick > out[-1][0]:
            out.append((tick, f"Section {len(out) + 1}"))
    return out
