"""Sustains, star power and section markers.

None of this can come from the model. audio2chart's token vocabulary is only the
set of lanes held at each frame (32 tokens); note duration and the is5/is6/isS
flags are separate tuple fields that training discretization drops. So the model
emits bare onsets, and everything expressive has to be derived here.

Tap notes (chartgen.taps) and forced HOPO flags (chartgen.motifs) are computed
elsewhere and applied at WRITE time, after reduction — the vendored difficulty
reducer treats `N 6` / `N 5` lines as ordinary notes and would keep or drop
them inconsistently per tier, so they never pass through it. Natural HOPOs
need no markup at all; they are implicit in note spacing.
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
    bpm: float | None = None,
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
    # Anything shorter reads as a tap rather than a sustain. An eighth note
    # covers that at moderate tempos, but it is only 150ms at 200 BPM — under
    # the 200ms the charting standard targets, and near the 100ms where the
    # Chorus Encore scanner flags a "baby sustain". Take the stricter floor.
    floor = resolution // 2
    if bpm:
        floor = max(floor, int(0.200 * bpm * resolution / 60.0))

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
    min_gap_bars: int = 8,
    min_notes: int = 4,
    beats_per_phrase: int = 50,
    end_guard_bars: int = 8,
) -> list[tuple[int, int]]:
    """Pick bar-aligned [(start_tick, length_ticks)] star power phrases.

    Frequency and length track what charters measurably do, cross-checked
    against the RBN/C3 spec. Measured on 800 full-ladder community charts:
    phrase length median 8 beats (two bars — twice the RBN spec's one),
    starts 44 beats apart (RBN: 40), 0.79 phrases per 40 beats (RBN: 1.0),
    and a median 48 beats left clear after the last phrase. The constants
    sit on the community medians; RBN and YARG arriving at ~40-beat spacing
    independently makes the neighbourhood trustworthy. Counting beats
    rather than seconds matters — the old one-per-25s rule under-filled
    slow songs and over-filled fast ones.

    Nothing is placed in the last 8 measures (human p10 is ~7): Clone Hero
    needs half a meter (two phrases) to activate at all, so late star power
    is meter the player can never spend. Within those rules phrases go on
    the densest bars, spaced out so activations spread across the song.
    """
    bar = beats_per_bar * resolution
    ticks = sorted(t for t, _, _ in notes)
    if not ticks:
        return []

    length = phrase_bars * bar
    last_usable = ticks[-1] - end_guard_bars * bar
    candidates = []
    for start in range(0, ticks[-1] + 1, bar):
        if start > last_usable:
            break
        count = sum(1 for t in ticks if start <= t < start + length)
        if count >= min_notes:
            candidates.append((count, start))

    played_beats = (ticks[-1] - ticks[0]) / resolution
    target = max(2, int(played_beats / beats_per_phrase))
    chosen: list[tuple[int, int]] = []
    # Densest first; tie-break on position so the result is deterministic.
    for _, start in sorted(candidates, key=lambda c: (-c[0], c[1])):
        if all(abs(start - s) >= min_gap_bars * bar for s, _ in chosen):
            chosen.append((start, length))
            if len(chosen) >= target:
                break
    return sorted(chosen)


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


# Median sustain share per tier across 49 human charts with a full ladder.
# Note it RISES as tiers get easier: fewer notes, so each is held longer.
# Propagating Expert's sustains downward unchanged left every tier flat at
# Expert's rate, which is why our Easy felt clipped next to a real one.
# Median share of positions carrying a sustain, per tier, across 800
# genre-balanced full-ladder charts (2026-08-24). Scale raised every tier
# from the 49-chart numbers (8.7/11.1/14.9/15.3%), Easy most of all -
# sparse tiers lean on held notes to stay musical.
TIER_SUSTAIN_SHARE = {
    "ExpertSingle": 0.116, "HardSingle": 0.146,
    "MediumSingle": 0.170, "EasySingle": 0.222,
}


def top_up_sustains(
    notes: list[tuple[int, int, int]],
    resolution: int,
    end_tick: int,
    target_share: float,
    release_beats: float = 0.25,
    max_beats: float = 4.0,
) -> list[tuple[int, int, int]]:
    """Sustain the roomiest notes until the tier reaches its measured share.

    Deriving sustains from a reduced tier's own gaps was the original mistake
    — it made 39 of Easy's 40 notes sustains, turning a busy song slow. The
    bound is what makes it safe now: only the notes with the most space get
    one, and only until the tier matches what human charts do, so the failure
    mode is capped by construction rather than by hoping the gaps behave.
    """
    if not notes:
        return notes

    release = int(release_beats * resolution)
    cap = int(max_beats * resolution)
    floor = resolution // 2

    ticks = sorted({t for t, _, _ in notes})
    next_tick = dict(zip(ticks, ticks[1:]))
    held = {t for t, _, sus in notes if sus > 0}
    target = int(round(target_share * len(ticks)))
    if len(held) >= target:
        return notes

    room = []
    for tick in ticks:
        if tick in held:
            continue
        gap = next_tick.get(tick, end_tick) - tick
        length = min(gap - release, cap)
        if length >= floor:
            room.append((length, tick))
    room.sort(reverse=True)

    added = {tick: length for length, tick in room[:max(0, target - len(held))]}
    return [(t, lane, added.get(t, sus) if sus == 0 else sus)
            for t, lane, sus in notes]
