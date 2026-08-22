"""Reuse note patterns across repeated sections of a song.

Measured on the local library: human charts show a median 55% best-match
pattern overlap between similar-length sections; ours showed 13%. We re-chart
every section from scratch, so a returning chorus arrives with different frets
than the first one — the specific thing that reads as machine-made even when
every note lands on a real onset.

Charters reuse only partially, though: same-named sections overlap 28% at the
median against an 8% control, and 44% of pairs share at least half their notes.
So this does not paste a section wholesale. It keeps the later section's own
RHYTHM — which follows what the audio actually does there — and takes the
earlier section's LANE choices wherever the two line up. Where the sections
genuinely differ, the later one keeps its own material.
"""
import numpy as np

OPEN = 7


def section_signatures(y, sr, tempo, section_ticks: list[int]):
    """One normalised mean-chroma vector per section, or None where a section
    is too short to characterise. Chroma because it captures harmony while
    ignoring timbre and loudness — a chorus is the same chords each time."""
    import librosa

    if not section_ticks:
        return []
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    times = librosa.times_like(chroma, sr=sr)
    res = tempo.resolution

    signatures = []
    edges = list(section_ticks) + [None]
    for start, end in zip(edges, edges[1:]):
        t0 = tempo.beat_to_time(start / res)
        t1 = tempo.beat_to_time(end / res) if end is not None else float(times[-1])
        window = chroma[:, (times >= t0) & (times < t1)]
        if window.shape[1] < 4:
            signatures.append(None)
            continue
        mean = window.mean(axis=1)
        norm = float(np.linalg.norm(mean))
        signatures.append(mean / norm if norm else None)
    return signatures


def find_repeats(signatures, lengths, min_similarity: float = 0.95,
                 length_tolerance: float = 0.15) -> dict[int, int]:
    """{later section index: the earlier section it repeats}.

    Requires near-identical harmony AND comparable length: a chorus that runs
    twice as long is a different arrangement, and copying into it would drift
    out of alignment.
    """
    repeats: dict[int, int] = {}
    for i, sig_i in enumerate(signatures):
        if sig_i is None or lengths[i] <= 0:
            continue
        for j in range(i):
            sig_j = signatures[j]
            if sig_j is None or lengths[j] <= 0:
                continue
            ratio = lengths[i] / lengths[j]
            if not (1 - length_tolerance) <= ratio <= (1 + length_tolerance):
                continue
            if float(np.dot(sig_i, sig_j)) >= min_similarity:
                # The EARLIEST acceptable match, deliberately, not the
                # closest one: every instance of a part should trace back to
                # the same canonical section, so all the choruses inherit from
                # the first chorus rather than each from its predecessor.
                # (Whether "closest" would do better is untested — the run
                # that appeared to show it losing used an inconsistent
                # metric.)
                repeats[i] = j
                break
    return repeats


def reuse_patterns(notes, section_ticks: list[int], repeats: dict[int, int]):
    """Copy lane choices from a section onto later repeats of it.

    Reads every source pattern from the ORIGINAL notes, so a chain of repeats
    all inherit from the first instance rather than from each other.
    """
    if not notes or not repeats:
        return notes

    by_tick: dict[int, list[tuple[int, int]]] = {}
    for tick, lane, sus in notes:
        by_tick.setdefault(tick, []).append((lane, sus))

    last = max(by_tick) + 1
    edges = list(section_ticks) + [last]
    bounds = list(zip(edges, edges[1:]))
    updated = {tick: list(group) for tick, group in by_tick.items()}

    for target, source in sorted(repeats.items()):
        if target >= len(bounds) or source >= len(bounds):
            continue
        (t_start, t_end), (s_start, _) = bounds[target], bounds[source]
        offset = t_start - s_start
        for tick in [t for t in by_tick if t_start <= t < t_end]:
            origin = by_tick.get(tick - offset)
            if not origin:
                continue  # the repeat plays something the first pass did not
            # The target keeps its own sustain: spacing there is its own.
            sus = max(s for _, s in by_tick[tick])
            updated[tick] = [(lane, sus) for lane, _ in origin]

    out = {(tick, lane, sus)
           for tick, group in updated.items() for lane, sus in group}
    return sorted(out)
