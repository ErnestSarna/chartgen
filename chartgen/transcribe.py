"""Basic Pitch engine: transcribed notes -> Expert chart notes.

The permissively-licensed replacement for the audio2chart model (Apache-2.0
weights, ONNX runtime — no TensorFlow, no EnCodec). Basic Pitch gives real
(start, end, midi_pitch, amplitude) note events, which is strictly more than
the old model offered: pitch is exact instead of inferred from CQT argmax,
chords are actual polyphony instead of sampling accidents, and sustains come
from real note durations instead of the gap heuristic.

What stands between a raw transcription and a playable chart — and what this
module does — is selection: a full mix transcribes EVERYTHING (bass, pads,
vocal melody, bells). A chart wants the lead line plus supporting chords.
"""
import numpy as np

# Below guitar low E: bass lines and kick-drum fundamentals. A guitar chart
# following the bass feels wrong even when the transcription is correct.
MIN_PITCH = 40
# Basic Pitch amplitudes run 0-1; quiet events are usually harmonic ghosts.
MIN_AMPLITUDE = 0.20
# Two, not three. Playtest: three EDM charts were unplayable inside 30
# seconds, and the human charts of the same songs use 0-10% three-note
# chords against our 6-17%.
MAX_CHORD = 2
# A chord is several strings struck TOGETHER, so its notes have comparable
# energy. A lead note over a quiet pad is a single note with accompaniment —
# charting the pad as a second button is what buried the EDM songs in chords
# (41-54% of positions against a human 5-13%).
CHORD_AMPLITUDE_RATIO = 0.65
# Open notes (lane 7, the no-fret purple strum): 60% of human charts use
# them, as PUNCTUATION - median 4.5% of positions, median run length 1,
# never inside chords. Musically they are the register BELOW the melody:
# bass drops, chugs, pedal tones. A lone note this many semitones under
# the melodic body (p25 of kept pitches) becomes an open.
OPEN = 7
OPEN_GAP_SEMITONES = 5


def transcribe(audio_path: str, progress=lambda m: None):
    """[(start_s, end_s, midi_pitch, amplitude)] via Basic Pitch (ONNX)."""
    import logging

    logging.getLogger().setLevel(logging.ERROR)  # its TF warnings are noise here
    from basic_pitch.inference import predict

    progress("      transcribing notes (Basic Pitch)")
    _, _, note_events = predict(str(audio_path))
    return [(float(s), float(e), int(p), float(a))
            for s, e, p, a, *_ in note_events]


def _fret_map(pitches: np.ndarray, n_frets: int = 5):
    """midi pitch -> fret via quantile bins over the song's own kept pitches.

    Same idea proven out in chartgen.pitch.frets_from_pitch, but on exact
    transcribed pitches instead of CQT argmax: a repeated pitch is a repeated
    fret, rising lines rise, and both verse-register and solo-register songs
    use the whole neck.
    """
    lo = float(np.percentile(pitches, 5))
    hi = float(np.percentile(pitches, 95))
    if hi <= lo:
        return lambda p: 0
    def to_fret(p):
        return int(np.clip((p - lo) / (hi - lo) * n_frets, 0, n_frets - 1))
    return to_fret


def swung_beats(events, tempo,
                min_pitch: int = MIN_PITCH,
                min_amplitude: float = MIN_AMPLITUDE) -> set[int]:
    """Beat indices whose onsets sit on the triplet grid, not the 16th grid.

    43% of the sub-16th content in the 800-chart library is 16th-triplets
    (24ths) - shuffle and swing feel, not "faster notes" - and a swung song
    hard-quantized to straight 16ths is charted wrong by the community's own
    "chart the intent" rule. Per beat, onsets vote: a position near 1/6, 1/3,
    2/3 or 5/6 of a beat fits only the triplet grid; near 1/4 or 3/4 fits
    only the 16th grid (0 and 1/2 are shared and say nothing). A beat swings
    when triplet-only onsets outnumber 16th-only ones there AND the song as
    a whole shows a real triplet population - the song gate stops one noisy
    beat from swinging an otherwise straight chart.

    Per-beat (not per-onset) grid choice is what keeps the result playable:
    two grids inside one beat can put notes 1/12 of a beat apart, which is
    brokenNote territory; whole beats on either grid keep every gap >= 1/6.

    OPT-IN (--swing), not a default. Measured on cached transcriptions of a
    straight EDM song vs a shuffle rock song: the detected beat grid's local
    phase error (librosa frame jitter plus interpolation, ~±40ms) exceeds
    the 55ms that separates a 16th slot from a triplet slot, and Basic
    Pitch's own onset timing smears the inter-onset histogram so badly that
    the straight song produced MORE triplet-only votes than straight votes
    (235 vs 133). No gate on top of votes that noisy can hold. ponytail:
    revisit if beat tracking ever gets tick-accurate.
    """
    TOL = 0.045  # beats; ~30ms at 120 BPM, tighter than half a 24th
    trip_votes: dict[int, int] = {}
    straight_votes: dict[int, int] = {}
    for start, _, pitch, amp in events:
        if pitch < min_pitch or amp < min_amplitude:
            continue
        beat = tempo.time_to_beat(start)
        b, frac = int(beat), beat % 1.0
        trip = min(abs(frac - g) for g in (1/6, 1/3, 2/3, 5/6))
        straight = min(abs(frac - g) for g in (0.25, 0.75))
        if trip <= TOL < straight:
            trip_votes[b] = trip_votes.get(b, 0) + 1
        elif straight <= TOL < trip:
            straight_votes[b] = straight_votes.get(b, 0) + 1
    total_trip = sum(trip_votes.values())
    # Song gate: a genuine shuffle produces dozens of triplet-only onsets
    # spread across many beats; jitter produces a scatter of single votes.
    if total_trip < 12 or len(trip_votes) < 8:
        return set()
    if total_trip < 0.5 * sum(straight_votes.values()):
        return set()
    return {b for b, v in trip_votes.items()
            if v >= 2 and v > straight_votes.get(b, 0)}


def expert_from_notes(
    events,
    tempo,
    subdiv: int = 4,
    min_pitch: int = MIN_PITCH,
    min_amplitude: float = MIN_AMPLITUDE,
    min_sustain_beats: float = 0.5,
    allow_opens: bool = True,
    swing_beats: set[int] = frozenset(),
    ornaments: bool = True,
) -> list[tuple[int, int, int]]:
    """Chart-ready Expert notes [(tick, lane, sustain_ticks)].

    Selection rules, in order:
    - register + amplitude filters drop bass fundamentals and harmonic ghosts
    - events quantize onto the same grid the rest of the pipeline uses
    - per tick, the loudest MAX_CHORD pitches survive; distinct frets after
      mapping form a chord (real voicings, capped for playability)
    - sustain when the transcribed note itself rings for >= 3/4 beat, released
      a 16th early, capped at 4 beats — real durations, not the gap heuristic
    """
    res = tempo.resolution
    kept = [(s, e, p, a) for s, e, p, a in events
            if p >= min_pitch and a >= min_amplitude]
    if not kept:
        return []

    pitches = np.array([p for _, _, p, _ in kept], dtype=float)
    fret_of = _fret_map(pitches)
    # Below this, a lone note is an open: clearly under the melodic body,
    # not just at the bottom of it.
    open_cut = min(float(np.percentile(pitches, 10)),
                   float(np.percentile(pitches, 25)) - OPEN_GAP_SEMITONES)

    by_tick: dict[int, list[tuple[float, int, float]]] = {}
    for start, end, pitch, amp in kept:
        # Swung beats quantize to the triplet grid (subdiv 6); everything
        # else keeps the straight grid. Chosen per whole beat, so no two
        # grids ever mix inside one (see swung_beats).
        beat_subdiv = 6 if int(tempo.time_to_beat(start)) in swing_beats else subdiv
        tick = tempo.quantize(start, subdiv=beat_subdiv)
        if tick < 0:
            continue
        by_tick.setdefault(tick, []).append((amp, pitch, end - start))

    ticks = sorted(by_tick)
    next_tick = dict(zip(ticks, ticks[1:]))
    floor = res // 2          # shorter reads as a tap, not a sustain
    release = res // 4
    cap = 4 * res

    notes: list[tuple[int, int, int]] = []
    for tick in ticks:
        group = sorted(by_tick[tick], reverse=True)[:MAX_CHORD]
        loudest = group[0][0]
        group = [g for g in group if g[0] >= loudest * CHORD_AMPLITUDE_RATIO]
        lanes: dict[int, float] = {}
        if allow_opens and len(group) == 1 and group[0][1] <= open_cut:
            # Lone low note: the purple strum. Never inside a chord - the
            # human corpus has 0.0% open+fret chords.
            lanes[OPEN] = group[0][2]
        else:
            for _, pitch, duration in group:
                lane = fret_of(pitch)
                lanes[lane] = max(lanes.get(lane, 0.0), duration)

        # One sustain for the whole position: chord members with different
        # lengths are "disjoint chords", which human charts never write
        # (median 0 per song; we accidentally wrote ~6) and the community
        # scanner flags.
        duration = max(lanes.values())
        beats_held = tempo.time_to_beat(
            tempo.beat_to_time(tick / res) + duration) - tick / res
        sustain = 0
        if beats_held >= min_sustain_beats:
            sustain = min(int(beats_held * res) - release, cap)
            limit = next_tick.get(tick)
            if limit is not None:
                sustain = min(sustain, limit - tick - release)
            if sustain < floor:
                sustain = 0
        for lane in sorted(lanes):
            notes.append((tick, lane, max(0, sustain)))
    if ornaments:
        notes = _add_ornaments(notes, kept, tempo, fret_of, swing_beats,
                               min_amplitude)
    return notes


# 32nd ornaments are the riskiest motif in the study: a flam the audio does
# not have feels broken instantly, so every gate here is deliberately tight.
ORNAMENT_AMP_RATIO = 1.25     # louder than the ghost-note floor by a margin
ORNAMENT_TOL_BEATS = 0.03     # ~20ms at 120 BPM; the onset must SIT there
ORNAMENT_CAP_PER_SONG = 12
ORNAMENT_REPEAT_BARS = 2      # the figure must recur nearby, like gallops


def _add_ornaments(notes, kept, tempo, fret_of, swing_beats, min_amplitude):
    """Recover a few 32nd grace notes the 16th grid swallowed.

    37% of human sub-16th content is 32nds, and their shape is telling:
    median burst run 3 notes - short ornamental flicks beside a main note,
    not streams. Quantization currently merges such an onset into its
    neighbour. One is re-emitted only when everything lines up: the onset
    truly sits on an odd 32nd slot (tight tolerance), lands a 32nd from an
    existing note with clear air on the other side, is louder than the
    ghost floor by a margin, maps to a DIFFERENT lane (a same-lane 32nd
    pair needs a 60ms double strum - and an invented contour is worse than
    no ornament), and the same bar-position figure recurs within two bars.
    At most one per bar and a dozen per song.
    """
    import bisect

    res = tempo.resolution
    step32 = res // 8
    bar = res * 4
    occupied = {t for t, _, _ in notes}
    if not occupied:
        return notes
    ticks = sorted(occupied)

    candidates = []
    for start, _, pitch, amp in kept:
        if amp < min_amplitude * ORNAMENT_AMP_RATIO:
            continue
        beat = tempo.time_to_beat(start)
        if int(beat) in swing_beats:
            continue  # triplet beats have their own grid
        tick32 = int(round(beat * res / step32)) * step32
        if tick32 <= 0 or tick32 % (res // 4) == 0 or tick32 in occupied:
            continue  # not a true 32nd-only slot, or already a note
        if abs(beat - tick32 / res) > ORNAMENT_TOL_BEATS:
            continue
        i = bisect.bisect_left(ticks, tick32)
        before = tick32 - ticks[i - 1] if i > 0 else bar
        after = ticks[i] - tick32 if i < len(ticks) else bar
        # a flam: exactly one 32nd from a real note, air on the other side
        if not ((before == step32 and after >= res // 4)
                or (after == step32 and before >= res // 4)):
            continue
        parent = ticks[i - 1] if before == step32 else ticks[i]
        parent_lanes = {l for t, l, _ in notes if t == parent}
        lane = fret_of(pitch)
        if lane in parent_lanes or OPEN in parent_lanes:
            continue
        candidates.append((tick32, lane))

    if not candidates:
        return notes
    # Repeat gate: the same bar-offset must appear in another bar nearby.
    by_offset: dict[int, list[int]] = {}
    for t, _ in candidates:
        by_offset.setdefault(t % bar, []).append(t // bar)
    picked = []
    used_bars = set()
    for tick32, lane in sorted(candidates):
        bars_here = by_offset[tick32 % bar]
        if not any(0 < abs(b - tick32 // bar) <= ORNAMENT_REPEAT_BARS
                   for b in bars_here):
            continue
        if tick32 // bar in used_bars or len(picked) >= ORNAMENT_CAP_PER_SONG:
            continue
        used_bars.add(tick32 // bar)
        picked.append((tick32, lane, 0))
    return sorted(notes + picked) if picked else notes


# Extended-sustain ladders: a held note keeps RINGING while higher lanes
# join on top (G held, R joins, Y stacks - the community's "building
# towards" idiom, the one sanctioned route to big chords). Library: 39% of
# the 800 charts use them, 25% meaningfully; 75% of joins sit ABOVE the
# held lane and 54% of joining notes are themselves sustained. Rate among
# charts that use them: ~1.2 joins per 100 notes.
LADDER_RATE_PER100 = 1.5      # cap just above the user-median rate
LADDER_MIN_SPACING_BEATS = 4  # human ladder accents sit ~2 bars apart
LADDER_MAX_HOLD_BEATS = 2.0
LADDER_JOIN_WINDOW = (0.2, 1.05)  # joins land a 16th to a beat after the host


def extend_ladders(notes, events, tempo,
                   min_pitch: int = MIN_PITCH,
                   min_amplitude: float = MIN_AMPLITUDE):
    """Let evidence-backed holds ring through the notes that join them.

    Everywhere else the pipeline CLAMPS a sustain at the next note. That
    made the staircase the user kept asking for - green ringing while red
    joins it - structurally impossible. The licence to overlap comes from
    the transcription itself: Basic Pitch reports real durations, so a
    note whose transcribed sound genuinely outlives the next onsets may
    keep its tail. The host qualifies by its TRANSCRIBED duration, not its
    charted sustain - the charted one was clamped at the next note and
    floored to zero, which is precisely what happens to every real ladder
    host (it rings 1.5 beats, something joins 0.25 later). Gates: the
    transcribed duration must reach past the first
    join, joins must sit on HIGHER lanes (the 75% norm - it also reads as
    a build-up, not a smear), at most three lanes ring at once, the tail
    stops a breath before the next note on the host's own lane, and
    ladders stay rare and spread out like the accents they are. Expert
    only - callers apply this after lower tiers are derived, because
    Rock-Band-lineage reducers and reduced spacing both dislike overlaps.
    """
    if not notes or not events:
        return notes, 0
    res = tempo.resolution
    by_tick: dict[int, list] = {}
    for note in notes:
        by_tick.setdefault(note[0], []).append(note)
    ticks = sorted(by_tick)
    n = len(ticks)

    # Longest true (transcribed) ring-out per charted position, in ticks.
    true_len: dict[int, int] = {}
    for start, end, pitch, amp in events:
        if pitch < min_pitch or amp < min_amplitude:
            continue
        tick = tempo.quantize(start, subdiv=4)
        if tick not in by_tick:
            continue
        length = int((tempo.time_to_beat(end) - tick / res) * res)
        if length > true_len.get(tick, 0):
            true_len[tick] = length

    next_on_lane: dict[int, dict[int, int]] = {}
    lane_last: dict[int, int] = {}
    for tick in reversed(ticks):
        next_on_lane[tick] = dict(lane_last)
        for _, lane, _ in by_tick[tick]:
            lane_last[lane] = tick

    budget = max(4, int(n * LADDER_RATE_PER100 / 100.0))
    spacing = int(LADDER_MIN_SPACING_BEATS * res)
    cap = int(LADDER_MAX_HOLD_BEATS * res)
    gap = res // 8
    lo, hi = (int(w * res) for w in LADDER_JOIN_WINDOW)

    stretch: dict[tuple[int, int], int] = {}
    moved: dict[tuple[int, int], int] = {}
    last_ladder = -spacing
    made = 0
    for i, tick in enumerate(ticks):
        if made >= budget or tick - last_ladder < spacing:
            continue
        group = by_tick[tick]
        fretted = sorted((lane, sus) for _, lane, sus in group if lane != OPEN)
        if not fretted:
            continue
        host_lane, host_sus = fretted[0]
        truth = true_len.get(tick, 0)
        if truth < (res * 3) // 4:
            continue  # the sound itself is not hold-worthy
        # joins: later positions inside the join window, strictly above the
        # host lane, no opens, that the host's real sound rings through
        joins = []          # (tick, new_lane or None to keep)
        relane: dict[int, int] = {}
        top = host_lane     # highest lane in the stack so far
        feasible = True
        for j in range(i + 1, n):
            jt = ticks[j]
            if jt - tick > hi:
                break
            lanes = [l for _, l, _ in by_tick[jt]]
            if jt - tick < lo or OPEN in lanes or min(lanes) <= host_lane:
                continue
            if truth < (jt - tick) + res // 4:
                continue
            # Library norm: 76% of joins land 1-2 lanes above the stack's
            # top - the awkward shape is one finger leaping a 3-4 lane gap
            # mid-ladder, not total width (span-3 stacks like G-Y-B are a
            # legitimate 25%). A wide SINGLE join with air after it pulls
            # inward to the highest free lane within reach, mirroring the
            # narrow-wide-chords doctrine; a wide join that cannot move (a
            # chord, or mid-line) vetoes the ladder rather than shipping
            # the leap.
            jl = min(lanes)
            if jl - top > 2:
                used = {host_lane} | {l for _, l in joins}
                target = next((c for c in (top + 2, top + 1)
                               if 0 <= c <= 4 and c not in used), None)
                after = ticks[j + 1] - jt if j + 1 < n else res
                if (target is None or len(lanes) > 1 or after < res // 2):
                    # An immovable leap as the FIRST join means the ladder
                    # itself would be the gappy shape - veto. Later, it just
                    # ends the stack: a one-join ladder is still a ladder.
                    if not joins:
                        feasible = False
                    break
                relane[jt] = target
                jl = target
            joins.append((jt, jl))
            top = max(top, jl)
            if len(joins) >= 2:  # host + two joins = three lanes ringing
                break
        if not joins or not feasible:
            continue
        limit = next_on_lane[tick].get(host_lane)
        new_sus = min(truth, cap, (limit - tick - gap) if limit else cap)
        if new_sus < (joins[0][0] - tick) + res // 4:
            continue  # the overlap would be too short to read as a ladder
        if new_sus <= host_sus:
            continue
        stretch[(tick, host_lane)] = new_sus
        for jt, target in relane.items():
            old = min(l for _, l, _ in by_tick[jt])
            moved[(jt, old)] = target
        last_ladder = tick
        made += 1

    if not stretch:
        return notes, 0
    out = [(t, moved.get((t, l), l), stretch.get((t, l), s))
           for t, l, s in notes]
    return sorted(out), made


def propagate_sustains(tiers: dict, expert_key: str = "ExpertSingle") -> dict:
    """Copy Expert's real sustains onto reduced tiers by tick (BP engine only;
    the audio2chart path derives sustains from gaps instead)."""
    by_tick = {}
    for tick, _, sus in tiers[expert_key]:
        by_tick[tick] = max(by_tick.get(tick, 0), sus)
    return {
        name: notes if name == expert_key
        else [(t, lane, by_tick.get(t, 0)) for t, lane, _ in notes]
        for name, notes in tiers.items()
    }
