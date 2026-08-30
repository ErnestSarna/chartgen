"""Self-check for the flow-motif layer and riff unification.

Run: python -m tests.test_motifs
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.chart import write_chart  # noqa: E402
from chartgen.motifs import (  # noqa: E402
    OPEN,
    break_machine_gun,
    chunk_rolls,
    legato_forcing,
    legato_stairs,
    pickup_roots,
    settle_pushes,
    shape_chord_walks,
    CHORD_RUNGS,
)
from chartgen.structure import unify_riff_bars  # noqa: E402
from chartgen.tempo import RESOLUTION, TempoMap  # noqa: E402
from chartgen.transcribe import (  # noqa: E402
    expert_from_notes,
    extend_ladders,
    swung_beats,
)

RES = RESOLUTION  # 480


def lanes_at(notes, tick):
    return sorted(lane for t, lane, _ in notes if t == tick)


def sus_at(notes, tick):
    return max((s for t, _, s in notes if t == tick), default=None)


def invariants(notes, res=RES):
    """Properties every transform must preserve."""
    seen = set()
    by_tick = {}
    for t, lane, sus in notes:
        assert t >= 0 and sus >= 0, (t, sus)
        assert 0 <= lane <= 4 or lane == OPEN, lane
        assert (t, lane) not in seen, f"duplicate note {(t, lane)}"
        seen.add((t, lane))
        by_tick.setdefault(t, []).append(sus)
    ticks = sorted(by_tick)
    for a, b in zip(ticks, ticks[1:]):
        assert max(by_tick[a]) <= b - a, \
            f"sustain at {a} overlaps next note at {b}"


# ---------------------------------------------------------------- stairs

def test_stairs_hold_stepped_phrase():
    beat = RES
    notes = [(0, 0, 0), (beat, 1, 0), (2 * beat, 2, 0), (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=120.0)
    # every phrase note but the last is held to just before its successor
    for tick in (0, beat, 2 * beat):
        s = sus_at(out, tick)
        assert s and s >= RES // 2, (tick, s)
        assert s < beat, s
    assert sus_at(out, 3 * beat) == 0
    invariants(out)


def test_stairs_fire_on_stepped_eighths_at_moderate_tempo():
    # The shipped-bug regression: at 84 BPM an 8th is 357ms and holding it
    # to ~80ms before the next note leaves ~270ms - over the 200ms floor,
    # so stepped 8th phrases MUST produce stairs (the human chart's own
    # stair sustains there are ~236ms).
    eighth = RES // 2
    notes = [(i * eighth, [0, 1, 2, 1, 0][i], 0) for i in range(5)]
    out = legato_stairs(notes, RES, bpm=84.0)
    held = [t for t, _, s in out if s > 0]
    assert len(held) >= 3, out


def test_stairs_leave_short_runs_and_wide_leaps():
    beat = RES
    # only two stepped notes: below STAIR_MIN_RUN
    out = legato_stairs([(0, 0, 0), (beat, 1, 0)], RES, bpm=120.0)
    assert all(s == 0 for _, _, s in out)
    # a 3-lane leap is not a stair
    out = legato_stairs([(0, 0, 0), (beat, 3, 0), (2 * beat, 4, 0),
                         (3 * beat, 3, 0)], RES, bpm=120.0)
    assert sus_at(out, 0) == 0


def test_stairs_respect_existing_and_chords():
    beat = RES
    long_sus = 2 * beat
    notes = [(0, 0, long_sus), (beat, 1, 0), (2 * beat, 2, 0), (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=120.0)
    assert sus_at(out, 0) == long_sus  # never shrink a real sustain
    # a chord breaks the phrase
    notes = [(0, 0, 0), (beat, 1, 0), (beat, 2, 0), (2 * beat, 2, 0),
             (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=120.0)
    assert sus_at(out, 0) == 0


def test_stairs_too_fast_tempo_no_baby_sustains():
    # At 240 BPM a beat is 250ms; holding to ~80ms before the next note
    # leaves ~170ms - under the 200ms floor, so no stair at all.
    beat = RES
    notes = [(0, 0, 0), (beat, 1, 0), (2 * beat, 2, 0), (3 * beat, 3, 0)]
    out = legato_stairs(notes, RES, bpm=240.0)
    assert all(s == 0 for _, _, s in out)


# ---------------------------------------------------------------- rolls

def test_chunk_rolls_structures_a_shapeless_run():
    step = RES // 4
    lanes = [0, 2, 0, 3, 1, 4, 0, 3, 1, 2]
    notes = [(i * step, lane, 0) for i, lane in enumerate(lanes)]
    out = chunk_rolls(notes, RES)
    new = [lanes_at(out, i * step)[0] for i in range(len(lanes))]
    assert new != lanes, "run should have been restructured"
    deltas = [b - a for a, b in zip(new, new[1:])]
    assert all(d != 0 for d in deltas), f"no same-lane repeats inside a run: {new}"
    nz = [d for d in deltas if d != 0]
    reversals = sum(1 for a, b in zip(nz, nz[1:]) if a * b < 0)
    assert reversals <= 3, f"still zigzagging: {new}"
    assert all(abs(d) <= 2 for d in deltas), f"jump introduced: {new}"
    invariants(out)


def test_chunk_rolls_preserves_deliberate_patterns():
    step = RES // 4
    trill = [(i * step, [0, 1][i % 2], 0) for i in range(8)]
    assert chunk_rolls(trill, RES) == sorted(trill)
    pedal = [(i * step, [0, 2, 0, 3, 0, 4][i % 6], 0) for i in range(12)]
    assert chunk_rolls(pedal, RES) == sorted(pedal)
    gun = [(i * step, 2, 0) for i in range(8)]
    assert chunk_rolls(gun, RES) == sorted(gun)
    # an already-directional line stays
    roll = [(i * step, i % 5, 0) for i in range(10)]
    assert chunk_rolls(roll, RES) == sorted(roll)


def test_chunk_rolls_leaves_slow_and_short_material():
    beat = RES
    slow = [(i * beat * 2, [0, 3, 1, 4, 2, 0][i], 0) for i in range(6)]
    assert chunk_rolls(slow, RES) == sorted(slow)
    short = [(i * (RES // 4), [0, 3, 1, 4][i], 0) for i in range(4)]
    assert chunk_rolls(short, RES) == sorted(short)


# ---------------------------------------------------------------- machine gun

def test_machine_gun_thins_one_off_wall():
    step = RES // 4
    wall = [(i * step, 2, 0) for i in range(10)]
    out = break_machine_gun(wall, RES)
    kept = {t for t, _, _ in out}
    assert len(kept) < 10
    assert all(t % (RES // 2) == 0 for t in kept), "on-8th pulse must survive"
    invariants(out)


def test_machine_gun_keeps_recurring_chug():
    step = RES // 4
    bar = 4 * RES
    riff = [(i * step, 2, 0) for i in range(10)]
    again = [(bar + i * step, 2, 0) for i in range(10)]
    out = break_machine_gun(riff + again, RES)
    assert len(out) == 20, "a chug that recurs is deliberate"


def test_machine_gun_ignores_short_and_varied_runs():
    step = RES // 4
    short = [(i * step, 2, 0) for i in range(6)]
    assert break_machine_gun(short, RES) == sorted(short)
    varied = [(i * step, [2, 2, 2, 2, 3, 2, 2, 2][i % 8], 0) for i in range(10)]
    assert break_machine_gun(varied, RES) == sorted(varied)


# ---------------------------------------------------------------- pushes

def test_settle_pushes_snaps_lone_stray():
    eighth = RES // 2
    push = 2 * RES + RES // 4  # off-8th 16th, air on both sides
    notes = [(0, 0, 0), (RES, 1, 0), (push, 2, 0), (3 * RES, 3, 0)]
    out = settle_pushes(notes, RES)
    assert lanes_at(out, push) == [], "stray push should have moved"
    snapped = int(round(push / eighth)) * eighth
    assert lanes_at(out, snapped) == [2]
    invariants(out)


def test_settle_pushes_keeps_recurring_syncopation():
    bar = 4 * RES
    offset = 2 * RES + RES // 4
    notes = []
    for b in range(3):
        notes += [(b * bar, 0, 0), (b * bar + offset, 2, 0)]
    out = settle_pushes(notes, RES)
    for b in range(3):
        assert lanes_at(out, b * bar + offset) == [2], \
            "recurring push is deliberate syncopation"


def test_settle_pushes_ignores_notes_inside_runs():
    step = RES // 4
    run = [(RES + i * step, [0, 1, 2, 1][i % 4], 0) for i in range(8)]
    assert settle_pushes(run, RES) == sorted(run)


# ---------------------------------------------------------------- forcing

def test_forcing_flags_stepwise_eighth_phrase():
    eighth = RES // 2
    lanes = [0, 1, 2, 3, 2, 1]
    notes = [(i * eighth, lane, 0) for i, lane in enumerate(lanes)]
    forced = legato_forcing(notes, RES)
    assert 0 not in forced, "phrase-initial note keeps its strum"
    assert forced == {i * eighth for i in range(1, 6)}, forced


def test_forcing_never_touches_natural_hopos_or_chords():
    sixteenth = RES // 4  # under the HOPO threshold: already natural HOPOs
    fast = [(i * sixteenth, i % 3, 0) for i in range(8)]
    assert legato_forcing(fast, RES) == set()
    eighth = RES // 2
    chords = [(i * eighth, lane, 0) for i in range(6) for lane in (0, 1)]
    assert legato_forcing(chords, RES) == set()
    same = [(i * eighth, 2, 0) for i in range(6)]
    assert legato_forcing(same, RES) == set()


def test_forcing_respects_budget():
    eighth = RES // 2
    # one long phrase: 40 stepwise notes - the budget (max of 8 and 6% of
    # notes) truncates it to a legato prefix rather than skipping it.
    lanes = [0, 1, 2, 1, 0, 1, 2, 3, 2, 1] * 4
    notes = [(i * eighth, lane, 0) for i, lane in enumerate(lanes)]
    forced = legato_forcing(notes, RES)
    assert 0 < len(forced) <= 8, len(forced)
    assert 0 not in forced


# ---------------------------------------------------------------- chords

def test_chord_walk_takes_the_rung_ladder():
    beat = RES
    walk = []
    for i, anchor in enumerate((3, 2, 1, 0)):  # descending anchors
        walk += [(i * beat, anchor, 0), (i * beat, anchor + 1, 0)]
    out = shape_chord_walks(walk, RES)
    shapes = [tuple(lanes_at(out, i * beat)) for i in range(4)]
    for a, b in zip(shapes, shapes[1:]):
        assert b in [tuple(r) for r in CHORD_RUNGS], b
        moved = len(set(a) ^ set(b)) // 2
        assert moved == 1, f"{a}->{b} should move exactly one finger"
    anchors = [s[0] for s in shapes]
    assert anchors == sorted(anchors, reverse=True), "descent must survive"
    invariants(out)


def test_chord_walk_leaves_riffs_and_short_runs():
    beat = RES
    riff = [(i * beat, lane, 0) for i in range(4) for lane in (0, 1)]
    assert shape_chord_walks(riff, RES) == sorted(riff)
    two = [(0, 0, 0), (0, 1, 0), (beat, 1, 0), (beat, 2, 0)]
    assert shape_chord_walks(two, RES) == sorted(two)


def test_pickup_takes_next_chord_root():
    eighth = RES // 2
    notes = [(0, 3, 0), (4 * RES - eighth, 4, 0),
             (4 * RES, 1, 0), (4 * RES, 2, 0)]
    out = pickup_roots(notes, RES)
    assert lanes_at(out, 4 * RES - eighth) == [1], "pickup takes the root"
    # inside a run it is not a pickup
    run = [(4 * RES - 4 * eighth + i * eighth, 4, 0) for i in range(4)]
    notes = run + [(4 * RES, 1, 0), (4 * RES, 2, 0)]
    out = pickup_roots(notes, RES)
    assert lanes_at(out, 4 * RES - eighth) == [4]


# ---------------------------------------------------------------- riffs

def _profiles(kinds):
    """Synthetic bar profiles: same letter = identical audio."""
    out = []
    rng = {}
    for k in kinds:
        if k is None:
            out.append(None)
            continue
        if k not in rng:
            c = np.zeros(12); c[len(rng) % 12] = 1.0
            o = np.zeros(16); o[(2 * len(rng)) % 16] = 1.0
            rng[k] = (c, o)
        out.append(rng[k])
    return out


def test_riff_unify_stamps_consensus():
    bar = 4 * RES
    beat = RES
    canonical = [(0, 0, 0), (beat, 1, 0), (2 * beat, 0, 0), (3 * beat, 1, 0)]
    notes = []
    for b in range(4):  # four bars of the same riff, one with jitter
        for t, lane, s in canonical:
            jitter = 1 if (b == 2 and t == beat) else 0
            notes.append((b * bar + t, lane + jitter, s))
    out, stamped = unify_riff_bars(notes, _profiles("AAAA"), RES)
    assert stamped >= 3, stamped
    fps = [tuple(sorted((t % bar, l) for t, l, _ in out if t // bar == b))
           for b in range(4)]
    assert len(set(fps)) == 1, "all four bars must be identical after stamping"
    invariants(out)


def test_riff_unify_needs_a_real_cluster():
    bar = 4 * RES
    notes = [(b * bar, b % 5, 0) for b in range(4)]
    # all-different audio: nothing to unify
    out, stamped = unify_riff_bars(notes, _profiles("ABCD"), RES)
    assert stamped == 0 and out == sorted(notes)
    # a pair is not a riff
    out, stamped = unify_riff_bars(notes, _profiles("AABB"), RES)
    assert stamped == 0


def test_riff_unify_spares_outliers_and_empty_bars():
    bar = 4 * RES
    beat = RES
    notes = []
    for b in range(3):
        notes += [(b * bar, 0, 0), (b * bar + beat, 1, 0),
                  (b * bar + 2 * beat, 0, 0)]
    # bar 3 sounds the same but is charted completely differently (a fill)
    notes += [(3 * bar + i * (RES // 4), 4, 0) for i in range(6)]
    out, stamped = unify_riff_bars(notes, _profiles("AAAA"), RES)
    fill = [(t, l) for t, l, _ in out if t // bar == 3]
    assert fill == [(3 * bar + i * (RES // 4), 4) for i in range(6)], \
        "an outlier bar is a variation and must survive"
    # bar 4 with no notes stays empty even if audio matches
    out2, _ = unify_riff_bars(notes, _profiles("AAAAA"), RES)
    assert not any(t // bar == 4 for t, _, _ in out2)


def test_riff_unify_clamps_sustains_at_the_boundary():
    bar = 4 * RES
    beat = RES
    riff = [(0, 0, 3 * beat), (3 * beat, 1, 0)]
    notes = []
    for b in range(3):
        notes += [(b * bar + t, l, s) for t, l, s in riff]
    # the note right after bar 2's copy sits early in bar 3
    notes += [(3 * bar + RES // 4, 2, 0)]
    out, stamped = unify_riff_bars(notes, _profiles("AAAB"), RES)
    invariants(out)


# ---------------------------------------------------------------- swing

def steady(bpm=120.0, n=200, first=0.5):
    return TempoMap(beat_times=first + np.arange(n) * (60.0 / bpm),
                    pickup_beats=1)


def test_swing_detected_on_triplet_onsets():
    t = steady()
    events = []
    for b in range(2, 60):
        for frac in (0.0, 1 / 3, 2 / 3):
            events.append((t.beat_to_time(b + frac), 0.0, 60, 0.6))
    swing = swung_beats(events, t)
    assert len(swing) >= 40, len(swing)


def test_swing_not_detected_on_straight_or_sparse():
    t = steady()
    straight = [(t.beat_to_time(b + f), 0.0, 60, 0.6)
                for b in range(2, 60) for f in (0.0, 0.25, 0.5, 0.75)]
    assert swung_beats(straight, t) == set()
    sparse = [(t.beat_to_time(b + 1 / 3), 0.0, 60, 0.6) for b in range(2, 8)]
    assert swung_beats(sparse, t) == set()


def test_swing_beats_quantize_to_triplets():
    t = steady()
    events = []
    for b in range(2, 40):
        for frac in (0.0, 1 / 3, 2 / 3):
            events.append((t.beat_to_time(b + frac),
                           t.beat_to_time(b + frac) + 0.05, 60 + b % 12, 0.6))
    swing = swung_beats(events, t)
    notes = expert_from_notes(events, t, swing_beats=swing, ornaments=False)
    offs = {tick % RES for tick, _, _ in notes}
    assert offs <= {0, RES // 3, 2 * RES // 3}, offs


# ---------------------------------------------------------------- ornaments

def test_ornament_recovered_only_with_repeat_evidence():
    t = steady(bpm=120.0)
    bar_beats = 4
    events = []
    # main notes on beats, plus a strong 32nd flam before beat 2 of bars 1-3
    for b in range(1, 13):
        events.append((t.beat_to_time(b), t.beat_to_time(b) + 0.1, 60, 0.6))
    # a grace note is quieter than its main note (here: under the chord
    # amplitude ratio, so it merges away at the 16th grid) but still well
    # above the ornament evidence floor
    for barn in (1, 2, 3):
        beat = 1 + barn * bar_beats
        events.append((t.beat_to_time(beat - 1 / 8),
                       t.beat_to_time(beat - 1 / 8) + 0.05, 64, 0.35))
    notes = expert_from_notes(events, t, ornaments=True)
    orn = [tick for tick, _, _ in notes if tick % (RES // 8) == 0
           and tick % (RES // 4) != 0]
    assert orn, "recurring, loud, well-placed flams should be recovered"
    # without repetition: nothing
    events2 = [e for e in events][:13]
    events2.append((t.beat_to_time(5 - 1 / 8), t.beat_to_time(5 - 1 / 8) + 0.05,
                    64, 0.35))
    notes2 = expert_from_notes(events2, t, ornaments=True)
    orn2 = [tick for tick, _, _ in notes2 if tick % (RES // 8) == 0
            and tick % (RES // 4) != 0]
    assert not orn2, "a one-off flam is jitter, not an ornament"


def test_ornament_quiet_onsets_ignored():
    t = steady(bpm=120.0)
    events = []
    for b in range(1, 13):
        events.append((t.beat_to_time(b), t.beat_to_time(b) + 0.1, 60, 0.6))
    for barn in (1, 2, 3):
        beat = 1 + barn * 4
        events.append((t.beat_to_time(beat - 1 / 8),
                       t.beat_to_time(beat - 1 / 8) + 0.05, 64, 0.22))
    notes = expert_from_notes(events, t, ornaments=True)
    orn = [tick for tick, _, _ in notes if tick % (RES // 8) == 0
           and tick % (RES // 4) != 0]
    assert not orn, "a quiet 32nd is a ghost, not evidence"


# ---------------------------------------------------------------- ladders

def _ladder_setup(t, host_ring_beats):
    """Host G sustained at beat 4, R joins a 16th later, next G at beat 8.
    The transcribed event for the host rings for host_ring_beats."""
    beat = RES
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 1, beat // 2),
             (6 * beat, 2, 0), (8 * beat, 0, 0)]
    host_t = t.beat_to_time(4)
    events = [(host_t, t.beat_to_time(4 + host_ring_beats), 50, 0.6),
              (t.beat_to_time(4.25), t.beat_to_time(5), 55, 0.6)]
    return notes, events


def test_ladder_extends_evidenced_hold_through_join():
    t = steady()
    notes, events = _ladder_setup(t, host_ring_beats=1.5)
    out, made = extend_ladders(notes, events, t)
    assert made == 1
    host = sus_at(out, 4 * RES)
    join_tick = 4 * RES + RES // 4
    assert host > join_tick - 4 * RES, "host must ring past the join"
    assert sus_at(out, join_tick) == RES // 2, "the join keeps its own hold"
    # tail stops before the next note on the host's own lane at beat 8
    assert 4 * RES + host < 8 * RES, out


def test_ladder_needs_duration_evidence_and_a_higher_join():
    t = steady()
    # host sound stops before the join: no licence to overlap
    notes, events = _ladder_setup(t, host_ring_beats=0.2)
    out, made = extend_ladders(notes, events, t)
    assert made == 0 and out == sorted(notes)
    # join BELOW the held lane: not a build-up
    beat = RES
    notes = [(4 * beat, 2, beat // 2), (4 * beat + beat // 4, 0, 0),
             (8 * beat, 2, 0)]
    events = [(t.beat_to_time(4), t.beat_to_time(6), 60, 0.6)]
    out, made = extend_ladders(notes, events, t)
    assert made == 0


def test_ladder_keeps_gyb_but_pulls_a_leap_inward():
    t = steady()
    beat = RES
    host_ev = [(t.beat_to_time(4), t.beat_to_time(5.5), 50, 0.6)]
    # G host, Y then B joins: every step <=2, the legitimate span-3 stack
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 2, 0),
             (4 * beat + beat // 2, 3, 0), (8 * beat, 0, 0)]
    out, made = extend_ladders(notes, host_ev, t)
    assert made == 1
    assert lanes_at(out, 4 * beat + beat // 4) == [2], "G-Y-B must survive"
    assert lanes_at(out, 4 * beat + beat // 2) == [3]
    # G host, isolated B join (+3 leap): pulled inward to Y
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 3, 0),
             (8 * beat, 0, 0)]
    out, made = extend_ladders(notes, host_ev, t)
    assert made == 1
    assert lanes_at(out, 4 * beat + beat // 4) == [2], out
    # same leap but the join is a chord: the ladder is vetoed, lanes stay
    notes = [(4 * beat, 0, beat // 2), (4 * beat + beat // 4, 3, 0),
             (4 * beat + beat // 4, 4, 0), (8 * beat, 0, 0)]
    out, made = extend_ladders(notes, host_ev, t)
    assert made == 0 and out == sorted(notes)


def test_ladder_respects_budget_and_spacing():
    t = steady()
    beat = RES
    notes = []
    events = []
    for k in range(6):  # six candidate ladders two beats apart
        base = (4 + 2 * k) * beat
        notes += [(base, 0, beat // 2), (base + beat // 4, 1, 0)]
        events.append((t.beat_to_time(base / RES),
                       t.beat_to_time(base / RES + 1.5), 50, 0.6))
    out, made = extend_ladders(notes, events, t)
    # 4-beat minimum spacing: of six candidates 2 beats apart, every
    # other one ladders
    assert made == 3, made


# ---------------------------------------------------------------- writing

def test_forced_flags_written_expert_only():
    t = steady()
    tiers = {
        "ExpertSingle": [(0, 0, 0), (RES, 1, 0)],
        "HardSingle": [(0, 0, 0), (RES, 1, 0)],
    }
    text = write_chart(tiers, t, {"name": "x", "artist": "y", "charter": "z"},
                       "song.opus", forced={RES})
    expert = text.split("[ExpertSingle]")[1].split("[HardSingle]")[0]
    hard = text.split("[HardSingle]")[1]
    assert f"{RES} = N 5 0" in expert
    assert "N 5" not in hard, "forcing is spacing-relative: Expert only"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
