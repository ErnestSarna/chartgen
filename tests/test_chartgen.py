"""Self-check for the parts that would silently produce bad charts.

Run: python -m tests.test_chartgen
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.chart import notes_from_tokens, write_chart  # noqa: E402
from chartgen.expression import (  # noqa: E402
    add_sustains,
    add_sustains_all_tiers,
    hopo_threshold,
    star_power_phrases,
)
from chartgen.pitch import frets_from_pitch  # noqa: E402
from chartgen.quality import step_profile, variety_score, walk_score  # noqa: E402
from chartgen.tempo import RESOLUTION, TempoMap  # noqa: E402


def steady(bpm=120.0, n=64, first=0.5):
    """Perfectly steady beat grid, the case we can reason about exactly."""
    return TempoMap(
        beat_times=first + np.arange(n) * (60.0 / bpm),
        pickup_beats=max(1, round(first / (60.0 / bpm))),
    )


def test_quantize_lands_on_grid():
    t = steady()
    beat = 60.0 / 120.0
    # A note 12ms late off beat 4 must snap to exactly beat 4.
    tick = t.quantize(t.beat_times[4] + 0.012, subdiv=4)
    assert tick % RESOLUTION == 0, f"expected an integer beat, got {tick / RESOLUTION}"
    assert tick == (4 + t.pickup_beats) * RESOLUTION, tick

    # Anything on the grid must stay on the grid: this is the property that was
    # broken before (2 of 180 notes on a beat) and that difficulty reduction needs.
    step = RESOLUTION // 4
    for i in range(len(t.beat_times) - 1):
        for frac in (0.0, 0.25, 0.5, 0.75):
            noisy = t.beat_times[i] + frac * beat + 0.008
            assert t.quantize(noisy, subdiv=4) % step == 0


def test_sync_track_collapses_steady_tempo():
    # Pickup is exactly one beat, so it needs no separate tempo event.
    events = steady(bpm=120.0, first=0.5).sync_track()
    assert events == [(0, 120000)], events

    # Pickup shorter than a beat needs its own event to stay in sync, but the
    # rest of the song still collapses to a single steady event.
    events = steady(bpm=120.0, first=0.3).sync_track()
    assert len(events) == 2, events
    assert events[0] == (0, 200000), events[0]
    assert events[1] == (RESOLUTION, 120000), events[1]


def test_sync_track_follows_drift():
    # Tempo ramps 120 -> 150; a single BPM event cannot represent this.
    times, t_cur = [], 0.0
    for i in range(48):
        times.append(t_cur)
        t_cur += 60.0 / (120.0 + i * 0.625)
    t = TempoMap(beat_times=np.array(times), pickup_beats=0)
    assert len(t.sync_track()) > 1, "drift should produce multiple tempo events"


def test_sync_track_absorbs_frame_jitter():
    """librosa snaps beats to ~23ms frames; that jitter is not tempo drift.

    This is the regression that read a 120.0 BPM click track as 117.5 BPM with
    74 tempo events.
    """
    hop = 512 / 22050.0  # librosa's default analysis frame, ~23.2ms
    true_times = np.arange(80) * 0.5  # exactly 120 BPM
    detected = np.round(true_times / hop) * hop

    t = TempoMap(beat_times=detected, pickup_beats=0)
    assert abs(t.bpm - 120.0) < 0.5, f"fitted tempo {t.bpm:.2f} should be ~120"

    events = t.sync_track()
    assert len(events) <= 2, f"jitter must not spawn tempo events: {len(events)}"

    # And the emitted map must track the TRUE grid, not the jittered detections.
    worst = max(abs(_replay(events, i) - true_times[i]) for i in range(len(true_times)))
    assert worst < 0.020, f"drifts {worst * 1000:.1f}ms from the true grid"


def _replay(events, beat_idx):
    """Playback time of a beat, as a game reading the SyncTrack would compute it."""
    tick = beat_idx * RESOLUTION
    played, prev_tick, prev_bpm = 0.0, 0, events[0][1] / 1000.0
    for ev_tick, ev_bpm in events[1:]:
        if ev_tick >= tick:
            break
        played += (ev_tick - prev_tick) / RESOLUTION * (60.0 / prev_bpm)
        prev_tick, prev_bpm = ev_tick, ev_bpm / 1000.0
    return played + (tick - prev_tick) / RESOLUTION * (60.0 / prev_bpm)


def test_sync_track_honours_drift_tolerance():
    """A wobbling tempo must be tracked to whatever tolerance we ask for."""
    times, t_cur = [], 0.0
    for i in range(64):
        times.append(t_cur)
        t_cur += 60.0 / (128.0 + 6.0 * np.sin(i / 5.0))
    t = TempoMap(beat_times=np.array(times), pickup_beats=0)

    tol = 0.005
    events = t.sync_track(max_drift_s=tol)
    worst = max(abs(_replay(events, i) - times[i]) for i in range(len(times)))
    assert worst <= tol + 0.001, f"tempo map desyncs by {worst * 1000:.1f}ms"


def test_notes_from_tokens_dedupes_and_orders():
    t = steady()
    reverse_map = {0: (0,), 1: (1,), 5: (0, 2)}
    # 40ms frames: two frames close enough to snap to the same 16th tick.
    tokens = [99] * 100
    tokens[25] = 0   # 1.00s
    tokens[26] = 0   # 1.04s -> same 16th as above at 120bpm (125ms grid)
    tokens[50] = 5   # 2.00s, a chord
    notes = notes_from_tokens(tokens, 40, t, reverse_map, subdiv=4)

    ticks = [n[0] for n in notes]
    assert ticks == sorted(ticks), "notes must be tick-ordered"
    assert len(set(ticks)) < len([25, 26, 50]) or ticks[0] != ticks[1]
    assert (min(ticks), 0, 0) in [tuple(n) for n in notes]
    chord = [n for n in notes if n[0] == max(ticks)]
    assert {c[1] for c in chord} == {0, 2}, f"chord lanes lost: {chord}"
    assert all(0 <= lane <= 7 for _, lane, _ in notes)


def test_notes_from_tokens_does_not_alias_frames_into_chords():
    """Four 40ms frames fit inside one 16th at 90 BPM; different-lane frames
    landing on one tick must not stack into a fake chord. Measured on a real
    song, unioning turned 38 model chords into 210 chart chords."""
    t = steady(bpm=90.0)
    reverse_map = {0: (0,), 1: (1,), 2: (2,), 5: (0, 2)}
    tokens = [99] * 120
    # Three different single-lane frames all inside the first 16th after beat 2.
    beat2 = t.beat_times[2]
    for offset, tok in ((0.000, 0), (0.040, 1), (0.080, 2)):
        tokens[int(round((beat2 + offset) / 0.040))] = tok
    notes = notes_from_tokens(tokens, 40, t, reverse_map, subdiv=4)
    by_tick = {}
    for tick, lane, _ in notes:
        by_tick.setdefault(tick, []).append(lane)
    assert all(len(v) == 1 for v in by_tick.values()), f"fake chord: {by_tick}"

    # A genuine chord — one multi-lane frame — must still come through whole.
    tokens = [99] * 120
    tokens[int(round(beat2 / 0.040))] = 5
    notes = notes_from_tokens(tokens, 40, t, reverse_map, subdiv=4)
    assert {l for _, l, _ in notes} == {0, 2}, notes


def test_write_chart_is_parseable_and_has_real_tempo():
    t = steady()
    text = write_chart(
        {"ExpertSingle": [(480, 0, 0), (600, 2, 0)]}, t,
        {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
    )
    assert "[ExpertSingle]" in text and "[SyncTrack]" in text
    assert f"Resolution = {RESOLUTION}" in text
    assert "B 200000" not in text, "must not fall back to audio2chart's fake 200 BPM"
    assert "B 120000" in text, "real detected tempo missing from SyncTrack"
    assert "480 = N 0 0" in text


def test_write_chart_emits_all_tiers_star_power_and_sections():
    t = steady()
    tiers = {
        "ExpertSingle": [(0, 0, 0), (480, 1, 0)],
        "HardSingle": [(0, 0, 0)],
        "MediumSingle": [(0, 0, 0)],
        "EasySingle": [(0, 0, 0)],
    }
    text = write_chart(
        tiers, t, {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
        star_power=[(0, 4 * RESOLUTION)], events=[(0, "Section 1")],
    )
    for tier in tiers:
        assert f"[{tier}]" in text, tier
        # Star power is per-track in .chart, so it must repeat in each block.
        block = text.split(f"[{tier}]")[1].split("}")[0]
        assert "= S 2 " in block, f"{tier} lost its star power phrase"
    assert '= E "section Section 1"' in text


def test_write_chart_merges_lyrics_into_events_in_tick_order():
    t = steady()
    text = write_chart(
        {"ExpertSingle": [(0, 0, 0)]}, t,
        {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
        events=[(960, "Section 1")],
        lyrics=[(480, "phrase_start"), (480, "lyric Hello"),
                (720, "lyric world"), (900, "phrase_end")],
    )
    block = text.split("[Events]")[1].split("}")[0]
    lines = [ln.strip() for ln in block.splitlines() if "= E" in ln]
    assert any('lyric Hello' in ln for ln in lines)
    assert any('section Section 1' in ln for ln in lines)
    ticks = [int(ln.split(" = ")[0]) for ln in lines]
    assert ticks == sorted(ticks), "events must stay tick-ordered"
    # A phrase sharing a tick with its own first syllable must still open
    # first, or that word belongs to the previous phrase.
    same_tick = [ln for ln in lines if ln.startswith("480 = ")]
    assert "phrase_start" in same_tick[0], same_tick


def test_empty_star_power_phrases_are_dropped():
    """A phrase whose notes were all reduced away is unactivatable dead weight."""
    text = write_chart(
        {"ExpertSingle": [(0, 0, 0)], "EasySingle": [(0, 0, 0)]},
        steady(), {"name": "T", "artist": "A", "charter": "c"}, "song.ogg",
        # Second phrase sits far past every note.
        star_power=[(0, RESOLUTION), (100 * RESOLUTION, RESOLUTION)],
    )
    assert text.count("= S 2 ") == 2, "one phrase per tier should survive"
    assert f"{100 * RESOLUTION} = S 2" not in text


def test_sustains_respect_note_spacing():
    res = RESOLUTION
    # 8th-note run, then a note with two beats of space, then a final note.
    notes = [(0, 0, 0), (res // 2, 1, 0), (res, 2, 0), (3 * res, 3, 0)]
    out = add_sustains(notes, res, end_tick=8 * res)
    sus = {tick: length for tick, _, length in out}

    assert sus[0] == 0, "8th-note gap must not sustain"
    assert sus[res // 2] == 0, "8th-note gap must not sustain"
    # Two beats of room: sustain, but released before the next note.
    assert sus[res] > 0
    assert res + sus[res] < 3 * res, "sustain must end before the next note"
    assert sus[3 * res] > 0, "final note should ring out to the end"


def test_sustains_are_capped_and_never_negative():
    res = RESOLUTION
    out = add_sustains([(0, 0, 0)], res, end_tick=500 * res, max_beats=4.0)
    assert 0 < out[0][2] <= 4 * res, out
    # A note past the supplied end tick must not produce a negative sustain.
    out = add_sustains([(10 * res, 0, 0)], res, end_tick=res)
    assert out[0][2] >= 0


def test_lower_tiers_inherit_expert_sustains_not_their_own_gaps():
    """A sparse tier must not sustain just because reduction left it a gap."""
    res = RESOLUTION
    # A busy 16th-note run: the music has no room to hold anything.
    expert = [(i * (res // 4), i % 5, 0) for i in range(16)]
    # Reduction keeps only the downbeats, leaving one-beat gaps behind.
    easy = [n for n in expert if n[0] % res == 0]

    out = add_sustains_all_tiers(
        {"ExpertSingle": expert, "EasySingle": easy}, res, end_tick=16 * res
    )
    easy_sus = [s for _, _, s in out["EasySingle"][:-1]]
    assert all(s == 0 for s in easy_sus), f"busy song over-sustained in Easy: {easy_sus}"

    # Computing from Easy's own spacing is what got this wrong; prove it differs.
    naive = add_sustains(easy, res, end_tick=16 * res)
    assert any(s > 0 for _, _, s in naive[:-1]), "test no longer exercises the bug"


def test_inherited_sustains_never_reach_the_next_note():
    res = RESOLUTION
    expert = [(0, 0, 0), (4 * res, 1, 0), (5 * res, 2, 0)]
    easy = [(0, 0, 0), (5 * res, 2, 0)]  # middle note dropped
    out = add_sustains_all_tiers(
        {"ExpertSingle": expert, "EasySingle": easy}, res, end_tick=9 * res
    )
    for notes in out.values():
        for (tick, _, sus), (nxt, _, _) in zip(notes, notes[1:]):
            if nxt != tick:
                assert tick + sus < nxt, f"sustain at {tick} runs into {nxt}"


def test_star_power_is_bar_aligned_spaced_and_non_empty():
    res = RESOLUTION
    bar = 4 * res
    # Dense 8th notes across 32 bars.
    notes = [(i * (res // 2), i % 5, 0) for i in range(32 * 8)]
    phrases = star_power_phrases(notes, res, duration_s=120.0)

    assert phrases, "a dense 32-bar song should get star power"
    ticks = {t for t, _, _ in notes}
    for start, length in phrases:
        assert start % bar == 0, f"phrase at {start} is not bar-aligned"
        assert any(start <= t < start + length for t in ticks), "empty phrase"
    starts = [s for s, _ in phrases]
    assert starts == sorted(starts)
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 6 * bar for g in gaps), f"phrases too bunched: {gaps}"


def test_star_power_handles_sparse_and_empty_input():
    assert star_power_phrases([], RESOLUTION, 120.0) == []
    # Too few notes to be worth an activation.
    assert star_power_phrases([(0, 0, 0)], RESOLUTION, 120.0) == []


def _notes(lane_counts):
    """Build a note list with a given lane distribution."""
    out, tick = [], 0
    for lane, count in lane_counts.items():
        for _ in range(count):
            out.append((tick, lane, 0))
            tick += RESOLUTION // 4
    return out


def test_variety_score_separates_real_degenerate_charts():
    """Thresholded against distributions actually produced by the model.

    All four came off the same clip at identical settings, which is why the gate
    exists at all.
    """
    balanced = variety_score(_notes({0: 39, 1: 40, 2: 25, 3: 15}))
    spread = variety_score(_notes({0: 2, 1: 12, 2: 40, 3: 38, 4: 27}))
    one_lane = variety_score(_notes({0: 1, 1: 100, 2: 13, 3: 5}))
    open_heavy = variety_score(_notes({0: 10, 1: 39, 2: 30, 7: 40}))

    assert balanced > 0.80, balanced
    assert spread > 0.80, spread
    assert one_lane < 0.40, f"84%-on-one-lane must fail the gate, got {one_lane}"
    assert open_heavy < 0.40, f"34% open notes must fail the gate, got {open_heavy}"
    assert spread > one_lane and balanced > open_heavy


def test_variety_score_flags_two_fret_alternation():
    """Ping-ponging two frets is boring even though no single lane dominates."""
    assert variety_score(_notes({0: 60, 1: 60})) < 0.80


def test_variety_score_tolerates_realistic_low_fret_bias():
    """Real charts favour low frets; that must not trigger pointless retries."""
    assert variety_score(_notes({0: 40, 1: 30, 2: 20, 3: 8, 4: 2})) >= 0.80


def test_variety_score_handles_degenerate_input():
    assert variety_score([]) == 0.0
    assert variety_score(_notes({7: 20})) == 0.0  # open notes only, no frets


def _seq(lanes):
    return [(i * (RESOLUTION // 4), lane, 0) for i, lane in enumerate(lanes)]


def test_walk_score_catches_the_staircase_variety_missed():
    """The failure that got through playtesting: a chart nobody enjoyed.

    A G-R-Y-B-O staircase uses all five frets perfectly evenly, so entropy rates
    it near-perfect. Only a transition-based measure can see it.
    """
    staircase = _seq([0, 1, 2, 3, 4] * 20)
    assert variety_score(staircase) > 0.95, "entropy really is blind to this"
    assert walk_score(staircase) < 0.30, walk_score(staircase)


def test_walk_score_on_the_real_measured_chart():
    """The shipped chart's signature — heavy walking with almost no repeats —
    must fail the 0.80 gate. Built deterministically: a fretboard zigzag with
    one repeat per 16 notes (~6%, what the playtested chart measured)."""
    lanes, cur, direction = [], 0, 1
    for i in range(400):
        if i % 16 == 15:
            lanes.append(cur)  # the rare repeat
            continue
        cur += direction
        if cur in (0, 4):
            direction *= -1
        lanes.append(cur)
    chart = _seq(lanes)
    p = step_profile(chart)
    assert p["adjacent"] > 0.70 and p["repeat"] < 0.10, p
    assert walk_score(chart) < 0.80, f"should be penalised, got {walk_score(chart)}"
    assert walk_score(chart) < walk_score(_seq([0, 0, 3, 1, 1, 4, 2, 2] * 50))


def test_walk_score_accepts_a_riff_like_pattern():
    """Repeated frets and jumps — what real guitar parts do — must pass."""
    riff = _seq([0, 0, 3, 0, 0, 2, 4, 4, 1, 1, 0, 0, 2, 2, 4, 0] * 12)
    p = step_profile(riff)
    assert p["repeat"] > 0.12, p
    assert walk_score(riff) > 0.80, f"riff wrongly penalised: {walk_score(riff)}"


def test_step_profile_ignores_chords_and_opens():
    notes = [(0, 0, 0), (0, 2, 0), (120, 7, 0), (240, 1, 0), (360, 1, 0)]
    p = step_profile(notes)
    # Only the two single non-open notes form one step, and it is a repeat.
    assert p["steps"] == 1 and p["repeat"] == 1.0, p
    assert p["chord_share"] > 0


def test_rating_scales_with_density_and_stays_in_range():
    from chartgen.rating import rate_expert

    t = steady(n=256)
    res = RESOLUTION
    sparse = [(i * 2 * res, i % 3, 0) for i in range(60)]       # half notes
    busy = [(i * (res // 4), i % 5, 0) for i in range(600)]      # wall of 16ths
    lo, hi = rate_expert(sparse, t), rate_expert(busy, t)
    assert 0 <= lo <= 6 and 0 <= hi <= 6
    assert hi > lo, f"denser chart must rate higher: {lo} vs {hi}"
    assert rate_expert([(0, 0, 0)], t) == 0 or rate_expert([(0, 0, 0)], t) <= 6


def test_thin_to_rating_hits_the_target_and_keeps_beats():
    from chartgen.rating import rate_expert
    from chartgen.reduce import thin_to_rating

    t = steady(n=512)
    res = RESOLUTION
    # A wall of 16ths: rates high naturally.
    expert = [(i * (res // 4), (i * 3) % 5, 0) for i in range(800)]
    natural = rate_expert(expert, t)
    assert natural >= 4, f"fixture should rate hard, got {natural}"

    target = 2
    thinned, achieved = thin_to_rating(expert, res, t, target)
    assert achieved <= target, (natural, achieved)
    assert rate_expert(thinned, t) == achieved, "reported rating must be real"
    assert len(thinned) < len(expert)
    kept = {tick for tick, _, _ in thinned}
    beats = {tick for tick, _, _ in expert if tick % res == 0}
    assert beats <= kept, "thinning must never drop beats"

    # Already at or below target: untouched.
    sparse = [(i * 2 * res, i % 3, 0) for i in range(60)]
    out, achieved = thin_to_rating(sparse, res, t, 6)
    assert out == sparse


def test_song_ini_carries_rating():
    from chartgen.chart import write_song_ini

    meta = {"name": "n", "artist": "a", "charter": "c"}
    assert "diff_guitar = 4" in write_song_ini(meta, 1000, diff_guitar=4)
    assert "diff_guitar = -1" in write_song_ini(meta, 1000)


def test_song_ini_disables_hopos_by_default():
    from chartgen.chart import write_song_ini

    meta = {"name": "n", "artist": "a", "charter": "c"}
    assert "hopo_frequency = 1" in write_song_ini(meta, 1000)
    assert "hopo_frequency" not in write_song_ini(meta, 1000, hopos=True)


def test_frets_from_pitch_keeps_lowest_pitch_notes():
    """Regression: silence and lowest-pitch must not share an encoding.

    f0_norm originally used 0.0 for both "nothing sounding" and "bin 0", and bin
    0 is the open low E — the most-played region on the instrument. Every low
    note got filtered out as silence, collapsing the whole song onto one fret.
    """
    f0 = np.array([0.0, 0.8, 0.0, 0.8, 0.0, 0.8])  # alternating low / high
    voiced = np.ones(6, dtype=bool)
    frets = frets_from_pitch(f0, voiced, list(range(6)))
    assert len(set(frets)) > 1, f"low notes were dropped again: {frets}"
    assert frets[0] < frets[1], frets


def test_frets_from_pitch_maps_equal_pitch_to_equal_fret():
    """The property the generated charts lacked: repeats produce repeats."""
    f0 = np.array([0.2, 0.9, 0.2, 0.5, 0.2, 0.9])
    voiced = np.ones(6, dtype=bool)
    frets = frets_from_pitch(f0, voiced, list(range(6)))
    assert frets[0] == frets[2] == frets[4], frets
    assert frets[1] == frets[5], frets


def test_frets_from_pitch_uses_the_whole_neck():
    rng = np.random.default_rng(3)
    f0 = rng.random(400)
    frets = frets_from_pitch(f0, np.ones(400, dtype=bool), list(range(400)))
    assert set(frets) == {0, 1, 2, 3, 4}, sorted(set(frets))


def test_frets_from_pitch_degenerate_inputs():
    # Nothing voiced at all.
    assert set(frets_from_pitch(np.array([0.5, 0.5]), np.zeros(2, dtype=bool), [0, 1])) == {0}
    # One sustained pitch: no range to spread across.
    assert set(frets_from_pitch(np.full(5, 0.4), np.ones(5, dtype=bool), list(range(5)))) == {0}
    # Tick past the end of the feature array must clamp, not crash.
    frets_from_pitch(np.array([0.1, 0.9]), np.ones(2, dtype=bool), [0, 1, 99])


def _natural_hopos(notes, thr=162):
    by = {}
    for tick, lane, _ in notes:
        by.setdefault(tick, []).append(lane)
    ticks = sorted(by)
    return sum(
        1 for a, b in zip(ticks, ticks[1:])
        if len(by[b]) == 1 and by[b][0] != OPEN_LANE
        and b - a < thr and by[b] != by[a]
    )


OPEN_LANE = 7


def test_reassign_frets_makes_repeats_and_tracks_pitch():
    """The property the raw model lacks: same pitch -> same fret, and motion
    follows the music instead of walking the fretboard."""
    from chartgen.frets import reassign_frets

    t = steady(n=128)
    grid_ms = 40
    # Audio alternates a low phrase and a high phrase, 6 frames each.
    f0 = np.tile(np.repeat([0.15, 0.85], 8), 200)[:3000]
    voiced = np.ones(3000, dtype=bool)
    # Model output: a staircase — the exact failure the playtest found.
    notes = [(i * 120, i % 5, 0) for i in range(96)]
    out = reassign_frets(notes, t, f0, voiced, grid_ms)

    assert len(out) == len(notes)
    assert [n[0] for n in out] == [n[0] for n in notes], "timing must not move"
    from chartgen.quality import step_profile
    p = step_profile(out)
    assert p["repeat"] > 0.30, f"pitch plateaus must become fret repeats: {p}"


def test_reassign_frets_preserves_chords_opens_and_silence():
    from chartgen.frets import reassign_frets

    t = steady()
    f0 = np.full(2000, 0.5, dtype=np.float32)
    f0[:100] = 0.1
    voiced = np.ones(2000, dtype=bool)
    voiced[50:60] = False  # a silent stretch

    res = RESOLUTION
    notes = [
        (0, 1, 0), (0, 3, 0),        # chord, shape span 2
        (res, OPEN_LANE, 0),          # open note
        (2 * res, 4, 90),             # single with a sustain
    ]
    out = reassign_frets(notes, t, f0, voiced, 40)
    chord = sorted(l for tick, l, _ in out if tick == 0)
    assert chord[1] - chord[0] == 2, f"chord shape lost: {chord}"
    assert (res, OPEN_LANE, 0) in out, "open notes must stay open"
    sus = [s for tick, _, s in out if tick == 2 * res]
    assert sus == [90], "sustains must survive reassignment"

    # A note in the silent stretch keeps the model's lane: pitch is meaningless.
    silent_tick = t.quantize(50 * 0.040)
    silent_notes = [(silent_tick, 3, 0)]
    kept = reassign_frets(silent_notes, t, f0, voiced, 40)
    assert kept == silent_notes


def test_reduce_hard_keeps_hopos_and_thins_walls():
    """The old reducer's Hard was all strums: 3 HOPOs where Expert had 650."""
    from chartgen.reduce import reduce_hard

    res = RESOLUTION
    # 4 bars of saturated 16ths — a wall — then a short 3-note lick.
    wall = [(i * (res // 4), i % 5, 0) for i in range(64)]
    lick_start = 32 * res
    lick = [(lick_start + i * (res // 4), i, 0) for i in range(3)]
    hard = reduce_hard(wall + lick, res)

    assert len(hard) < len(wall) + len(lick), "walls must thin"
    assert _natural_hopos(hard) >= 16, f"Hard lost its HOPOs: {_natural_hopos(hard)}"
    kept_lick = [n for n in hard if n[0] >= lick_start]
    assert len(kept_lick) == 3, "short figures must survive whole"


def test_reduce_tiers_ladder_and_lane_ranges():
    from chartgen.reduce import derive_tiers

    res = RESOLUTION
    expert = [(i * (res // 4), (i * 3) % 5, 0) for i in range(128)]
    # Sprinkle chords so the caps are exercised.
    expert += [(i * res * 2, ((i * 3) % 4) + 1, 0) for i in range(16)]
    tiers = derive_tiers(sorted(expert), res)

    n_e = len({t for t, _, _ in expert})
    n_h = len({t for t, _, _ in tiers["HardSingle"]})
    n_m = len({t for t, _, _ in tiers["MediumSingle"]})
    n_y = len({t for t, _, _ in tiers["EasySingle"]})
    assert n_e > n_h > n_m > n_y, (n_e, n_h, n_m, n_y)

    assert max(l for _, l, _ in tiers["MediumSingle"]) <= 3, "no orange on Medium"
    assert max(l for _, l, _ in tiers["EasySingle"]) <= 2, "Easy is G/R/Y"
    from chartgen.quality import variety_score
    assert variety_score(tiers["MediumSingle"]) > 0.60, "remap must not collapse lanes"

    # Chord caps: Hard <= 2 notes per tick, Easy strictly single.
    for name, cap in (("HardSingle", 2), ("MediumSingle", 2), ("EasySingle", 1)):
        by_tick = {}
        for t, l, _ in tiers[name]:
            by_tick.setdefault(t, []).append(l)
        assert max(map(len, by_tick.values())) <= cap, name


def test_reduce_hard_respects_nps_cap_on_dense_charts():
    """Playtest: Hard kept 92% of a dense EDM Expert. With a BPM the reducer
    must enforce ~4.5 notes/sec, dropping fine positions before off-8ths and
    never touching beats."""
    from chartgen.reduce import HARD_MAX_NPS, reduce_hard

    res = RESOLUTION
    bpm = 128.0
    # Saturated 16ths for 16 bars: 16 positions per bar.
    expert = [(i * (res // 4), i % 5, 0) for i in range(16 * 16)]
    hard = reduce_hard(expert, res, bpm=bpm)

    seconds = 16 * 4 * 60.0 / bpm
    nps = len({t for t, _, _ in hard}) / seconds
    assert nps <= HARD_MAX_NPS + 0.3, f"Hard too dense: {nps:.1f} nps"
    kept = {t for t, _, _ in hard}
    beats = {t for t, _, _ in expert if t % res == 0}
    assert beats <= kept, "beats must never be dropped"
    # Without a BPM the cap is off and the old behaviour stands.
    assert len(reduce_hard(expert, res)) > len(hard)


def test_reassign_frets_limits_fast_chord_jumps():
    """Playtest: opposite-side two-note chords in quick succession are a hand
    shift no real chart asks for. Within a beat, the anchor moves at most one
    lane; with time to move, it may jump freely."""
    from chartgen.frets import reassign_frets

    t = steady(n=64)
    res = RESOLUTION
    # Pitch alternates extremes in 8-frame blocks (0.32s) — long enough to
    # survive median smoothing, short enough that consecutive 8th-note chords
    # see opposite extremes and raw pitch frets would slam between 0 and 4.
    f0 = np.tile(np.repeat([0.05, 0.95], 8), 200).astype(np.float32)[:3000]
    voiced = np.ones(3000, dtype=bool)
    # Two-note chords every 8th: fast succession.
    notes = []
    for i in range(16):
        tick = i * (res // 2)
        notes += [(tick, 0, 0), (tick, 1, 0)]
    out = reassign_frets(notes, t, f0, voiced, 40)
    by_tick = {}
    for tick, lane, _ in out:
        by_tick.setdefault(tick, []).append(lane)
    anchors = [min(v) for _, v in sorted(by_tick.items())]
    jumps = [abs(b - a) for a, b in zip(anchors, anchors[1:])]
    assert max(jumps) <= 1, f"fast chords still teleport: {anchors}"

    # Chords a full bar apart may jump the neck.
    slow = []
    for i in range(8):
        tick = i * 4 * res
        slow += [(tick, 0, 0), (tick, 1, 0)]
    out = reassign_frets(slow, t, f0, voiced, 40)
    by_tick = {}
    for tick, lane, _ in out:
        by_tick.setdefault(tick, []).append(lane)
    anchors = [min(v) for _, v in sorted(by_tick.items())]
    assert max(abs(b - a) for a, b in zip(anchors, anchors[1:])) >= 2, \
        "slow chords should still follow big pitch moves"


def test_chord_rules_remove_green_orange_stretches():
    """Chorus Encore errors on a Green+Orange chord on Hard; RBN keeps
    green-to-orange three-note chords off Expert too."""
    from chartgen.reduce import enforce_chord_rules

    tiers = enforce_chord_rules({
        "ExpertSingle": [(0, 0, 0), (0, 2, 0), (0, 4, 90)],   # 3-note G..O
        "HardSingle": [(0, 0, 0), (0, 4, 0)],                  # 2-note G+O
        "MediumSingle": [(0, 0, 0), (0, 3, 0)],                # G+B
        "EasySingle": [(0, 0, 0)],
    })
    for name, notes in tiers.items():
        lanes = {lane for _, lane, _ in notes}
        assert not (0 in lanes and 4 in lanes), f"{name} kept a G+O stretch"
    assert {l for _, l, _ in tiers["MediumSingle"]} == {0, 2}, tiers["MediumSingle"]
    # The moved note keeps its sustain rather than silently losing it.
    assert (0, 3, 90) in tiers["ExpertSingle"], tiers["ExpertSingle"]
    # A two-note G+O on EXPERT is allowed (sparingly) and must survive.
    kept = enforce_chord_rules({"ExpertSingle": [(0, 0, 0), (0, 4, 0)]})
    assert {l for _, l, _ in kept["ExpertSingle"]} == {0, 4}


def test_star_power_avoids_the_unspendable_ending():
    """CH needs two phrases to activate, so meter awarded in the last measures
    can never be spent. RBN: no phrase in roughly the last 8 measures."""
    res = RESOLUTION
    bar = 4 * res
    notes = [(i * (res // 2), i % 5, 0) for i in range(32 * 8)]
    phrases = star_power_phrases(notes, res, duration_s=120.0)
    last_note = max(t for t, _, _ in notes)
    for start, length in phrases:
        assert start <= last_note - 8 * bar, f"phrase at {start} is too late"
        # Two bars: the measured human median is 8 beats, twice the RBN
        # spec's one measure.
        assert length == 2 * bar, "phrases should be two measures"


def test_reduce_keeps_syncopation():
    """An isolated off-beat note is the rhythm; deleting it kills the song."""
    from chartgen.reduce import reduce_hard

    res = RESOLUTION
    # Sparse syncopated figure: beat, then a lone 16th pickup before beat 3.
    notes = [(0, 0, 0), (2 * res + 3 * res // 4, 2, 0), (3 * res, 1, 0)]
    hard = reduce_hard(notes, res)
    assert (2 * res + 3 * res // 4, 2, 0) in hard, "syncopation must survive on Hard"


def test_density_gate_drops_weak_notes_but_never_tears_holes():
    """Notes without onset evidence go; gaps larger than 2 beats never open."""
    from chartgen.density import gate_by_onsets

    sr = 22050
    t = steady(bpm=120.0, n=32)
    res = RESOLUTION
    # Audio: clicks on beats 1-8, then silence for beats 9-16.
    y = np.zeros(sr * 10, dtype=np.float32)
    for beat in range(8):
        s = int(t.beat_to_time(beat + 1) * sr)
        y[s:s + 200] = 1.0
    # Chart: a note on every 8th across beats 1-16.
    notes = [(res + i * (res // 2), i % 5, 0) for i in range(30)]

    out = gate_by_onsets(notes, y, sr, t)
    kept = sorted({tick for tick, _, _ in out})
    assert len(kept) < 30, "silent-section notes must thin out"
    # Louder half survives more densely than the silent half.
    mid = res + 15 * (res // 2)
    loud = sum(1 for k in kept if k < mid)
    quiet = sum(1 for k in kept if k >= mid)
    assert loud > quiet, (loud, quiet)
    # The gap constraint held everywhere.
    for a, b in zip(kept, kept[1:]):
        assert b - a <= 2 * res + res // 2, f"gate tore a hole: {a}->{b}"


def test_density_gate_never_eats_the_fade_out_tail():
    """Playtest: charts ended early on fade-outs — edge notes had no 'after'
    neighbour so the gap rule never fired. Coverage may retreat at most the
    gap budget from either end."""
    from chartgen.density import gate_by_onsets

    sr = 22050
    t = steady(bpm=120.0, n=64)
    res = RESOLUTION
    y = np.zeros(sr * 20, dtype=np.float32)
    # Loud clicks for beats 1-16, then a fading tail: beats 17-28 get quieter
    # and quieter but the song is still going.
    for beat in range(16):
        s = int(t.beat_to_time(beat + 1) * sr)
        y[s:s + 300] = 1.0
    for i, beat in enumerate(range(16, 28)):
        s = int(t.beat_to_time(beat + 1) * sr)
        y[s:s + 300] = max(0.02, 0.25 * (0.7 ** i))
    notes = [(res * (b + 1), b % 5, 0) for b in range(28)]

    out = gate_by_onsets(notes, y, sr, t)
    last_kept = max(tick for tick, _, _ in out)
    last_orig = max(tick for tick, _, _ in notes)
    assert last_orig - last_kept <= 2 * res, \
        f"tail retreated {(last_orig - last_kept) / res:.1f} beats"


def test_tier_block_writes_solo_markers_only_where_notes_exist():
    from chartgen.chart import _tier_block

    notes = [(0, 0, 0), (480, 1, 0), (960, 2, 0)]
    block = _tier_block("ExpertSingle", notes, [], solos=[(0, 960), (5000, 6000)])
    assert "0 = E solo" in block and "960 = E soloend" in block
    assert "5000 = E solo" not in block, "empty solo phrase must be dropped"


def test_density_gate_keeps_everything_on_strong_audio():
    from chartgen.density import gate_by_onsets

    sr = 22050
    t = steady(bpm=120.0, n=32)
    res = RESOLUTION
    y = np.zeros(sr * 10, dtype=np.float32)
    notes = []
    for i in range(16):
        s = int(t.beat_to_time(i + 1) * sr)
        y[s:s + 300] = 1.0
        notes.append((res * (i + 1), i % 5, 0))
    out = gate_by_onsets(notes, y, sr, t)
    assert len(out) == len(notes), "well-evidenced notes must all survive"


def test_youtube_url_detection_and_title_parsing():
    from chartgen.youtube import is_youtube_url, parse_title

    for url in ("https://www.youtube.com/watch?v=abc123",
                "https://youtu.be/abc123",
                "https://music.youtube.com/watch?v=abc123",
                "http://m.youtube.com/watch?v=abc123"):
        assert is_youtube_url(url), url
    for not_url in (r"C:\songs\track.mp3", "song.opus",
                    "https://vimeo.com/123", "youtube song name"):
        assert not is_youtube_url(not_url), not_url

    assert parse_title("Alan Walker - Faded (Official Video)") == ("Alan Walker", "Faded")
    assert parse_title("Daft Punk - Get Lucky [Official Audio] (HD)") == ("Daft Punk", "Get Lucky")
    assert parse_title("Bohemian Rhapsody (Remastered 2011)") == ("", "Bohemian Rhapsody")
    artist, name = parse_title("YOASOBI「アイドル」 Official Music Video")
    assert name, (artist, name)


def test_guess_metadata_prefers_tags_then_filename():
    import tempfile

    import mutagen.flac
    import soundfile as sf

    from chartgen.tags import guess_metadata

    with tempfile.TemporaryDirectory() as tmp:
        # Tagged file: embedded metadata wins over the filename.
        tagged = Path(tmp) / "random_download_x93.flac"
        sf.write(tagged, np.zeros(2048, dtype=np.float32), 22050)
        flac = mutagen.flac.FLAC(tagged)
        flac["artist"], flac["title"] = "Daft Punk", "Around the World"
        flac.save()
        assert guess_metadata(tagged) == ("Daft Punk", "Around the World")

        # Untagged file: the "Artist - Title" filename convention.
        plain = Path(tmp) / "Queen - Bohemian Rhapsody (Remastered).flac"
        sf.write(plain, np.zeros(2048, dtype=np.float32), 22050)
        assert guess_metadata(plain) == ("Queen", "Bohemian Rhapsody")

        # No tags, no dash: title falls back to the stem, artist stays empty.
        bare = Path(tmp) / "jam.flac"
        sf.write(bare, np.zeros(2048, dtype=np.float32), 22050)
        artist, name = guess_metadata(bare)
        assert (artist, name) == ("", "jam")


def test_folder_name_helper_and_skip_check_agree():
    """skip_existing predicts the folder before generating; the prediction
    must match what the writer actually creates, or skips silently break."""
    from chartgen.pipeline import _folder_name

    assert _folder_name("A", 'B - "C"') == "A - B - C"
    assert _folder_name("", "Song") == "Song"
    assert _folder_name("", '"??"') == "song"
    assert _folder_name("Alan Walker", "Faded [chartgen]") == "Alan Walker - Faded [chartgen]"


def test_song_folder_names_survive_windows_illegal_characters():
    """A real YouTube title ('Baldur's Gate 3 - "I Want To Live"') crashed
    folder creation: quotes and friends are illegal in Windows paths. The
    folder is sanitised; metadata keeps the original text."""
    import re

    for title in ('BG3 - OST - "I Want To Live" (Acoustic)',
                  "What?! <Live/Mix> C:\\stuff|here*"):
        cleaned = re.sub(r'[<>:"/\\|?*]', "", f"Artist - {title}").strip(" -.")
        assert not re.search(r'[<>:"/\\|?*]', cleaned), cleaned
        assert cleaned, "sanitised name must not be empty"
    # A name that is nothing but illegal chars falls back to non-empty.
    assert (re.sub(r'[<>:"/\\|?*]', "", '"??"').strip(" -.") or "song") == "song"


def test_playlist_url_detection_is_strict():
    """Only pure playlist URLs expand; a video that happens to carry a
    &list= parameter must stay a single video."""
    from chartgen.youtube import expand_inputs, is_playlist_url

    assert is_playlist_url("https://www.youtube.com/playlist?list=PLx123")
    assert is_playlist_url("https://music.youtube.com/playlist?list=PLx123")
    assert not is_playlist_url("https://www.youtube.com/watch?v=abc&list=PLx123")
    assert not is_playlist_url("https://youtu.be/abc123")
    assert not is_playlist_url(r"C:\songs\track.mp3")

    # Non-playlist inputs pass through expand_inputs untouched, in order.
    items = [r"C:\a.mp3", "https://youtu.be/abc", r"C:\b.flac"]
    assert expand_inputs(items) == items


def test_tempo_octave_helpers():
    """The octave label decides HOPO vs strum feel; helpers must move only the
    label, never the physical grid."""
    from chartgen.tempo import TempoMap, _double

    slow = np.arange(32) * 1.0  # 60 BPM
    doubled = _double(slow)
    assert abs(TempoMap(beat_times=doubled, pickup_beats=0).bpm - 120.0) < 0.5
    # Every original beat survives doubling: the grid only gains midpoints.
    assert all(any(abs(doubled - b) < 1e-9) for b in slow)


def test_conditioner_is_exactly_identity_when_untrained():
    """Zero-init is load-bearing: it lets a pretrained checkpoint keep its quality.

    If this drifts to small-random init, adding the conditioner would perturb a
    trained model on step 0 and the fine-tune would start from damage.
    """
    import torch

    from chartgen.conditioning import PitchConditioner

    cond = PitchConditioner(d_model=64)
    assert cond.is_identity
    bias = cond(torch.randn(3, 40, cond.proj.in_features))
    assert bias.shape == (3, 40, 64)
    assert torch.count_nonzero(bias) == 0, "untrained conditioner must add nothing"

    torch.nn.init.normal_(cond.proj.weight, std=0.2)
    assert not cond.is_identity
    assert torch.count_nonzero(cond(torch.randn(3, 40, cond.proj.in_features))) > 0


def test_pack_features_shape_and_mismatch():
    import torch

    from chartgen.conditioning import N_PITCH_FEATURES, pack_features

    packed = pack_features(torch.rand(50, 48), torch.rand(50), torch.ones(50))
    assert packed.shape == (50, N_PITCH_FEATURES)
    try:
        pack_features(torch.rand(50, 48), torch.rand(49), torch.ones(50))
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched frame counts must be rejected loudly")


def test_align_frames_pads_with_edge_not_silence():
    """Padding with zeros would read as 'nothing sounding' and fight `voiced`."""
    import torch

    from chartgen.conditioning import align_frames

    x = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    padded = align_frames(x, 6)
    assert padded.shape == (1, 6, 3)
    assert torch.equal(padded[0, 4], x[0, -1]), "should repeat the last frame"
    assert torch.equal(padded[0, 5], x[0, -1])
    assert align_frames(x, 2).shape == (1, 2, 3)
    assert align_frames(x, 4) is x


def test_windows_for_chunks_maps_sample_offsets_to_frames():
    """Chunk offsets are sample counts; frames must line up exactly."""
    import torch

    from chartgen.conditioning import windows_for_chunks

    grid_ms, sr = 40, 24000
    frames_per_chunk = int(30 * 1000 / grid_ms)  # 750
    pitch = torch.arange(3000, dtype=torch.float32).reshape(3000, 1)
    starts = [0, 30 * sr, 60 * sr]
    batch = windows_for_chunks(pitch, starts, frames_per_chunk, sr, grid_ms)
    assert batch.shape == (3, frames_per_chunk, 1)
    # 30s at 40ms is 750 frames, so chunk n starts at frame 750n.
    assert batch[0, 0, 0] == 0
    assert batch[1, 0, 0] == 750
    assert batch[2, 0, 0] == 1500


def test_sixteenth_grid_stays_inside_the_natural_hopo_window():
    """HOPOs are implicit in .chart, so the grid choice is what creates them.

    At resolution 480 a 16th is 120 ticks, inside the 160-tick window, so 16th
    runs HOPO and 8ths strum. If either the resolution or the default subdivision
    changes, HOPOs silently disappear — hence this guard.
    """
    thr = hopo_threshold(RESOLUTION)
    assert thr == 162, thr
    assert hopo_threshold(192) == 65, "must match the spec's reference value"
    assert RESOLUTION // 4 < thr, "16ths must fall inside the HOPO window"
    assert RESOLUTION // 2 > thr, "8ths must strum, not HOPO"


def test_smooth_fret_jumps_matches_human_hand_movement():
    """Human charts of the same songs jump 3+ lanes on 0-1% of consecutive
    single notes; ours measured 9-30%. Clamping must kill the teleports while
    keeping pitch direction, and leave long rests free to relocate."""
    from chartgen.frets import smooth_fret_jumps, step_share

    t = steady(n=64)
    res = RESOLUTION
    teleporting = [(i * (res // 2), 4 if i % 2 == 0 else 0, 0) for i in range(32)]
    assert step_share(teleporting, 3) > 0.9, "fixture should be all teleports"

    out = smooth_fret_jumps(teleporting, t, max_step=2)
    assert step_share(out, 3) == 0.0, out[:6]
    assert [n[0] for n in out] == [n[0] for n in teleporting], "timing moved"
    assert all(0 <= lane <= 4 for _, lane, _ in out)

    # A rising line still rises — contour is what the guidelines ask for.
    rising = [(i * (res // 2), min(4, i), 0) for i in range(10)]
    lanes = [lane for _, lane, _ in smooth_fret_jumps(rising, t, max_step=2)]
    assert lanes == sorted(lanes), lanes

    # Two beats of rest is enough time to move the hand anywhere.
    far = [(0, 0, 0), (8 * res, 4, 0)]
    assert smooth_fret_jumps(far, t, max_step=2) == far

    # Chords move as a unit, keeping their shape.
    chord = [(0, 0, 0), (0, 2, 0), (res // 2, 3, 0), (res // 2, 4, 0)]
    out = smooth_fret_jumps(chord, t, max_step=1)
    second = sorted(lane for tick, lane, _ in out if tick == res // 2)
    assert second[1] - second[0] == 1, f"chord shape distorted: {second}"


def test_top_up_sustains_is_bounded_by_the_measured_share():
    """Deriving sustains from a reduced tier's own gaps once made 39 of Easy's
    40 notes sustains. The share bound is what makes it safe to do at all."""
    from chartgen.expression import TIER_SUSTAIN_SHARE, top_up_sustains

    res = RESOLUTION
    # Sparse tier with room everywhere: every note COULD sustain.
    notes = [(i * 4 * res, i % 3, 0) for i in range(40)]
    out = top_up_sustains(notes, res, end_tick=200 * res,
                          target_share=TIER_SUSTAIN_SHARE["EasySingle"])
    share = sum(1 for _, _, sus in out if sus > 0) / len(out)
    # The bound tracks the measured Easy-tier median (22.2% across the
    # 800-chart calibration set), not a hardcoded band that goes stale
    # every time the calibration data grows.
    target = TIER_SUSTAIN_SHARE["EasySingle"]
    assert target - 0.08 <= share <= target + 0.05, \
        f"share {share:.0%} strays from the {target:.0%} target"

    # Sustains still never reach the next note.
    for (tick, _, sus), (nxt, _, _) in zip(out, out[1:]):
        if nxt != tick:
            assert tick + sus < nxt, f"sustain at {tick} runs into {nxt}"

    # Notes that already sustain are left alone, and a tier already at target
    # gains nothing.
    already = [(i * 4 * res, 0, res) for i in range(10)]
    assert top_up_sustains(already, res, 200 * res, 0.05) == already


def test_section_reuse_copies_lanes_but_keeps_the_repeat_rhythm():
    """Human charts overlap 55% between repeated sections; ours managed 13%.
    Reuse must raise that without pasting a section wholesale — the later
    section keeps whatever rhythm the audio gave it."""
    from chartgen.structure import find_repeats, reuse_patterns

    res = RESOLUTION
    bar = 4 * res
    # Two 4-bar sections. The second has the same rhythm on beats but adds an
    # extra note the first does not have, and different lanes throughout.
    first = [(i * res, i % 5, 0) for i in range(16)]
    second = [(4 * bar + i * res, (i + 2) % 5, 0) for i in range(16)]
    second.append((4 * bar + res // 2, 3, 0))   # unique to the repeat
    notes = sorted(first + second)

    out = reuse_patterns(notes, [0, 4 * bar], {1: 0})
    out_by_tick = {}
    for tick, lane, _ in out:
        out_by_tick.setdefault(tick, []).append(lane)

    # Lanes now match the first section wherever the two line up.
    for i in range(16):
        assert out_by_tick[4 * bar + i * res] == [i % 5], i
    # The repeat's own extra note survives untouched.
    assert out_by_tick[4 * bar + res // 2] == [3]
    # Timing is never invented or dropped.
    assert {t for t, _, _ in out} == {t for t, _, _ in notes}

    # find_repeats needs both similar harmony and comparable length.
    import numpy as np
    a = np.ones(12) / np.sqrt(12)
    b = np.zeros(12); b[0] = 1.0
    assert find_repeats([a, a], [bar, bar]) == {1: 0}
    assert find_repeats([a, b], [bar, bar]) == {}, "different harmony"
    assert find_repeats([a, a], [bar, 4 * bar]) == {}, "different length"


def test_tempo_map_extrapolates_past_the_last_detected_beat():
    """librosa stops tracking beats when a song fades out. np.interp clamps
    past its last beat, so every outro note quantized onto one tick and charts
    ended up to 25s early — on both engines, since the tempo map is shared."""
    t = steady(bpm=120.0, n=32, first=0.5)   # beats end at ~16s
    last_beat = float(t.beat_times[-1])

    # Ten seconds past the last beat must be ten seconds of ticks further on,
    # not pinned to the final beat.
    a = t.quantize(last_beat + 1.0)
    b = t.quantize(last_beat + 10.0)
    assert b > a, "outro notes collapsed onto one tick"
    beats_apart = (b - a) / RESOLUTION
    assert 17 < beats_apart < 19, f"9s at 120bpm should be ~18 beats, got {beats_apart}"

    # And the mapping round-trips, so sustains and section times stay honest.
    for probe in (last_beat + 2.0, last_beat + 12.0):
        tick = t.quantize(probe)
        assert abs(t.beat_to_time(tick / RESOLUTION) - probe) < 0.3, probe


def test_playability_simulator_flags_and_fixes_the_unplayable():
    """Statistical gates passed charts that no hand could execute. This models
    the hands: strumming and re-shaping cost time, and the chart supplies a
    fixed amount. Human charts sit at ~1.3% impossible transitions."""
    from chartgen.playability import analyse, enforce

    t = steady(bpm=180.0, n=200)
    res = RESOLUTION
    # Alternating two-note chords on every 16th at 180 BPM: the hand would
    # have to re-form a shape every 83ms.
    brutal = []
    for i in range(80):
        tick = i * (res // 4)
        lanes = (0, 2) if i % 2 == 0 else (2, 4)
        brutal += [(tick, l, 0) for l in lanes]
    before = analyse(brutal, t)
    assert before["impossible"] > 0.5, before
    assert before["worst"], "should report where it breaks down"

    fixed = enforce(brutal, t)
    assert analyse(fixed, t)["impossible"] == 0.0
    assert {n[0] for n in fixed} <= {n[0] for n in brutal}, "no notes invented"

    # A comfortable chart is left completely alone.
    easy = [(i * res, i % 3, 0) for i in range(32)]
    assert analyse(easy, t)["impossible"] == 0.0
    assert enforce(easy, t) == sorted(easy)


def test_lrc_lines_become_ordered_phrases():
    from chartgen.lyrics import from_lines

    t = steady(bpm=120.0, n=400)
    lines = [(10.0, "You were the shadow to my light"),
             (14.0, "Did you feel us"),
             (60.0, "So lost, I'm faded")]
    events = from_lines(lines, t, duration_s=200.0)

    words = [e for _, e in events if e.startswith("lyric ")]
    assert len(words) == 15, words
    ticks = [tick for tick, e in events if e.startswith("lyric ")]
    assert ticks == sorted(ticks), "words must never move backwards"
    # Every phrase opens before its own first word and closes after its last.
    depth = 0
    for _, event in events:
        if event == "phrase_start":
            assert depth == 0, "phrases must not nest"
            depth = 1
        elif event == "phrase_end":
            assert depth == 1, "phrase closed without opening"
            depth = 0
        else:
            assert depth == 1, "a word landed outside any phrase"
    assert depth == 0

    # A line before a long instrumental break must not crawl across it: the
    # third line's words all sit near 60s, not spread to the end of the song.
    tail = [tick for tick, e in events if e.startswith("lyric ")][-3:]
    assert max(tail) / t.resolution < 70 * 2, tail


def test_lrc_parsing_skips_instrumental_markers():
    from chartgen.lrclib import parse_lrc

    # Blank timestamps mark instrumental breaks and a lone note glyph marks
    # a vocalise; both are real information, neither belongs on the highway.
    body = "\n".join((
        "[00:11.14]You were the shadow",
        "[00:14.00]",
        "[00:20.05] Another star",
        "[00:17.77]♪",
    ))
    lines = parse_lrc(body)
    assert lines == [(11.14, "You were the shadow"), (20.05, "Another star")], lines


def test_invented_outro_is_dropped_but_a_real_ending_survives():
    from chartgen.lyrics import _drop_invented_tail

    def seg(start, words):
        step = 0.3
        return (start, start + step * len(words),
                [(start + i * step, w) for i, w in enumerate(words)])

    verse = [seg(10.0, ["I", "walk", "alone"]), seg(14.0, ["through", "the", "night"])]

    # Measured in a real chart: eleven words of YouTube caption boilerplate,
    # every one stamped with the same timestamp, stranded after the outro.
    hallucinated = (95.0, 96.0, [(95.0, w) for w in
                                 "Thank you for watching I'll see you in the next one".split()])
    assert _drop_invented_tail(verse + [hallucinated]) == verse

    # A bare word marooned after a long instrumental break: also invented.
    assert _drop_invented_tail(verse + [seg(95.0, ["you"])]) == verse

    # But a genuine closing line, sung right after the last verse, stays.
    ending = seg(17.0, ["and", "then", "I", "was", "alone"])
    assert _drop_invented_tail(verse + [ending]) == verse + [ending]


def test_same_tick_lyrics_keep_their_written_order():
    from chartgen.chart import write_chart

    t = steady(bpm=120.0, n=64)
    # Whisper stamps a hallucinated phrase with one timestamp for every word;
    # sorting those lines by text read "thank i'll and for in next one".
    words = ["Thank", "you", "for", "watching"]
    lyrics = [(1920, "phrase_start")] + [(2400, f"lyric {w}") for w in words]         + [(2400, "phrase_end")]
    chart = write_chart({"ExpertSingle": [(0, 0, 0), (480, 1, 0)]}, t,
                        {"name": "x", "artist": "y", "charter": "z"},
                        "song.ogg", lyrics=lyrics)
    written = [line.split('"')[1] for line in chart.splitlines()
               if "lyric " in line or "phrase_" in line]
    assert written == ["phrase_start"] + [f"lyric {w}" for w in words] + ["phrase_end"],         written


def test_synced_lyrics_are_rejected_when_they_miss_the_audio():
    from chartgen.lyrics import _lands_on_sound

    sr = 22050
    # A song that is silent for its first half: a lyric sheet timed against a
    # different edit (a YouTube rip with an added intro matches on duration
    # but not on where the words land) puts every line in the silence.
    quiet = np.zeros(30 * sr, dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.arange(30 * sr) / sr)
    y = np.concatenate([quiet, tone]).astype(np.float32)

    assert not _lands_on_sound([(t, "x") for t in range(2, 28, 2)], y, sr)
    assert _lands_on_sound([(t, "x") for t in range(32, 58, 2)], y, sr)
    # With no audio to check against, the sheet is taken at its word.
    assert _lands_on_sound([(2.0, "x")], None, None)


def test_solo_scoring_rewards_the_standout_and_stays_quiet_otherwise():
    """The audio detector's scoring, on synthetic evidence.

    The full detector needs audio; the decision logic does not. A section
    that is brighter, more voiced AND more novel than the rest of its song
    must clear the swept threshold, and a song whose sections all look alike
    must produce nothing - 70% of human charts mark no solo, so silence is
    the default being protected here.
    """
    from chartgen import solo

    def rows(standout=None):
        out = []
        for i in range(8):
            hot = i == standout
            out.append({
                "voiced": 0.55 if hot else 0.25,
                "confidence": 0.4,
                "spread": 1.2,
                "bright": 1400.0 if hot else 950.0,
                "novelty": 0.55 if hot else 0.18,
            })
        return out

    spans = [(i * 16 * 480, (i + 1) * 16 * 480) for i in range(8)]

    z = solo._zscores(rows(standout=5))
    cands = solo._candidates(spans, z, 480)
    scores = {(a, b): sum(solo.WEIGHTS[k] * zz[k] for k in solo.FEATURES)
              for a, b, zz in cands}
    hot_span = spans[5]
    assert scores[hot_span] >= solo.MIN_Z, scores[hot_span]
    assert scores[hot_span] == max(scores.values())

    # A flat song: nothing may clear the bar.
    z = solo._zscores(rows(standout=None))
    cands = solo._candidates(spans, z, 480)
    assert all(sum(solo.WEIGHTS[k] * zz[k] for k in solo.FEATURES) < solo.MIN_Z
               for _, _, zz in cands)


def test_tap_phrases_need_soft_runs_and_respect_the_share_cap():
    from chartgen import taps

    res = RESOLUTION
    notes = [(i * res // 2, i % 5, 0) for i in range(60)]  # 8ths, 60 notes
    ticks = [t for t, _, _ in notes]

    # A soft island shorter than MIN_RUN never taps; a long one does, whole.
    soft = {t: -1.0 for t in ticks}
    for t in ticks[10:10 + taps.MIN_RUN - 1]:
        soft[t] = 1.5
    assert taps.phrases(notes, soft, res) == set()

    for t in ticks[20:36]:
        soft[t] = 1.5
    got = taps.phrases(notes, soft, res)
    assert got == set(ticks[20:36]), "the full soft run should tap"

    # A gap wider than MAX_GAP_BEATS splits a run; the halves stand alone.
    notes2 = [(i * res * 2, 0, 0) for i in range(20)]  # half notes: 2-beat gaps
    soft2 = {t: 1.5 for t, _, _ in notes2}
    assert taps.phrases(notes2, soft2, res) == set(),         "notes too far apart never form a tapped phrase"

    # Share cap: when everything reads soft, most of the chart must stay
    # strummed - tapping it all would just change instruments.
    soft3 = {t: 1.5 for t in ticks}
    got = taps.phrases(notes, soft3, res)
    assert len(got) <= taps.MAX_SHARE * len(ticks) + 1, len(got)


def test_tap_markers_written_only_where_the_tier_keeps_the_note():
    from chartgen.chart import _tier_block

    notes = [(0, 0, 0), (480, 1, 0), (960, 2, 0)]
    block = _tier_block("HardSingle", notes, [], taps={480, 5000})
    assert "480 = N 6 0" in block, "tap marker missing at a kept note"
    assert "5000 = N 6 0" not in block, "bare tap marker with no note is dead weight"
    # The modifier and its note share a tick and both survive sorting.
    lines = [l.strip() for l in block.splitlines() if "480" in l]
    assert "480 = N 1 0" in lines and "480 = N 6 0" in lines


def test_lrc_phrases_stay_on_screen_until_the_next_line():
    """Playtest: lines vanished while their last word was still sung. A
    phrase must close when the NEXT line begins (LRC line timing means
    'displayed until then'), not a quarter-beat after its last word."""
    from chartgen.lyrics import from_lines

    t = steady(bpm=120.0, n=400)
    lines = [(10.0, "first line of words here"), (18.0, "second line"),
             (26.0, "final line")]
    events = from_lines(lines, t, duration_s=120.0)

    ends = [tick for tick, e in events if e == "phrase_end"]
    starts = [tick for tick, e in events if e == "phrase_start"]
    res = t.resolution
    # First phrase holds until just before the second line starts (18s at
    # 120bpm = beat 36), not until shortly after its last word (~11s).
    assert ends[0] >= int(17.0 * 2 * res), (ends[0], "closed too early")
    assert ends[0] <= starts[1], "phrases must not overlap"
    # The last line rings out rather than dying instantly.
    assert ends[-1] >= int(27.0 * 2 * res)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
