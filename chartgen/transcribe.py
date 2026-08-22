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
MAX_CHORD = 3


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


def expert_from_notes(
    events,
    tempo,
    subdiv: int = 4,
    min_pitch: int = MIN_PITCH,
    min_amplitude: float = MIN_AMPLITUDE,
    min_sustain_beats: float = 0.5,
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

    fret_of = _fret_map(np.array([p for _, _, p, _ in kept], dtype=float))

    by_tick: dict[int, list[tuple[float, int, float]]] = {}
    for start, end, pitch, amp in kept:
        tick = tempo.quantize(start, subdiv=subdiv)
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
        lanes: dict[int, float] = {}
        for _, pitch, duration in group:
            lane = fret_of(pitch)
            lanes[lane] = max(lanes.get(lane, 0.0), duration)

        for lane, duration in sorted(lanes.items()):
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
            notes.append((tick, lane, max(0, sustain)))
    return notes


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
