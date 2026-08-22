"""Does the model's fret choice respond to pitch at all?

This decides whether fine-tuning can fix fret assignment. If frets track pitch
even weakly, the signal is there and better training data should sharpen it. If
frets are identical for a rising and a falling scale, the model is pitch-blind
and no amount of curated data will help — the architecture would need pitch
conditioning added.

Feeds the model three unambiguous signals and correlates fret against pitch.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402  (puts vendor trees on sys.path)
from chartgen import chart as chartio  # noqa: E402
from chartgen import tempo as tempomod  # noqa: E402

SR = 44100
BPM = 120.0


def pluck(freq, dur):
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    env = np.exp(-4.0 * t / dur)
    return (sum(np.sin(2 * np.pi * freq * h * t) / h for h in (1, 2, 3, 4)) * env).astype(np.float32)


def render(freqs, path, seconds=40.0):
    """One note per eighth, cycling through freqs, with a click track."""
    eighth = 60.0 / BPM / 2
    buf = np.zeros(int(SR * seconds) + SR, dtype=np.float32)
    for i in range(int(seconds / eighth) + 1):
        note = pluck(freqs[i % len(freqs)], eighth * 1.6)
        start = int(i * eighth * SR)
        buf[start:start + len(note)] += note
    for i in range(int(seconds / (60.0 / BPM)) + 1):
        k = pluck(55.0, 0.12)
        start = int(i * (60.0 / BPM) * SR)
        buf[start:start + len(k)] += k * 1.2
    buf = buf[:int(SR * seconds)]
    sf.write(path, buf / (np.max(np.abs(buf)) * 1.05), SR)


# Two octaves of E minor, and its exact reverse.
RISING = [82.41, 98.00, 110.00, 123.47, 146.83, 164.81, 196.00, 220.00,
          246.94, 293.66, 329.63, 392.00, 440.00, 493.88, 587.33, 659.26]
FALLING = RISING[::-1]
ALTERNATING = [82.41, 659.26] * 8  # low/high octave leaps


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="3podi/charter-v1.0-40-S-best-acc",
                    help="hub repo id or local export dir from tools/finetune.py")
    ap.add_argument("--conditioner", default=None,
                    help="conditioner.pt — probe the pitch-conditioned model; "
                         "a successful fine-tune must turn the falling-scale "
                         "correlation strongly negative")
    args = ap.parse_args()

    from chart.tokenizer import SimpleTokenizerGuitar
    from chartgen.model import PitchCharter, load_charter
    import torch

    model = load_charter(args.model)
    if args.conditioner:
        pitched = PitchCharter(model)
        pitched.conditioner.load_state_dict(
            torch.load(args.conditioner, map_location="cpu"))
        model = pitched
        assert not pitched.conditioner.is_identity, \
            "conditioner is all zeros — this probes the unconditioned model"
    reverse_map = SimpleTokenizerGuitar().reverse_map

    Path("work").mkdir(exist_ok=True)
    results = {}
    for label, freqs in (("rising", RISING), ("falling", FALLING),
                         ("alternating", ALTERNATING)):
        path = f"work/probe_{label}.wav"
        render(freqs, path)
        y, sr = tempomod.load(path)
        tm = tempomod.detect(y, sr)
        seqs = model.generate(path, temperature=0.5, top_k=32)
        tokens = torch.cat(seqs).flatten().cpu().tolist()
        notes = chartio.notes_from_tokens(tokens, model.config.grid_ms, tm, reverse_map)

        by_tick = {}
        for tick, lane, _ in notes:
            by_tick.setdefault(tick, []).append(lane)
        singles = [v[0] for _, v in sorted(by_tick.items())
                   if len(v) == 1 and v[0] != 7]
        # Pitch the fixture actually played at each successive onset.
        pitches = [freqs[i % len(freqs)] for i in range(len(singles))]
        corr = (np.corrcoef(np.log(pitches), singles)[0, 1]
                if len(set(singles)) > 1 else float("nan"))
        results[label] = (singles, corr)
        print(f"{label:<12} n={len(singles):<4} corr(log pitch, fret) = {corr:+.3f}")
        print(f"             first 24 frets: {singles[:24]}")

    print("\n--- interpretation ---")
    r, f = results["rising"][1], results["falling"][1]
    print(f"rising corr {r:+.3f} vs falling corr {f:+.3f}")
    # corr(log pitch, fret) is POSITIVE for any true mapping regardless of
    # scale direction — when pitch descends, frets descend with it. (The
    # original version expected the falling case to go negative, which no
    # correct mapping can produce: chartgen/pitch.py's direct features score
    # +1.000 on all three fixtures.) The falling fixture is still the decisive
    # one because a +/-1 walk drifts upward against the rising fixture's
    # monotone pitch and fakes a positive number there.
    if not np.isnan(r) and not np.isnan(f):
        if r > 0.35 and f > 0.35:
            print("PITCH-AWARE: frets follow pitch in both directions.")
        elif abs(r) < 0.2 and abs(f) < 0.2:
            print("PITCH-BLIND: fret choice ignores pitch entirely.")
            print("Fine-tuning alone will NOT fix frets; the model needs pitch input.")
        else:
            print("PARTIAL: some pitch signal, but not a clean mapping.")


if __name__ == "__main__":
    main()
