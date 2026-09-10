"""Go/no-go for rhythm rescue on guitar: in starved guitar-followed windows,
do guitar-stem onsets that keyed rescue did NOT already supply land on
human notes?

For each cached guitar-led song: rebuild the chart's positions up to and
including keyed rescue, find guitar windows still starved (positions <
half the stem's onsets, stem audible, onsets >= 3/s), collect the stem
onsets there that are not within a 16th of an existing position, and
score them against the human chart: precision at 80 ms, and the recall
gain (human notes in those windows newly covered). Bar: precision >= 0.85
across the songs, as the synth/bass version measured, with a real recall
gain over keyed rescue.

    python tools/guitar_onset_gonogo.py
"""
import glob
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from chartgen import prominence, stems as stemsmod  # noqa: E402
from chartgen.tempo import TempoMap  # noqa: E402
from keyed_rescue_eval import near  # noqa: E402

CAL = ROOT / "data" / "calibration"
GUITAR_LED = ("Slipknot", "Haste", "Chiodos", "Cradle", "Scary", "Stuck", "Ryuichi", "Thrice")
TOL = 0.08


def main():
    rows = []
    for pk in sorted(glob.glob(str(ROOT / "work" / "keyed_cache" / "*.pkl"))):
        c = pickle.loads(open(pk, "rb").read())
        if not any(c["song"].startswith(g) for g in GUITAR_LED):
            continue
        tm = TempoMap(beat_times=c["beat_times"], pickup_beats=c["pickup"])
        res = tm.resolution
        ws = c["windows"] or []
        keyed = prominence.keyed_rescue_events(c["events"], ws, c["stem_events"], tm)[0] if ws else []
        taken = {tm.quantize(s, subdiv=4) for s, _, p, a in c["events"] if p >= 40 and a >= 0.2}
        taken |= {tm.quantize(s, subdiv=4) for s, _, _, _ in keyed}
        human = np.sort(c["human"])
        audio = next(iter(CAL.glob(c["song"] + "/song.*")), None)
        if audio is None:
            continue
        stemsmod._CACHE.clear()
        mono = stemsmod.separate(str(audio), backend="sw")
        sr = stemsmod.SR
        env = librosa.onset.onset_strength(y=mono["guitar"], sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="time", delta=0.05, backtrack=False)
        cand, h_in, h_before = [], [], []
        fired = 0
        for w in ws:
            if w.get("stem") != "guitar" or not w.get("features") or w["features"]["guitar"]["energy"] < 0.10:
                continue
            t0, t1 = w["t0"], w["t1"]
            on = onsets[(onsets >= t0) & (onsets < t1)]
            if len(on) / (t1 - t0) < 3.0:
                continue
            lo, hi = tm.quantize(t0, subdiv=4), tm.quantize(t1, subdiv=4)
            present = sum(1 for t in taken if lo <= t < hi)
            if present >= 0.5 * len(on):
                continue
            fired += 1
            new = [t for t in on if tm.quantize(t, subdiv=4) not in taken
                   and not any(abs(tm.quantize(t, subdiv=4) - e) <= res // 4
                               for e in (taken & {tm.quantize(t, subdiv=4) - res // 4, tm.quantize(t, subdiv=4) + res // 4}))]
            cand += new
            hw = human[(human >= t0) & (human < t1)]
            h_in += list(hw)
            existing = np.array(sorted(tm.beat_to_time(t / res) for t in taken if lo <= t < hi))
            h_before += [near(np.array([h]), existing) if len(existing) else 0.0 for h in hw]
        cand = np.array(sorted(cand))
        h_in = np.array(h_in)
        prec = near(cand, human) if len(cand) else 0.0
        recall_before = float(np.mean(h_before)) if len(h_before) else 0.0
        combined = np.sort(np.concatenate([cand, np.array(sorted(tm.beat_to_time(t / res) for t in taken))])) if len(cand) else None
        recall_after = float(np.mean([near(np.array([h]), combined) for h in h_in])) if combined is not None and len(h_in) else recall_before
        rows.append((c["song"][:34], fired, len(cand), prec, len(h_in), recall_before, recall_after))
        print(f"{c['song'][:34]:36s} starved guitar windows {fired:2d}  new onsets {len(cand):4d}  precision {prec:.2f}  "
              f"human notes there {len(h_in):4d}  covered before {recall_before:.0%} -> after {recall_after:.0%}", flush=True)
    fired = [r for r in rows if r[2]]
    if fired:
        print(f"\n=== {len(fired)} songs fire; median precision {np.median([r[3] for r in fired]):.2f}, "
              f"min {min(r[3] for r in fired):.2f}; median recall gain "
              f"{np.median([r[6] - r[5] for r in fired]):+.0%}; bar: precision >= 0.85")
    return 0


if __name__ == "__main__":
    sys.exit(main())
