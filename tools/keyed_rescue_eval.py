"""Keyed rescue vs the shipped stem rescue, against human note times.

For each song: the mix transcription, the followed-instrument timeline
(which also yields every stem's transcription), then both rescues on the
same starved material. Precision = share of rescued events whose onset
lands within 80 ms of a human-charted note. The bar: keyed must beat the
shipped rescue's ~0.70 median precision on the 13 songs where the
starved-run trigger fires (2026-09-03 verification) without filling
fewer holes.

Everything expensive is cached per song under work/keyed_cache, so rule
variants (amplitude floor, density floor, which stems, union with the
old rescue) sweep in seconds:

    python tools/keyed_rescue_eval.py FOLDER [FOLDER ...]   # fills the cache, prints the baseline
    python tools/keyed_rescue_eval.py --sweep                # variants over the cache
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_foreground import human_truth  # noqa: E402
from validate_solos import find_audio  # noqa: E402
from chartgen import prominence, stems as stemsmod, tempo as tempomod, transcribe  # noqa: E402
from chartgen.tempo import TempoMap  # noqa: E402

TOL = 0.08
CACHE = Path(__file__).resolve().parent.parent / "work" / "keyed_cache"


def near(times, human):
    if not len(times) or not len(human):
        return 0.0
    h = np.sort(human)
    i = np.searchsorted(h, times)
    lo = np.abs(times - h[np.clip(i - 1, 0, len(h) - 1)])
    hi = np.abs(times - h[np.clip(i, 0, len(h) - 1)])
    return float((np.minimum(lo, hi) <= TOL).mean())


def build(folder: Path):
    chart, audio = folder / "notes.chart", find_audio(folder)
    if not chart.is_file() or audio is None:
        return None
    truth = human_truth(chart)
    if truth is None:
        return None
    y, sr = tempomod.load(str(audio))
    tm = tempomod.detect(y, sr)
    events = transcribe.transcribe(str(audio))
    stemsmod._CACHE.clear()
    stem_events = {}
    windows = prominence.timeline(str(audio), tm, len(y) / sr, known_events=stem_events)
    st = stemsmod.separate(str(audio), backend="sw")
    old = transcribe.stem_rescue_events(events, tm, st) if st is not None else []
    return {"song": folder.name, "human": np.array(truth[0]), "beat_times": tm.beat_times,
            "pickup": tm.pickup_beats, "events": events, "stem_events": stem_events,
            "windows": windows, "old": old}


def score(c, admit_amp=0.50, min_per_s=None, stems=None, union=False, fallback=False):
    tm = TempoMap(beat_times=c["beat_times"], pickup_beats=c["pickup"])
    saved = (prominence.KEYED_MIN_STEM_PER_S, prominence.KEYED_STEMS)
    try:
        if min_per_s is not None:
            prominence.KEYED_MIN_STEM_PER_S = min_per_s
        if stems is not None:
            prominence.KEYED_STEMS = stems
        keyed, touched, counts = (prominence.keyed_rescue_events(
            c["events"], c["windows"], c["stem_events"], tm, admit_amplitude=admit_amp)
            if c["windows"] else ([], 0, {}))
    finally:
        prominence.KEYED_MIN_STEM_PER_S, prominence.KEYED_STEMS = saved
    if fallback and not keyed:
        keyed = list(c["old"])   # song-level: the old blended rescue where keyed found nothing
    if union:
        taken = {tm.quantize(s, subdiv=4) for s, _, p, a in c["events"] if p >= 40 and a >= 0.2}
        taken |= {tm.quantize(s, subdiv=4) for s, _, _, _ in keyed}
        extra = [e for e in c["old"] if tm.quantize(e[0], subdiv=4) not in taken]
        keyed = keyed + extra
    t = np.array([e[0] for e in keyed])
    return len(keyed), near(t, c["human"]), touched


def report(caches, label, **kw):
    rows = []
    for c in caches:
        n, p, touched = score(c, **kw)
        on = len(c["old"])
        op = near(np.array([e[0] for e in c["old"]]), c["human"]) if on else 0.0
        rows.append((n, p, on, op))
    both = [(n, p, on, op) for n, p, on, op in rows if n and on]
    fired = [(n, p) for n, p, _, _ in rows if n]
    print(f"{label:44s} fires {len(fired):2d}/{len(rows)}  median P {np.median([p for _, p in fired]) if fired else 0:.2f}  "
          f"notes/song {np.median([n for n, _ in fired]) if fired else 0:4.0f}  |  where both fire ({len(both)}): "
          f"keyed {np.median([p for _, p, _, _ in both]) if both else 0:.2f} vs old {np.median([op for _, _, _, op in both]) if both else 0:.2f}, "
          f"keyed better {sum(1 for _, p, _, op in both if p > op)}/{len(both)}")


def main(argv):
    CACHE.mkdir(parents=True, exist_ok=True)
    if argv and argv[0] == "--sweep":
        caches = [pickle.loads(p.read_bytes()) for p in sorted(CACHE.glob("*.pkl"))]
        print(f"{len(caches)} cached songs; shipped rescue median P on them: "
              f"{np.median([near(np.array([e[0] for e in c['old']]), c['human']) for c in caches if c['old']]):.2f}\n")
        # admission floor decoupled from the starvation test (gates stay at 0.20)
        for amp in (0.20, 0.30, 0.40, 0.50, 0.60):
            report(caches, f"admit amp>={amp:.2f}", admit_amp=amp)
        report(caches, "admit>=0.50, >=3 stem notes/s", admit_amp=0.50, min_per_s=3.0)
        report(caches, "admit>=0.50, guitar+piano only", admit_amp=0.50, stems=("guitar", "piano"))
        report(caches, "admit>=0.40 + song-level fallback to old", admit_amp=0.40, fallback=True)
        report(caches, "admit>=0.50 + song-level fallback to old", admit_amp=0.50, fallback=True)
        report(caches, "admit>=0.60 + song-level fallback to old", admit_amp=0.60, fallback=True)
        return 0
    for folder in map(Path, argv):
        out = CACHE / (folder.name.replace("/", "_")[:80] + ".pkl")
        if out.is_file():
            c = pickle.loads(out.read_bytes())
        else:
            c = build(folder)
            if c is None:
                print(f"skip {folder.name}", flush=True)
                continue
            out.write_bytes(pickle.dumps(c))
        n, p, touched = score(c)
        on = len(c["old"])
        op = near(np.array([e[0] for e in c["old"]]), c["human"]) if on else 0.0
        print(f"{c['song'][:40]:42s} keyed {n:4d} notes P {p:.2f} ({touched} win)  | old rescue {on:4d} notes P {op:.2f}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
