"""Keyed rescue vs the shipped stem rescue, against human note times.

For each song: the mix transcription, the followed-instrument timeline
(which also yields every stem's transcription), then both rescues on the
same starved material. Precision = share of rescued events whose onset
lands within 80 ms of a human-charted note; recall-ish = human notes in
the rescued windows that gained a note within 80 ms. The bar: keyed must
beat the shipped rescue's 0.70-0.72 median precision on the 13 songs
where the starved-run trigger fires (2026-09-03 verification), and it
must not fill fewer holes.

    python tools/keyed_rescue_eval.py FOLDER [FOLDER ...]
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dump_foreground import human_truth  # noqa: E402
from validate_solos import find_audio  # noqa: E402
from chartgen import prominence, stems as stemsmod, tempo as tempomod, transcribe  # noqa: E402

TOL = 0.08


def near(times, human):
    if not len(times) or not len(human):
        return 0.0
    h = np.sort(human)
    i = np.searchsorted(h, times)
    lo = np.abs(times - h[np.clip(i - 1, 0, len(h) - 1)])
    hi = np.abs(times - h[np.clip(i, 0, len(h) - 1)])
    return float((np.minimum(lo, hi) <= TOL).mean())


def main(argv):
    rows = []
    for folder in map(Path, argv):
        chart, audio = folder / "notes.chart", find_audio(folder)
        if not chart.is_file() or audio is None:
            print(f"skip {folder.name}", flush=True)
            continue
        truth = human_truth(chart)
        if truth is None:
            continue
        human = np.array(truth[0])
        y, sr = tempomod.load(str(audio))
        tm = tempomod.detect(y, sr)
        events = transcribe.transcribe(str(audio))
        stemsmod._CACHE.clear()
        stem_events = {}
        windows = prominence.timeline(str(audio), tm, len(y) / sr, known_events=stem_events)
        st = stemsmod.separate(str(audio), backend="sw")
        keyed, touched, counts = (prominence.keyed_rescue_events(events, windows, stem_events, tm)
                                  if windows else ([], 0, {}))
        old = transcribe.stem_rescue_events(events, tm, st) if st is not None else []
        kt = np.array([e[0] for e in keyed])
        ot = np.array([e[0] for e in old])
        # holes filled: human notes inside keyed windows without a mix note within TOL
        mix_t = np.array([e[0] for e in events if e[2] >= 40 and e[3] >= 0.2])
        row = {"song": folder.name, "keyed_n": len(keyed), "keyed_p": near(kt, human),
               "old_n": len(old), "old_p": near(ot, human), "windows": touched, "counts": counts,
               "timeline": prominence.letters(windows) if windows else "-"}
        rows.append(row)
        print(f"{folder.name[:40]:42s} keyed {row['keyed_n']:4d} notes P {row['keyed_p']:.2f} "
              f"({touched} win, {counts})  | old rescue {row['old_n']:4d} notes P {row['old_p']:.2f}", flush=True)
    both = [r for r in rows if r["keyed_n"] and r["old_n"]]
    print("\n=== summary ===")
    if both:
        print(f"songs where both fired: {len(both)}; median precision keyed {np.median([r['keyed_p'] for r in both]):.2f} "
              f"vs old {np.median([r['old_p'] for r in both]):.2f}; keyed better on "
              f"{sum(1 for r in both if r['keyed_p'] > r['old_p'])}/{len(both)}; notes/song median keyed "
              f"{np.median([r['keyed_n'] for r in both]):.0f} vs old {np.median([r['old_n'] for r in both]):.0f}")
    ko = [r for r in rows if r["keyed_n"] and not r["old_n"]]
    ok = [r for r in rows if r["old_n"] and not r["keyed_n"]]
    print(f"keyed only: {len(ko)} songs (median P {np.median([r['keyed_p'] for r in ko]) if ko else 0:.2f}); "
          f"old only: {len(ok)} songs; neither: {sum(1 for r in rows if not r['keyed_n'] and not r['old_n'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
