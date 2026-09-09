"""Chord audit: where chords are added and culled, stage by stage, against
the human charts, over the songs cached by tools/keyed_rescue_eval.py.

Reproduces the chord-affecting part of the Basic Pitch path offline
(keyed rescue -> Expert -> chord texture -> triples -> texture -> rapid
chords -> playability -> late smoothing) with the same gates the
pipeline uses, prints the chord share after each stage, the human
chart's share, and which of the rapid-chord pass's three conditions
(rapid reshape / stray off-grid / flicker) would have culled each chord.

    python tools/chord_audit.py
"""
import collections
import glob
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chartgen import guitar as G, playability, prominence, reduce, texture, transcribe as T  # noqa: E402
from chartgen.tempo import TempoMap  # noqa: E402

CAL = ROOT / "data" / "calibration"
STAGES = ["expert", "chordify", "triples", "texture", "simplify", "playability", "late smooth"]


def share(notes):
    by = collections.defaultdict(set)
    for t, l, _ in notes:
        if l <= 4:
            by[t].add(l)
    return len(by), sum(1 for v in by.values() if len(v) >= 2)


def human_share(song):
    p = CAL / song / "notes.chart"
    if not p.is_file():
        return None
    body = re.search(r"\[ExpertSingle\]\s*\{(.*?)\}", p.read_text(encoding="utf-8-sig", errors="replace"), re.S).group(1)
    by = collections.defaultdict(set)
    for t, l in re.findall(r"(\d+)\s*=\s*N\s*(\d+)\s+\d+", body):
        if int(l) <= 4:
            by[int(t)].add(int(l))
    return sum(1 for v in by.values() if len(v) >= 2) / max(1, len(by))


def cull_reasons(pre, res, chord_spans):
    groups = collections.defaultdict(list)
    for t, l, s in pre:
        groups[t].append((t, l, s))
    ordered = sorted(groups)
    bar, beside = 4 * res, res // 4
    is_chord = {t: sum(1 for n in groups[t] if n[1] != 7) >= 2 for t in ordered}
    shape = {t: tuple(sorted(n[1] for n in groups[t] if n[1] != 7)) for t in ordered}
    offgrid = {t for t in ordered if is_chord[t] and t % (res // 2) != 0}
    push_ok = {t for t in offgrid if any((t + d * bar) in offgrid for d in (-2, -1, 1, 2))}
    in_span = (lambda t: any(a <= t < b for a, b in chord_spans)) if chord_spans else (lambda t: False)
    rapid = stray = flick = 0
    for i, t in enumerate(ordered):
        if not is_chord[t]:
            continue
        prev = ordered[i - 1] if i else None
        if prev is not None and is_chord[prev] and shape[prev] != shape[t] and t - prev < res // 2:
            rapid += 1
        elif t in offgrid and t not in push_ok and not in_span(t) and not (
                prev is not None and shape[prev] == shape[t] and t - prev <= beside):
            stray += 1
        elif not in_span(t) and ((prev is not None and not is_chord[prev] and t - prev <= beside)
                                 or (i + 1 < len(ordered) and not is_chord[ordered[i + 1]]
                                     and ordered[i + 1] - t <= beside)):
            flick += 1
    return rapid, stray, flick


def main():
    rows = []
    for pk in sorted(glob.glob(str(ROOT / "work" / "keyed_cache" / "*.pkl"))):
        c = pickle.loads(open(pk, "rb").read())
        tm = TempoMap(beat_times=c["beat_times"], pickup_beats=c["pickup"])
        res = tm.resolution
        ws = c["windows"] or []
        extra = prominence.keyed_rescue_events(c["events"], ws, c["stem_events"], tm)[0] if ws else []
        events = sorted(c["events"] + extra)
        gspans = [(int(w["beat0"] * res), int(w["beat1"] * res)) for w in ws if w.get("stem") == "guitar"]
        guitar_led = bool(gspans) and len(gspans) >= 0.25 * len(ws)
        x = T.expert_from_notes(events, tm, ornaments=False)
        shares = {"expert": share(x)}
        n, ch = shares["expert"]
        if guitar_led or ch / max(1, n) >= G.TRIGGER_CHORD_SHARE:
            voices = G.second_voices(c["stem_events"]["guitar"], tm)
            x, _ = G.chordify(x, voices, res, allowed_spans=gspans if guitar_led else None)
        shares["chordify"] = share(x)
        ev = T.merge_triple_evidence(T.triple_song_evidence(c["events"], tm),
                                     T.triple_song_evidence(c["stem_events"]["guitar"], tm), tm)
        if ev["qualifies"]:
            x, _ = T.promote_triple_runs(x, ev, tm)
        shares["triples"] = share(x)
        x = texture.consolidate_gallops(texture.narrow_wide_chords(texture.smooth_chord_shapes(x, res), res), res)
        shares["texture"] = share(x)
        pre = x
        chord_spans = gspans if guitar_led else None
        x = reduce.simplify_rapid_chords(x, res, tempo=tm, chord_spans=chord_spans)
        shares["simplify"] = share(x)
        x = playability.enforce(x, tm)
        shares["playability"] = share(x)
        x = texture.smooth_chord_shapes(texture.smooth_chord_shapes(x, res), res)
        shares["late smooth"] = share(x)
        rows.append((c["song"][:30], shares, human_share(c["song"]), cull_reasons(pre, res, chord_spans),
                     "G" if guitar_led else "-"))
    print(f"{'song':32s} G " + " ".join(f"{s:>10s}" for s in STAGES) + "   human   cull: rapid/stray/flicker")
    for name, sh, hs, (rapid, stray, flick), g in rows:
        print(f"{name:32s} {g} " + " ".join(f"{sh[s][1] / max(1, sh[s][0]):9.0%} " for s in STAGES)
              + f"  {hs if hs is not None else float('nan'):6.0%}   {rapid:3d}/{stray:3d}/{flick:3d}")
    fin = [r[1]["late smooth"][1] / max(1, r[1]["late smooth"][0]) for r in rows]
    hum = [r[2] for r in rows if r[2] is not None]
    gl = [(r[1]["late smooth"][1] / max(1, r[1]["late smooth"][0]), r[2]) for r in rows if r[4] == "G" and r[2] is not None]
    print(f"\nmedian final chord share {np.median(fin):.0%} vs human {np.median(hum):.0%}; "
          f"guitar-led songs {np.median([a for a, _ in gl]):.0%} vs human {np.median([b for _, b in gl]):.0%}; "
          f"median cull in the rapid-chord pass {np.median([r[1]['texture'][1] - r[1]['simplify'][1] for r in rows]):.0f} chords/song")
    return 0


if __name__ == "__main__":
    sys.exit(main())
