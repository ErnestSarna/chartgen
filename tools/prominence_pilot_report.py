"""Hand-QA report for the Route B pilot (tools/prominence_pilot.py).

Prints each song's window timeline (which stem the charter followed),
the label distribution, the known-answer checks, and the Route A
question: on labeled windows, does the loudest stem by K-weighted
loudness name the followed stem more often than the loudest by RMS
energy? Also ranks single features by how well argmax-over-stems
recovers the label.

    python tools/prominence_pilot_report.py work/prominence_pilot.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

STEMS = ("vocals", "bass", "guitar", "piano", "sw_other")
LETTER = {"vocals": "V", "bass": "B", "guitar": "G", "piano": "P", "sw_other": "S",
          "ambiguous": "?", "none": "."}

rows = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
# Re-label from the stored per-stem scores so thresholds can be tuned
# offline: argv[2] = F1 floor, argv[3] = gap over the runner-up.
LABEL_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
LABEL_MARGIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
for r in rows:
    for w in r["windows"]:
        if w["label"] == "none":
            continue
        ranked = sorted(w["scores"].items(), key=lambda kv: -kv[1]["f1"])
        w["top"] = ranked[0][0]
        best, second = ranked[0][1]["f1"], ranked[1][1]["f1"]
        w["label"] = ranked[0][0] if best >= LABEL_MIN and best - second >= LABEL_MARGIN else "ambiguous"
print(f"{len(rows)} songs   (label rule: F1 >= {LABEL_MIN}, margin >= {LABEL_MARGIN})")
print()
total = Counter()
labeled = []
for r in rows:
    wins = r["windows"]
    tl = "".join(LETTER[w["label"]] for w in wins)
    c = Counter(w["label"] for w in wins)
    total.update(c)
    scored = [w for w in wins if w["label"] != "none"]
    amb = c["ambiguous"] / max(1, len(scored))
    print(f"{r['song'][:44]:46s} offset {r['offset']:+.2f}s  mix-match {r['mix_matches']}/{r['human_onsets']}  "
          f"ambiguous {amb:.0%}\n    {tl}")
    for w in scored:
        if w["label"] != "ambiguous" and w.get("features"):
            labeled.append((r["song"], w))

print("\n=== label distribution (all windows) ===")
n = sum(total.values())
for k, v in total.most_common():
    print(f"  {k:10s} {v:4d} ({v / n:.0%})")
scored_n = n - total["none"]
print(f"  labeled share of scored windows: {(scored_n - total['ambiguous']) / max(1, scored_n):.0%}")

print("\n=== known-answer checks ===")
def span_labels(song_prefix, t0, t1):
    for r in rows:
        if r["song"].startswith(song_prefix):
            return [w["label"] for w in r["windows"] if w["t1"] > t0 and w["t0"] < t1]
    return None
for name, t0, t1, expect in [("Perturbator", 238, 258, "guitar (charter strummed it)"),
                             ("Perturbator", 258, 298, "sw_other (charter tapped it)"),
                             ("Tom Petty", 0, 400, "guitar"),
                             ("Jessica Curry", 0, 400, "piano"),
                             ("Thaehan", 0, 400, "sw_other"),
                             ("Getter", 82, 187, "sw_other (the drop)")]:
    labs = span_labels(name, t0, t1)
    if labs:
        tops = [w["top"] for r in rows if r["song"].startswith(name)
                for w in r["windows"] if w["t1"] > t0 and w["t0"] < t1 and w["label"] != "none"]
        print(f"  {name:12s} {t0:3d}-{t1:3d}s expect {expect:32s} labels {Counter(labs).most_common(2)}  "
              f"top-1 regardless {Counter(tops).most_common(2)}")

print()
print("=== label-rule sweep: share of scored windows that get a label ===")
scored_all = [w for r in rows for w in r["windows"] if w["label"] != "none"]
for mn in (0.25, 0.30, 0.35):
    line = []
    for mg in (0.05, 0.08, 0.10, 0.15):
        k = 0
        for w in scored_all:
            ranked = sorted(w["scores"].values(), key=lambda v: -v["f1"])
            k += ranked[0]["f1"] >= mn and ranked[0]["f1"] - ranked[1]["f1"] >= mg
        line.append(f"min{mn:.2f}/gap{mg:.2f}: {k / len(scored_all):4.0%}")
    print("  " + "   ".join(line))

print("\n=== Route A: which stem-ranking recovers the label? (labeled windows) ===")
def acc(key):
    ok = 0
    for _, w in labeled:
        f = w["features"]
        best = max(STEMS, key=lambda s: f[s][key])
        ok += best == w["label"]
    return ok / max(1, len(labeled))
for key in ("energy", "loudness", "density", "mono"):
    print(f"  argmax {key:9s} == label: {acc(key):.0%}   (n={len(labeled)})")
# vocals-excluded variant: charters rarely chart the vocal line
def acc_novoc(key):
    ok = 0
    for _, w in labeled:
        f = w["features"]
        best = max((s for s in STEMS if s != "vocals"), key=lambda s: f[s][key])
        ok += best == w["label"]
    return ok / max(1, len(labeled))
for key in ("energy", "loudness"):
    print(f"  argmax {key:9s} (no vocals) == label: {acc_novoc(key):.0%}")

print("\n=== per-label feature medians (energy share / loudness share / density) of the FOLLOWED stem ===")
import statistics
for lab in STEMS:
    ws = [w for _, w in labeled if w["label"] == lab]
    if not ws:
        continue
    e = statistics.median(w["features"][lab]["energy"] for w in ws)
    l = statistics.median(w["features"][lab]["loudness"] for w in ws)
    d = statistics.median(w["features"][lab]["density"] for w in ws)
    print(f"  {lab:9s} n={len(ws):3d}  energy {e:.2f}  loudness {l:.2f}  density {d:.1f}/s")
