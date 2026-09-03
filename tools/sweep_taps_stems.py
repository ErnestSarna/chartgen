"""Does a piano/synth stem predict what charters tap? Sweep on the six-stem
dump (tools/dump_taps_stems.py).

Same truth as the recal sweep: a section whose human chart taps >= 50% of
its notes is a positive, <= 5% a negative, the middle is left out. The
baseline to beat is the shim rule (guitar+piano+residual share, the 42%
precision wall). Each candidate feature is scored alone (median gap,
AUC, best F0.5, precision-first), then the best pairs, and the winners
are checked split-half BY SONG so a rule tuned on one half must hold on
the other.

    python tools/sweep_taps_stems.py work/taps_stems6.json
"""
import argparse
import itertools
import json
import random
import statistics
import sys
from pathlib import Path

POSITIVE_TAP_SHARE = 0.50
NEGATIVE_TAP_SHARE = 0.05
MIN_NOTES = 4

FEATURES = {
    "shim_other": lambda s: s["share"]["guitar"] + s["share"]["piano"] + s["share"]["sw_other"],
    "piano": lambda s: s["share"]["piano"],
    "synth": lambda s: s["share"]["sw_other"],
    "keys+synth": lambda s: s["share"]["piano"] + s["share"]["sw_other"],
    "guitar": lambda s: s["share"]["guitar"],
    "vocals": lambda s: s["share"]["vocals"],
    "bass": lambda s: s["share"]["bass"],
    "drums": lambda s: s["share"]["drums"],
    "keys+synth-guitar": lambda s: s["share"]["piano"] + s["share"]["sw_other"] - s["share"]["guitar"],
    "melodic_max": lambda s: max(s["share"]["piano"], s["share"]["sw_other"], s["share"]["guitar"]),
    "piano_cen_khz": lambda s: s["cen"]["piano"] / 1000.0,
    "synth_cen_khz": lambda s: s["cen"]["sw_other"] / 1000.0,
}


def load(path):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    secs = []
    for r in rows:
        for s in r["sections"]:
            if s["notes"] < MIN_NOTES:
                continue
            ts = s["taps"] / s["notes"]
            if ts >= POSITIVE_TAP_SHARE:
                label = True
            elif ts <= NEGATIVE_TAP_SHARE:
                label = False
            else:
                continue
            secs.append({"song": r["song"], "label": label,
                         **{k: f(s) for k, f in FEATURES.items()}})
    return rows, secs


def auc(pos, neg):
    """Probability a random positive outranks a random negative."""
    allv = sorted(set(pos + neg))
    rank = {v: i for i, v in enumerate(allv)}
    p = sorted(rank[v] for v in pos)
    wins = 0.0
    import bisect
    for v in neg:
        r = rank[v]
        wins += bisect.bisect_left(p, r) + 0.5 * (bisect.bisect_right(p, r) - bisect.bisect_left(p, r))
    # `wins` counts positives ranked BELOW each negative; AUC is the complement
    return 1.0 - wins / (len(pos) * len(neg))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = (1.25 * p * r / (0.25 * p + r)) if p + r else 0.0
    return p, r, f


def score_rule(secs, rule):
    tp = fp = fn = 0
    for s in secs:
        fired = rule(s)
        if fired and s["label"]:
            tp += 1
        elif fired:
            fp += 1
        elif s["label"]:
            fn += 1
    return (*prf(tp, fp, fn), tp, fp, fn)


def thresholds(values, n=25):
    qs = sorted(set(statistics.quantiles(values, n=n)))
    return qs


def sweep_single(secs, feat, direction):
    best = []
    vals = [s[feat] for s in secs]
    for th in thresholds(vals):
        rule = (lambda s, th=th: s[feat] >= th) if direction == ">=" else (lambda s, th=th: s[feat] <= th)
        p, r, f, tp, fp, fn = score_rule(secs, rule)
        best.append({"feat": feat, "dir": direction, "th": th, "p": p, "r": r, "f": f, "tp": tp, "fp": fp})
    return best


def report_rules(label, grid, strict=0.60):
    grid = sorted(grid, key=lambda g: -g["f"])
    print(f"   {label} best F0.5: " + "; ".join(
        f"{g['feat']}{g['dir']}{g['th']:.2f} P{g['p']:.0%} R{g['r']:.0%} (tp{g['tp']} fp{g['fp']})"
        for g in grid[:3]))
    ok = [g for g in grid if g["p"] >= strict]
    if ok:
        b = max(ok, key=lambda g: g["r"])
        print(f"   {label} precision-first (>={strict:.0%}): {b['feat']}{b['dir']}{b['th']:.2f} "
              f"P{b['p']:.0%} R{b['r']:.0%} (tp{b['tp']} fp{b['fp']})")
    else:
        print(f"   {label} no rule reaches {strict:.0%} precision")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)
    rows, secs = load(args.dump)
    pos = [s for s in secs if s["label"]]
    neg = [s for s in secs if not s["label"]]
    print(f"{len(rows)} songs ({sum(1 for r in rows if r['cls'] == 'pos')} tap, "
          f"{sum(1 for r in rows if r['cls'] == 'neg')} control); {len(secs)} scored sections: "
          f"{len(pos)} tapped-whole, {len(neg)} untapped")
    if len(pos) < 20 or len(neg) < 20:
        print("not enough sections yet")
        return 0

    print("\n=== single features: median tapped vs untapped, AUC (0.5 = no signal) ===")
    single = {}
    for feat in FEATURES:
        pv = [s[feat] for s in pos]
        nv = [s[feat] for s in neg]
        a = auc(pv, nv)
        single[feat] = a
        print(f"   {feat:20s} tapped {statistics.median(pv):.3f}  untapped {statistics.median(nv):.3f}  "
              f"AUC {a:.3f}")

    print("\n=== single-feature rules (baseline = shim_other, the recal's 42% wall) ===")
    grids = {}
    for feat in FEATURES:
        direction = ">=" if single[feat] >= 0.5 else "<="
        grids[feat] = sweep_single(secs, feat, direction)
    for feat in ("shim_other", "piano", "synth", "keys+synth", "keys+synth-guitar", "melodic_max"):
        report_rules(feat, grids[feat])

    print("\n=== pairs: keys/synth share AND a second gate ===")
    pair_grid = []
    primaries = ("piano", "synth", "keys+synth", "keys+synth-guitar")
    gates = ("guitar", "vocals", "drums", "piano_cen_khz", "synth_cen_khz")
    for a_feat, g_feat in itertools.product(primaries, gates):
        a_th = thresholds([s[a_feat] for s in secs], n=12)
        g_th = thresholds([s[g_feat] for s in secs], n=8)
        for ta, tg in itertools.product(a_th, g_th):
            rule = lambda s, a_feat=a_feat, g_feat=g_feat, ta=ta, tg=tg: s[a_feat] >= ta and s[g_feat] <= tg
            p, r, f, tp, fp, fn = score_rule(secs, rule)
            pair_grid.append({"feat": f"{a_feat}>={ta:.2f}&{g_feat}", "dir": "<=", "th": tg,
                              "p": p, "r": r, "f": f, "tp": tp, "fp": fp,
                              "_rule": (a_feat, ta, g_feat, tg)})
    report_rules("pairs", pair_grid)

    print("\n=== split-half by song (rule picked on half A, scored on half B, and vice versa) ===")
    songs = sorted({s["song"] for s in secs})
    rng = random.Random(args.seed)
    for seed_i in range(3):
        rng.shuffle(songs)
        half = set(songs[: len(songs) // 2])
        A = [s for s in secs if s["song"] in half]
        B = [s for s in secs if s["song"] not in half]
        for name, pick_from, score_on in (("A->B", A, B), ("B->A", B, A)):
            cands = []
            for feat in ("shim_other", "piano", "synth", "keys+synth", "keys+synth-guitar"):
                cands += sweep_single(pick_from, feat, ">=")
            best = max(cands, key=lambda g: g["f"])
            p, r, f, tp, fp, fn = score_rule(score_on, lambda s, b=best: s[b["feat"]] >= b["th"])
            base = max(sweep_single(pick_from, "shim_other", ">="), key=lambda g: g["f"])
            bp, br, bf, *_ = score_rule(score_on, lambda s, b=base: s[b["feat"]] >= b["th"])
            print(f"   seed{seed_i} {name}: picked {best['feat']}>={best['th']:.2f} -> held-out "
                  f"P{p:.0%} R{r:.0%} | shim baseline held-out P{bp:.0%} R{br:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
