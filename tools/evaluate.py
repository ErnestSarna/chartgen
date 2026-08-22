"""Score generated charts against the human charts on held-out songs.

    python tools/evaluate.py --val data/val.json --outdir out/eval-pitch
    python tools/evaluate.py --val data/val.json --fret-mode model --outdir out/eval-model
    python tools/evaluate.py --val data/val.json --model runs/pitch-m/export \
        --conditioner runs/pitch-m/export/conditioner.pt --outdir out/eval-ft

Uses the vendor's ChartProcessor + tokenizer for BOTH charts, so human and
generated notes go through identical tick->seconds conversion. Metrics per
song, then medians:

    onset F1   how well our note times match the human chart's (tolerance ms)
    fret agree of the onsets that matched, how often the lane is the same
               (singles only; chords compared as root lane)
    contour    Spearman-ish sign agreement of fret motion on matched onsets —
               "did we move the same direction the charter did", which is what
               following the music feels like even when absolute lanes differ

The human chart is a stylistic reference, not ground truth — two humans chart
the same song differently. Trends across 13 songs matter; single numbers don't.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402  (vendor trees on sys.path)


def chart_note_times(chart_path: str, section: str = "ExpertSingle"):
    """[(seconds, lanes tuple)] for a chart file, via the vendor pipeline."""
    from chart.chart_processor import ChartProcessor
    from chart.tokenizer import SimpleTokenizerGuitar

    tokenizer = SimpleTokenizerGuitar()
    processor = ChartProcessor([section.replace("Single", "")], ["Single"])
    processor.read_chart(chart_path=chart_path, target_sections=section)
    notes = processor.notes[section]
    tokenized = tokenizer.encode(note_list=notes)
    timed = tokenizer.format_seconds(
        tokenized, processor.synctrack,
        resolution=int(processor.song_metadata["Resolution"]),
        offset=float(processor.song_metadata.get("Offset", 0) or 0),
    )
    out = []
    for t, token, _dur, *_ in timed:
        lanes = tokenizer.reverse_map.get(token)
        if lanes:
            out.append((float(t), tuple(sorted(lanes))))
    return out


def match_onsets(ours, theirs, tol_s: float):
    """Greedy nearest-neighbour matching inside the tolerance."""
    matches, used = [], set()
    j = 0
    for i, (t, _) in enumerate(ours):
        best, best_d = None, tol_s
        for k in range(max(0, j - 3), len(theirs)):
            d = abs(theirs[k][0] - t)
            if theirs[k][0] - t > tol_s:
                break
            if k not in used and d <= best_d:
                best, best_d = k, d
        if best is not None:
            used.add(best)
            matches.append((i, best))
            j = best
    return matches


def root(lanes):
    fretted = [l for l in lanes if l != 7]
    return min(fretted) if fretted else 7


def evaluate_song(gen_chart: Path, human_chart: Path, tol_s: float):
    ours = chart_note_times(str(gen_chart))
    theirs = chart_note_times(str(human_chart))
    if not ours or not theirs:
        return None
    matches = match_onsets(ours, theirs, tol_s)
    precision = len(matches) / len(ours)
    recall = len(matches) / len(theirs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    agree = moves = same_dir = 0
    prev = None
    for i, k in matches:
        a, b = root(ours[i][1]), root(theirs[k][1])
        if a == b:
            agree += 1
        if prev is not None:
            pa, pb = prev
            da, db = a - pa, b - pb
            if da or db:
                moves += 1
                if (da > 0) == (db > 0) and (da < 0) == (db < 0):
                    same_dir += 1
        prev = (a, b)
    return {
        "gen_notes": len(ours), "human_notes": len(theirs),
        "precision": precision, "recall": recall, "f1": f1,
        "fret_agree": agree / len(matches) if matches else 0.0,
        "contour": same_dir / moves if moves else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", type=Path, default=Path("data/val.json"))
    ap.add_argument("--outdir", type=Path, default=Path("out/eval"))
    ap.add_argument("--model", default="3podi/charter-v1.0-40-M-best-acc")
    ap.add_argument("--conditioner", default=None)
    ap.add_argument("--fret-mode", choices=("pitch", "model"), default="pitch")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--tolerance-ms", type=float, default=70.0)
    ap.add_argument("--limit", type=int, default=0, help="cap song count (0 = all)")
    ap.add_argument("--skip-generate", action="store_true",
                    help="re-score existing charts in --outdir")
    args = ap.parse_args()

    entries = json.loads(args.val.read_text(encoding="utf-8"))
    # One row per song (val.json has one row per tier).
    songs = {}
    for e in entries:
        songs.setdefault(e["chart_path"], e)
    rows = list(songs.values())
    if args.limit:
        rows = rows[:args.limit]

    from chartgen import pipeline

    results = []
    for e in rows:
        folder = Path(e["chart_path"]).parent
        name = folder.name
        gen_dir = args.outdir / f"eval - {name}"[:80]
        gen_chart = gen_dir / "notes.chart"
        if not args.skip_generate or not gen_chart.is_file():
            opts = SimpleNamespace(
                audio=Path(e["audio_path"]), outdir=args.outdir,
                model=args.model, conditioner=args.conditioner,
                subdiv=4, temperature=0.5, top_k=32, attempts=2,
                min_variety=0.80, min_sustain_gap=1.0, hopos=True,
                no_star_power=True, no_sections=True, no_sustains=True,
                # density pinned to "model" so every eval config measures the
                # same thing regardless of the CLI default changing.
                fret_mode=args.fret_mode, seed=args.seed, reducer="chartgen",
                density="model",
                bpm_mult="auto", name=f"eval - {name}"[:76], artist="",
                album="", genre="", year="",
            )
            try:
                result = pipeline.run(opts, progress=lambda m: None)
            except Exception as error:
                print(f"SKIP {name}: {type(error).__name__}: {error}")
                continue
            gen_chart = Path(result["song_dir"]) / "notes.chart"
        try:
            row = evaluate_song(gen_chart, Path(e["chart_path"]),
                                args.tolerance_ms / 1000.0)
        except Exception as error:
            print(f"SKIP {name} (parse): {type(error).__name__}: {error}")
            continue
        if row:
            row["name"] = name[:40]
            results.append(row)
            print(f"{row['name']:<42} F1 {row['f1']:.2f}  "
                  f"fret {row['fret_agree']:.2f}  contour {row['contour']:.2f}  "
                  f"({row['gen_notes']}/{row['human_notes']} notes)")

    if not results:
        print("no results")
        return 1

    def med(key):
        vals = sorted(r[key] for r in results)
        return vals[len(vals) // 2]

    print(f"\n{len(results)} songs — medians: "
          f"F1 {med('f1'):.2f}  precision {med('precision'):.2f}  "
          f"recall {med('recall'):.2f}  fret-agree {med('fret_agree'):.2f}  "
          f"contour {med('contour'):.2f}")
    (args.outdir / "results.json").parent.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
