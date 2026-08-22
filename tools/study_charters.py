"""Study how human charters map the song's audio onto frets.

For every song in the dataset manifest (human chart + cached pitch features of
the mixed audio), sample the pitch features at each Expert note onset and ask:

    contour     when the audio's pitch moves up/down between two single notes,
                how often does the human's fret move the same direction?
    repeat|same when the pitch stays put, how often does the fret stay put?
    repeat|diff when the pitch moves, how often does the fret stay anyway?
    corr        corr(f0, fret) per song

This is the empirical justification (or refutation) of `frets_from_pitch`: if
humans track pitch contour strongly, imitating pitch is the right default; the
conditional-repeat gap shows how much of the mapping is pitch vs. phrasing.

    python tools/study_charters.py --manifest data/audio_dataset_with_raw.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402


def _stem_pitch(entry, stem: str, grid_ms: int):
    """Pitch features from an isolated stem, cached beside the mix features.

    The mix's dominant pitch at a note onset is usually the vocal; the charter
    follows the guitar. Only songs that ship the stem participate.
    """
    import librosa

    from chartgen.pitch import pitch_features

    folder = Path(entry["audio_path"]).parent
    stem_file = next((folder / f"{stem}{ext}" for ext in
                      (".opus", ".ogg", ".mp3", ".wav", ".flac")
                      if (folder / f"{stem}{ext}").is_file()), None)
    if stem_file is None:
        return None
    cache = Path(entry["pitch_path"]).parent.parent / f"pitch_{stem}" / (
        Path(entry["pitch_path"]).name)
    if cache.is_file() and cache.stat().st_size:
        data = np.load(cache)
        return data["f0"].astype(np.float32), data["voiced"].astype(bool)
    y, sr = librosa.load(str(stem_file), mono=True)
    _, f0, voiced = pitch_features(y, sr, grid_ms)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, f0=f0.astype(np.float16), voiced=voiced)
    return f0.astype(np.float32), voiced


def study_song(entry, stem: str | None = None) -> dict | None:
    from tools.evaluate import chart_note_times

    grid_ms = int(entry.get("grid_ms", 40))
    if stem:
        stem_data = _stem_pitch(entry, stem, grid_ms)
        if stem_data is None:
            return None
        f0, voiced = stem_data
    else:
        data = np.load(entry["pitch_path"])
        f0 = data["f0"].astype(np.float32)
        voiced = data["voiced"].astype(bool)

    notes = chart_note_times(entry["chart_path"])
    # Singles only: chord roots follow harmony rules, a different question.
    singles = [(t, lanes[0]) for t, lanes in notes if len(lanes) == 1 and lanes[0] != 7]
    if len(singles) < 50:
        return None

    last = len(f0) - 1
    rows = []
    for t, lane in singles:
        i = min(last, int(round(t * 1000.0 / grid_ms)) + 1)
        if voiced[i]:
            rows.append((f0[i], lane))
    if len(rows) < 50:
        return None

    pitches = np.array([p for p, _ in rows])
    frets = np.array([l for _, l in rows], dtype=float)

    same_tol = 1.0 / 47.0  # one semitone in f0_norm units (48 bins)
    moves = same = same_kept = diff = diff_kept = agree = 0
    for (p0, l0), (p1, l1) in zip(rows, rows[1:]):
        dp, dl = p1 - p0, l1 - l0
        if abs(dp) <= same_tol:
            same += 1
            same_kept += dl == 0
        else:
            diff += 1
            diff_kept += dl == 0
            if dl:
                moves += 1
                agree += (dp > 0) == (dl > 0)
    corr = float(np.corrcoef(pitches, frets)[0, 1]) if len(set(frets)) > 1 else np.nan
    return {
        # audio_path names the real song folder even for converted .mid charts
        "name": Path(entry["audio_path"]).parent.name[:40],
        "n": len(rows),
        "corr": corr,
        "contour": agree / moves if moves else np.nan,
        "repeat_same_pitch": same_kept / same if same else np.nan,
        "repeat_diff_pitch": diff_kept / diff if diff else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("data/audio_dataset_with_raw.json"))
    ap.add_argument("--per-song", action="store_true", help="print every song")
    ap.add_argument("--stem", default=None, metavar="NAME",
                    help="use an isolated stem (e.g. guitar) instead of the "
                         "mix; songs without that stem are skipped")
    args = ap.parse_args()

    entries = json.loads(args.manifest.read_text(encoding="utf-8"))
    songs = {}
    for e in entries:
        if e.get("difficulty", "ExpertSingle") == "ExpertSingle" and "pitch_path" in e:
            songs.setdefault(e["chart_path"], e)

    results = []
    for e in songs.values():
        try:
            row = study_song(e, stem=args.stem)
        except Exception as error:
            print(f"skip {Path(e['chart_path']).parent.name[:40]}: "
                  f"{type(error).__name__}: {error}")
            continue
        if row:
            results.append(row)
            if args.per_song:
                print(f"{row['name']:<42} corr {row['corr']:+.2f}  "
                      f"contour {row['contour']:.0%}  "
                      f"rep|same {row['repeat_same_pitch']:.0%}  "
                      f"rep|diff {row['repeat_diff_pitch']:.0%}")

    if not results:
        print("no songs usable (is the pitch cache built?)")
        return 1

    def med(key):
        vals = sorted(r[key] for r in results if not np.isnan(r[key]))
        return vals[len(vals) // 2] if vals else float("nan")

    print(f"\n{len(results)} songs analysed — how HUMAN charters track the audio:")
    print(f"  corr(pitch, fret)            median {med('corr'):+.2f}")
    print(f"  contour agreement            median {med('contour'):.0%}   "
          f"(fret moves the same direction as the pitch)")
    print(f"  fret repeat | same pitch     median {med('repeat_same_pitch'):.0%}")
    print(f"  fret repeat | pitch moved    median {med('repeat_diff_pitch'):.0%}")
    print("\nReading: high contour + a large gap between the two repeat rates "
          "means charters really do map pitch to frets, and frets_from_pitch "
          "is imitating the right thing. Low numbers mean phrasing dominates "
          "and the pitch rule needs softening.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
