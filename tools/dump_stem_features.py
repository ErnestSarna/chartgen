"""Add Demucs stem evidence to the cached solo sections.

The solo caches (work/cal_solo_*.json) hold, per song, the chroma sections
with full-mix features and the human solo markers. The measured recall
ceiling came from full-mix blindness: a solo that does not stand out in
the mix is invisible to those features, and lyric-word counting is a poor
vocal detector. Stems fix both blind spots:

- singing: share of the section where the VOCAL stem is audibly active,
  a true it-is-sung signal per section;
- lead: the OTHER stem's energy (where lead guitar/synth lives) as a
  per-song z-score, so "the lead instrument surges here" is measurable
  even when the full mix stays flat.

Writes work/cal_stems_<tag>.json aligned 1:1 with the cached sections, so
sweep code can join by song name + section index.

    python tools/dump_stem_features.py work/cal_solo_A.json -o work/cal_stems_A.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_solos import find_audio


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cache", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--library", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data/calibration")
    args = ap.parse_args(argv)

    import numpy as np

    from chartgen import stems as stemsmod
    from chartgen import tempo as tempomod

    songs = json.loads(args.cache.read_text(encoding="utf-8"))
    folders = {p.name: p for p in args.library.iterdir() if p.is_dir()}
    out = []
    started = time.time()
    for i, song in enumerate(songs, 1):
        folder = folders.get(song["name"])
        audio = find_audio(folder) if folder else None
        if audio is None:
            print(f"  [{i}] {song['name'][:44]}: no audio", flush=True)
            continue
        try:
            mono = stemsmod.separate(str(audio))
            if mono is None:
                continue
            y, sr = tempomod.load(str(audio))
            tmap = tempomod.detect(y, sr, resolution=song["resolution"])
        except Exception as error:
            print(f"  [{i}] {song['name'][:44]}: {type(error).__name__}: {error}",
                  flush=True)
            continue

        res = song["resolution"]
        vocal = mono["vocals"]
        _, other_rms = stemsmod.activity(mono["other"])
        other_times = (np.arange(len(other_rms)) + 0.5) * 0.05
        o_mean = float(other_rms.mean())
        o_std = float(other_rms.std()) or 1e-9
        # Guitar-stem evidence (SW backend only): the "featured line vs
        # accompaniment" signal every previous solo sweep lacked. Recorded
        # here so ONE separation pass serves both the controlled backend
        # comparison and the guitar-aware re-sweep. Same frame grid as
        # other_rms, so the section windows line up.
        guitar_rms = None
        if "guitar" in mono:
            _, guitar_rms = stemsmod.activity(mono["guitar"])
            g_mean = float(guitar_rms.mean())
            g_std = float(guitar_rms.std()) or 1e-9
            canon = {k: stemsmod.activity(mono[k])[1]
                     for k in ("drums", "bass", "other", "vocals")}

        rows = []
        for section in song["sections"]:
            t0 = tmap.beat_to_time(section["start"] / res)
            t1 = tmap.beat_to_time(section["end"] / res)
            window = (other_times >= t0) & (other_times < t1)
            row = {
                "singing": stemsmod.singing_share(vocal, stemsmod.SR, t0, t1),
                "lead": (float(other_rms[window].mean()) - o_mean) / o_std
                        if window.any() else 0.0,
            }
            if guitar_rms is not None and window.any():
                n = min(len(window), len(guitar_rms))
                w = window[:n]
                total = sum(float(v[:n][w].mean()) for v in canon.values())
                row["guitar"] = (float(guitar_rms[:n][w].mean()) - g_mean) / g_std
                row["guitar_share"] = (float(guitar_rms[:n][w].mean()) / total
                                       if total > 0 else 0.0)
            rows.append(row)
        out.append({"name": song["name"], "stem_sections": rows})
        args.out.write_text(json.dumps(out), encoding="utf-8")
        print(f"  [{i}/{len(songs)}] {song['name'][:48]:<50} "
              f"{time.time() - started:.0f}s", flush=True)
    print(f"\nwrote {len(out)} songs to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
