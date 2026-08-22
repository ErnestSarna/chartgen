"""Turn a folder of Clone Hero songs into an audio2chart training manifest.

audio2chart's own `dataloader/convert_to_raw.py` does this but shells out to
ffmpeg, which is not installed here. soundfile and librosa are already
dependencies and write the same headerless 16 kHz mono s16le PCM, so this needs
no new tooling.

    python tools/build_dataset.py --root "D:/CH Songs" --out data/

Input: any directory tree where a folder contains notes.chart plus audio.
That is exactly the shape Chorus Encore / Bridge downloads arrive in, so no API
or scraping is involved — download charts normally and point this at them.

Output: data/audio_dataset_with_raw.json + data/raw_audio/*.raw, which is what
`main.py` expects via `config.data.root_folder`.
"""
import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_NAMES = ("song", "guitar", "rhythm", "bass", "drums", "keys", "vocals")
# Stems that belong in a training mix. Multi-stem rips split the instruments
# across files and their `song.*` is only the backing track — training on it
# alone means the model never hears the guitar it is charting. crowd/preview
# are deliberately absent.
MIX_NAMES = AUDIO_NAMES + (
    "drums_1", "drums_2", "drums_3", "drums_4", "vocals_1", "vocals_2",
    "guitar_1", "guitar_2", "rhythm_1", "rhythm_2", "keys_1", "keys_2",
)
AUDIO_EXTS = (".ogg", ".mp3", ".opus", ".wav", ".flac")
# Must match config.model.sample_rate for the model being trained. The Encodec
# checkpoints (charter-v1.0-*) use 24 kHz; load_raw_audio() trusts whatever
# rate it is told, so a mismatch silently plays every song 1.5x fast.
RAW_RATE = 24000


def find_audio(folder: Path):
    """Prefer a full mix; fall back to any stem so a song is not skipped."""
    for name in AUDIO_NAMES:
        for ext in AUDIO_EXTS:
            candidate = folder / f"{name}{ext}"
            if candidate.is_file():
                return candidate
    return next((p for p in sorted(folder.iterdir())
                 if p.suffix.lower() in AUDIO_EXTS), None)


def find_mix_files(folder: Path) -> list[Path]:
    """Every stem that should sound in the training mix, in stable order."""
    files = [folder / f"{name}{ext}"
             for name in MIX_NAMES for ext in AUDIO_EXTS
             if (folder / f"{name}{ext}").is_file()]
    if files:
        return files
    single = find_audio(folder)
    return [single] if single else []


def chart_report(chart_path: Path):
    """Which tiers a chart actually contains, and its note counts."""
    try:
        text = chart_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return None, f"unreadable: {error}"
    if "[SyncTrack]" not in text:
        return None, "no SyncTrack"
    tiers = {}
    for tier in ("ExpertSingle", "HardSingle", "MediumSingle", "EasySingle"):
        block = re.search(rf"\[{tier}\]\s*\{{(.*?)\}}", text, re.S)
        if block:
            count = len(re.findall(r"= N \d+ \d+", block.group(1)))
            if count:
                tiers[tier] = count
    if not tiers:
        return None, "no guitar tracks with notes"
    return tiers, None


def to_raw(audio_paths: list[Path], dest: Path) -> int:
    """Mix the stems and write RAW_RATE mono s16le PCM; returns sample count."""
    import librosa
    import numpy as np

    mix = None
    for path in audio_paths:
        y, _ = librosa.load(str(path), sr=RAW_RATE, mono=True)
        if mix is None:
            mix = y
        else:
            if len(y) > len(mix):
                mix, y = y, mix
            mix[:len(y)] += y
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 1.0:  # summing stems clips; rescale, don't hard-clip
        mix = mix * (0.98 / peak)
    pcm = (np.clip(mix, -1.0, 1.0) * 32767).astype("<i2")
    dest.write_bytes(pcm.tobytes())
    return len(pcm)


def to_pitch(raw_path: Path, dest: Path, grid_ms: int) -> int:
    """Cache pitch conditioning features from the mixed raw PCM.

    Reads the .raw mix rather than the original audio so pitch describes the
    same signal the model hears — for multi-stem songs the original `song.*`
    is missing the instruments.

    Precomputed rather than extracted in the dataloader because CQT costs ~1.0s
    per 30s window — a batch of 4 is ~4s of CPU against a sub-second GPU step, so
    on-the-fly extraction starves the GPU no matter how many workers you throw at
    it. salience is stored float16: it is normalised to 0-1, so the precision is
    irrelevant and it halves what the dataloader has to read.
    """
    import numpy as np

    from chartgen.pitch import pitch_features

    y = np.frombuffer(raw_path.read_bytes(), dtype="<i2").astype(np.float32) / 32768.0
    salience, f0, voiced = pitch_features(y, RAW_RATE, grid_ms)
    np.savez_compressed(dest, salience=salience.astype(np.float16),
                        f0=f0.astype(np.float16), voiced=voiced)
    return len(f0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="folder of song folders")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--require", nargs="*", default=["ExpertSingle"],
                    help="skip songs missing these tracks")
    ap.add_argument("--scan-only", action="store_true",
                    help="report what is there without decoding any audio")
    ap.add_argument("--grid-ms", type=int, default=40,
                    help="must match the checkpoint: 40 for …-40-… , 20 for …-20-…")
    ap.add_argument("--no-pitch", action="store_true",
                    help="skip pitch features (only if not training the conditioned model)")
    ap.add_argument("--no-mid", action="store_true",
                    help="skip .mid conversion and use only real notes.chart songs")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel decode workers; 0 = cpu_count-1")
    args = ap.parse_args()

    if not args.root.is_dir():
        ap.error(f"not a directory: {args.root}")
    raw_dir = args.out / "raw_audio"
    pitch_dir = args.out / "pitch"

    # .mid-only songs get a converted .chart cached under the output folder —
    # the Clone Hero library itself is never written to.
    converted_dir = args.out / "converted"
    charts = [(p, p) for p in sorted(args.root.rglob("notes.chart"))]
    converted = 0
    if not args.no_mid:
        from tools.mid2chart import convert as mid_convert

        for mid_path in sorted(args.root.rglob("notes.mid")):
            folder = mid_path.parent
            if (folder / "notes.chart").is_file():
                continue
            dest = converted_dir / f"{folder.name}.chart"
            if not (dest.is_file() and dest.stat().st_size):
                try:
                    text = mid_convert(mid_path)
                except Exception as error:
                    charts.append((mid_path, None))
                    continue
                if text is None:
                    charts.append((mid_path, None))
                    continue
                converted_dir.mkdir(parents=True, exist_ok=True)
                dest.write_text(text, encoding="utf-8")
            charts.append((mid_path, dest))
            converted += 1

    found, skipped = [], []
    for source, chart_path in sorted(charts):
        folder = source.parent
        if chart_path is None:
            skipped.append({"path": str(folder), "reason": "mid conversion failed"})
            continue
        tiers, error = chart_report(chart_path)
        if error:
            skipped.append({"path": str(folder), "reason": error})
            continue
        missing = [t for t in args.require if t not in tiers]
        if missing:
            skipped.append({"path": str(folder), "reason": f"missing {','.join(missing)}"})
            continue
        stems = find_mix_files(folder)
        if not stems:
            skipped.append({"path": str(folder), "reason": "no audio file"})
            continue
        found.append({"chart_path": str(chart_path), "folder": str(folder),
                      "audio_path": str(stems[0]),
                      "stems": [str(s) for s in stems], "tiers": tiers})

    print(f"usable songs      : {len(found)} ({converted} via .mid conversion)")
    print(f"skipped           : {len(skipped)}")
    for reason in sorted({s["reason"] for s in skipped}):
        print(f"   {sum(1 for s in skipped if s['reason'] == reason):>5}  {reason}")

    if found:
        tier_totals = {}
        for entry in found:
            for tier, count in entry["tiers"].items():
                tier_totals.setdefault(tier, []).append(count)
        print("\ntier coverage:")
        for tier, counts in sorted(tier_totals.items()):
            print(f"   {tier:<14} {len(counts):>4} songs, median {sorted(counts)[len(counts)//2]:>5} notes")

    if args.scan_only or not found:
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_pitch:
        pitch_dir.mkdir(parents=True, exist_ok=True)

    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"\ndecoding with {workers} worker(s)"
          f"{'' if args.no_pitch else ' + pitch features'} …")

    def prepare(entry):
        """Runs in a worker: decode + mix stems and cache pitch features.

        Emits one manifest row per difficulty tier — the schema the vendor
        splitter and dataloader expect (each row carries a 'difficulty'
        section name; rows for one song share raw/pitch caches, and the
        splitter groups by raw_path so a song never straddles train/val).
        """
        folder = Path(entry["folder"])
        raw = raw_dir / f"{folder.name}.raw"
        length = (raw.stat().st_size // 2
                  if raw.is_file() and raw.stat().st_size
                  else to_raw([Path(s) for s in entry["stems"]], raw))
        row = {"chart_path": entry["chart_path"], "audio_path": entry["audio_path"],
               "raw_path": str(raw), "length_samples": length}
        if not args.no_pitch:
            pitch = pitch_dir / f"{folder.name}.npz"
            if not (pitch.is_file() and pitch.stat().st_size):
                to_pitch(raw, pitch, args.grid_ms)
            row["pitch_path"] = str(pitch)
            row["grid_ms"] = args.grid_ms
        return [dict(row, difficulty=tier) for tier in sorted(entry["tiers"])
                if tier in args.require]

    entries, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare, e): e for e in found}
        for done, future in enumerate(as_completed(futures), 1):
            entry = futures[future]
            try:
                entries.extend(future.result())
            except Exception as error:  # one bad file must not kill the run
                failures.append({"path": str(Path(entry["chart_path"]).parent),
                                 "error": repr(error)})
            if done % 10 == 0 or done == len(found):
                print(f"   {done}/{len(found)}")

    entries.sort(key=lambda row: (row["chart_path"], row["difficulty"]))
    manifest = args.out / "audio_dataset_with_raw.json"
    manifest.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"\nwrote {manifest} ({len(entries)} entries)")
    if failures:
        (args.out / "failed.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        print(f"{len(failures)} failed -> {args.out / 'failed.json'}")
    print(f"train with: python main.py data.root_folder={args.out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
