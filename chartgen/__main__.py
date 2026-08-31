"""audio file -> Clone Hero song folder with all four difficulty tiers.

Pipeline: detect beat grid -> audio2chart generates Expert onsets -> quantize onto
the grid -> derive Hard/Medium/Easy -> add sustains, star power and sections ->
write notes.chart + song.ini + playable audio.

For the GUI, run `python -m chartgen.app`.
"""
import argparse
import sys
from pathlib import Path

from . import pipeline


def build_args(argv):
    ap = argparse.ArgumentParser(prog="chartgen", description=__doc__)
    # str, not Path: a YouTube URL must survive untouched (Path collapses //).
    ap.add_argument("audio", type=str, nargs="+",
                    help="one or more songs (mp3/flac/wav/ogg, >= 30s) and/or "
                         "YouTube links; each becomes its own song folder. "
                         "With several inputs, artist/title come from tags or "
                         "video titles per song")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("out"))
    ap.add_argument("--engine", choices=("basicpitch", "audio2chart"),
                    default="basicpitch",
                    help="Expert-note source. basicpitch (default): Apache-"
                         "licensed transcription — deterministic and "
                         "consistent. audio2chart: the neural charter, which "
                         "samples, so it lands bigger wins AND bigger losses; "
                         "--attempts rolls it several times and keeps the roll "
                         "that best matches the audio")
    ap.add_argument("--min-fidelity", type=float, default=0.60, metavar="F1",
                    help="stop rolling the neural engine once a roll matches "
                         "the audio this well (human charts median 0.57)")
    ap.add_argument("--model", default="3podi/charter-v1.0-40-M-best-acc",
                    help="audio2chart checkpoint: hub repo id or a local "
                         "export folder from tools/finetune.py "
                         "(…-S-… is ~9x smaller/faster)")
    ap.add_argument("--conditioner", default=None, metavar="PATH",
                    help="trained PitchConditioner weights (conditioner.pt "
                         "from tools/finetune.py); makes the model itself "
                         "pitch-aware during generation")
    ap.add_argument("--subdiv", type=int, default=4,
                    help="quantize grid: 4 = 16th notes, 2 = 8ths, 3 = triplet 8ths")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--fret-mode", choices=("pitch", "model"), default="pitch",
                    help="pitch: keep the model's timing but assign frets from "
                         "the audio's pitch (repeats repeat, rising lines rise). "
                         "model: raw model lanes, which are pitch-blind")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed sampling so runs are reproducible; each retry "
                         "attempt advances the seed by one")
    ap.add_argument("--target-diff", type=int, choices=range(0, 7), default=None,
                    metavar="0-6",
                    help="thin the chart until its predicted tier drops to "
                         "this; the whole difficulty ladder scales with it. "
                         "Only thins — notes the audio doesn't support are "
                         "never invented to raise a rating")
    ap.add_argument("--max-fret-jump", type=int, choices=(1, 2, 3, 4), default=2,
                    metavar="LANES",
                    help="cap lane movement between consecutive positions "
                         "(human charts almost never jump 3+); 4 disables")
    ap.add_argument("--min-sustain-beats", type=float, default=0.5,
                    metavar="BEATS",
                    help="transcribed note length needed to become a sustain "
                         "on the basicpitch engine")
    ap.add_argument("--density", choices=("onset", "model"), default="onset",
                    help="onset: drop notes lacking onset evidence in the "
                         "audio (the model over-generates ~1.4-2x vs human "
                         "charts). model: keep every generated note")
    ap.add_argument("--bpm-mult", choices=("auto", "0.5", "1", "2"), default="auto",
                    help="tempo octave: auto picks the conventional 70-165 BPM "
                         "range (the label decides HOPO vs strum feel); 1 keeps "
                         "the raw detection; 0.5/2 force half/double time")
    ap.add_argument("--reducer", choices=("chartgen", "easygen"), default="chartgen",
                    help="chartgen: HOPO-preserving reduction with order-"
                         "preserving lane remaps. easygen: the vendored "
                         "reducer, kept for comparison")
    ap.add_argument("--attempts", type=int, default=3, metavar="N",
                    help="regenerate up to N times if quality is poor; stops as "
                         "soon as a candidate passes, so a good first roll is free")
    ap.add_argument("--min-variety", type=float, default=0.80, metavar="SCORE",
                    help="0-1 quality bar a chart must clear (see quality.py)")
    ap.add_argument("--min-sustain-gap", type=float, default=0.75, metavar="BEATS",
                    help="beats of space before a note sustains; raise for fewer "
                         "sustains (needs playtesting to calibrate)")
    ap.add_argument("--cookies-from", default=None, metavar="BROWSER",
                    help="read YouTube cookies from this browser (firefox/"
                         "chrome/edge/…) so downloads carry your session — "
                         "the remedy for persistent 403 bot-detection blocks")
    ap.add_argument("--lyrics", action=argparse.BooleanOptionalAction, default=True,
                    help="transcribe vocals (faster-whisper) into CH lyric "
                         "events so words scroll at the top during play; "
                         "instrumentals detect as such and skip cleanly")
    ap.add_argument("--lyric-source", choices=("auto", "online", "transcribe"),
                    default="auto",
                    help="auto (default): look the song up on LRCLIB for real "
                         "hand-synced lyrics and transcribe only if it has "
                         "none. online: LRCLIB or nothing. transcribe: always "
                         "use faster-whisper, which invents words over "
                         "instrumental outros")
    ap.add_argument("--hopos", action=argparse.BooleanOptionalAction, default=True,
                    help="allow natural HOPOs (default on). They were off while "
                         "fret choice was a +/-1 walk — every 16th HOPOed — but "
                         "pitch-based frets repeat notes, and same-fret pairs "
                         "never HOPO, so the saturation problem is gone")
    ap.add_argument("--opens", action=argparse.BooleanOptionalAction, default=True,
                    help="open notes (the no-fret purple strum) for lone "
                         "notes clearly below the melody - bass drops, "
                         "chugs, pedal tones. 60%% of human charts use "
                         "them; default on")
    ap.add_argument("--taps", action=argparse.BooleanOptionalAction, default=False,
                    help="EXPERIMENTAL: mark soft phrases (piano lines, "
                         "plucks, gentle synth runs) as tap notes, which "
                         "play without strumming. Off by default: no "
                         "charting standard defines tap usage, and measured "
                         "against human charts the community itself only "
                         "half-agrees (taps lean darker-timbred 2:1, but "
                         "attack softness is a coin flip)")
    ap.add_argument("--no-solos", action="store_true",
                    help="skip solo markers (detected as busy instrumental "
                         "breaks in songs that otherwise have vocals)")
    ap.add_argument("--no-section-reuse", action="store_true",
                    help="chart every section independently; by default a "
                         "repeated chorus reuses the first one's lane choices "
                         "(human charts overlap 55%% between repeats, ours 13%%)")
    ap.add_argument("--keep-dense-chords", action="store_true",
                    help="keep every chord the audio suggests; by default "
                         "chords are capped at 2 notes and a chord following "
                         "a different chord within an 8th drops to its root")
    ap.add_argument("--no-playability", action="store_true",
                    help="skip the human-hand simulation that simplifies or "
                         "drops passages nobody could physically play")
    ap.add_argument("--no-riff-unify", action="store_true",
                    help="skip stamping repeated-sounding bars with one "
                         "consensus riff pattern (the median human chart "
                         "repeats a bar 8x; ungated generation repeats none)")
    ap.add_argument("--no-motifs", action="store_true",
                    help="skip the flow-motif layer: sustain stairs, wrapped "
                         "rolls, chord ladders, pickup roots, legato HOPO "
                         "forcing, and machine-gun/stray-push consolidation")
    ap.add_argument("--swing", action="store_true",
                    help="quantize beats whose onsets fit the triplet grid to "
                         "24ths (shuffle feel; Basic Pitch engine only). "
                         "Off by default: measured on real songs, beat-grid "
                         "phase error exceeds the 16th/triplet slot distance, "
                         "so detection misfires on straight songs")
    ap.add_argument("--max-chord", type=int, choices=(2, 3), default=2,
                    help="max simultaneous notes admitted per position at "
                         "transcription (default 2; 3 lets three-note "
                         "voicings through where three comparable-amplitude "
                         "pitches coincide - experimental)")
    ap.add_argument("--no-brightness-lanes", action="store_true",
                    help="never re-lane stuck stretches from the spectral "
                         "contour; by default a long single-note run stuck "
                         "on <=2 lanes whose audio shows a real filter sweep "
                         "takes its lanes from brightness (growl wobbles)")
    ap.add_argument("--no-bass-fallback", action="store_true",
                    help="never chart the bassline; by default, stretches "
                         "where the melodic selection leaves <=1 note per bar "
                         "for 2+ bars admit the bass register octave-lifted "
                         "(EDM breakdowns where the growl IS the foreground)")
    ap.add_argument("--no-ornaments", action="store_true",
                    help="skip recovering 32nd grace notes the 16th grid "
                         "swallowed (tightly gated: max one per bar, twelve "
                         "per song; Basic Pitch engine only)")
    ap.add_argument("--no-star-power", action="store_true")
    ap.add_argument("--no-sections", action="store_true")
    ap.add_argument("--no-sustains", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--artist", default="Unknown")
    ap.add_argument("--album", default="")
    ap.add_argument("--genre", default="")
    ap.add_argument("--year", default="")
    return ap.parse_args(argv), ap


def main(argv=None):
    import copy

    from . import youtube

    args, ap = build_args(argv)
    for item in args.audio:
        if not youtube.is_youtube_url(item) and not Path(item).is_file():
            ap.error(f"no such file: {item}")

    try:
        inputs = youtube.expand_inputs(args.audio, print)
    except ValueError as error:
        sys.exit(str(error))
    def chart_one(item, position, total):
        if total > 1:
            print(f"\n=== [{position}/{total}] {item} ===")
        opts = copy.copy(args)
        opts.audio = item
        if total > 1:
            # Per-song metadata must come from each file/video, not be shared,
            # and songs already charted are skipped so re-pasting a batch
            # after failures never redoes finished work.
            opts.name, opts.artist = None, "Unknown"
            opts.skip_existing = True
        result = pipeline.run(opts)
        if result.get("skipped"):
            return
        print(f"  {'tier':<14}{'notes':>7}{'sustains':>10}{'star power':>12}"
              f"{'variety':>9}{'walk':>7}")
        for tier, row in result["summary"].items():
            flag = "  <-" if min(row["variety"], row["walk"]) < args.min_variety else ""
            print(f"  {tier:<14}{row['notes']:>7}{row['sustains']:>10}"
                  f"{row['star_power']:>12}{row['variety']:>9.2f}{row['walk']:>7.2f}{flag}")

    failed = []
    for i, item in enumerate(inputs, 1):
        try:
            chart_one(item, i, len(inputs))
        except (ValueError, OSError) as error:
            if len(inputs) == 1:
                sys.exit(str(error))
            print(f"FAILED: {type(error).__name__}: {error}")
            failed.append(item)

    # One retry pass, mainly for downloads YouTube 403-blocked mid-batch: by
    # the time the rest of the queue has charted, the throttle has usually
    # lifted (and the fallback player clients get a fresh roll).
    if failed and len(inputs) > 1:
        print(f"\nretrying {len(failed)} failed song(s)…")
        still = []
        for i, item in enumerate(failed, 1):
            try:
                chart_one(item, i, len(failed))
            except (ValueError, OSError) as error:
                print(f"FAILED again: {type(error).__name__}: {error}")
                still.append(item)
        failed = still

    if failed:
        print(f"\n{len(failed)} of {len(inputs)} failed: " + ", ".join(str(f) for f in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
