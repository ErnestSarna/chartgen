"""The audio -> Clone Hero song folder pipeline, driven by CLI or GUI.

Progress is pushed through a callback rather than printed, so the GUI can show
it live. Runs take minutes, so there is a cancel hook checked between stages.
"""
import shutil
from pathlib import Path

from . import chart as chartio
from . import density, expression, frets, pitch as pitchmod, quality, reduce
from . import tempo as tempomod

# The only formats the .chart docs list as being in wide use. wav is deliberately
# absent: it is not in that table, so it gets transcoded rather than passed through.
PLAYABLE_AUDIO = {".ogg", ".mp3", ".opus"}
OPUS_RATE = 48000  # Opus accepts only 8/12/16/24/48 kHz


class Cancelled(Exception):
    """Raised when the caller asked to stop between stages."""


def _folder_name(artist: str, name: str) -> str:
    """Song folder name: Windows-illegal characters stripped, never empty."""
    import re

    return re.sub(r'[<>:"/\\|?*]', "", f"{artist} - {name}").strip(" -.") or "song"


def _display_name(opts, audio: Path) -> str:
    """Song title with the [chartgen] tag, so generated charts are obvious in
    the CH song list next to human charts of the same track."""
    name = opts.name or audio.stem
    return name if "[chartgen]" in name else f"{name} [chartgen]"


def stage_audio(src: Path, dest_dir: Path) -> str:
    """Put playable audio in the song folder under the reserved name `song.*`.

    Transcodes to Opus rather than Vorbis. Vorbis would be the obvious choice,
    but libsndfile 1.2.2 on Windows hard-crashes encoding it (0xC00000FD stack
    overflow, for float64, float32 and int16 alike) — a native crash that cannot
    even be caught. Opus encodes fine and the .chart docs recommend it anyway.
    """
    if src.suffix.lower() in PLAYABLE_AUDIO:
        shutil.copy2(src, dest_dir / f"song{src.suffix.lower()}")
        return f"song{src.suffix.lower()}"

    import librosa
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(src, dtype="float32", always_2d=True)
    data = data.T  # librosa wants (channels, samples)
    if sr != OPUS_RATE:
        # Resampling is mandatory, not a nicety: writing 44.1k samples while
        # claiming 48k plays the whole song ~9% fast and desyncs every note.
        data = librosa.resample(data, orig_sr=sr, target_sr=OPUS_RATE)
    sf.write(dest_dir / "song.opus", np.ascontiguousarray(data.T), OPUS_RATE,
             format="OGG", subtype="OPUS")
    return "song.opus"


def run(opts, progress=print, should_cancel=lambda: False) -> dict:
    """Generate a song folder. `opts` matches the CLI namespace.

    Returns {'song_dir', 'summary', 'bpm', 'tempo_events', 'star_power',
    'sections', 'variety'}. Raises Cancelled, or ValueError for bad input.
    """
    def check():
        if should_cancel():
            raise Cancelled()

    import librosa
    import torch
    from chart.tokenizer import SimpleTokenizerGuitar

    from . import youtube

    if youtube.is_youtube_url(str(opts.audio)):
        progress("[0/6] downloading from YouTube")
        fetched = youtube.download(str(opts.audio), Path(opts.outdir) / "_downloads",
                                   progress,
                                   cookies_from=getattr(opts, "cookies_from", None))
        progress(f"      {fetched['title']}")
        opts.audio = fetched["audio"]
        opts.thumbnail_url = fetched.get("thumbnail")
        # Autofill metadata the user did not supply; a pasted link rarely
        # comes with artist/title typed in.
        if not getattr(opts, "name", None):
            opts.name = fetched["name"]
        if getattr(opts, "artist", "Unknown") in ("", "Unknown"):
            opts.artist = fetched["artist"]
        check()

    audio = Path(opts.audio)
    if not getattr(opts, "name", None) or getattr(opts, "artist", "Unknown") in ("", "Unknown"):
        from . import tags

        artist, name = tags.guess_metadata(audio)
        if not getattr(opts, "name", None) and name:
            opts.name = name
        if getattr(opts, "artist", "Unknown") in ("", "Unknown") and artist:
            opts.artist = artist

    if getattr(opts, "skip_existing", False):
        # Batch reruns (re-pasted playlists after 403s) must not re-chart
        # finished songs. Single-song runs skip this: regenerating in place
        # is how you reroll a chart you did not like.
        existing = Path(opts.outdir) / _folder_name(opts.artist, _display_name(opts, audio))
        if (existing / "notes.chart").is_file():
            progress(f"      already charted, skipping: {existing.name}")
            return {"skipped": True, "song_dir": existing}

    progress(f"[1/6] reading {audio.name}")
    y, sr = tempomod.load(str(audio))
    duration_s = librosa.get_duration(y=y, sr=sr)
    if duration_s < 30:
        raise ValueError(f"audio must be at least 30s, got {duration_s:.0f}s")
    tempo = tempomod.detect(y, sr, bpm_mult=getattr(opts, "bpm_mult", "auto"))
    res = tempo.resolution
    progress(f"      {duration_s:.0f}s, {tempo.bpm:.1f} BPM, "
             f"{len(tempo.sync_track())} tempo event(s)")
    check()

    engine = getattr(opts, "engine", "audio2chart")
    if engine == "basicpitch":
        from . import transcribe

        progress("[2/6] transcribing (Basic Pitch engine)")
        events = transcribe.transcribe(str(audio), progress)
        check()
        progress("[3/6] building Expert from transcription")
        expert = transcribe.expert_from_notes(events, tempo, subdiv=opts.subdiv)
        if getattr(opts, "density", "onset") == "onset":
            before = len({t for t, _, _ in expert})
            expert = density.gate_by_onsets(expert, y, sr, tempo)
            after = len({t for t, _, _ in expert})
            if before > after:
                progress(f"      density: {before} -> {after} positions")
        if not expert:
            raise ValueError(
                "transcription found no chartable notes — is this audio "
                "mostly percussion or noise?")
        best = min(quality.variety_score(expert), quality.walk_score(expert))
        progress(f"      {quality.describe(expert)}")
        if best < opts.min_variety:
            progress(f"      warning: score {best:.2f} below "
                     f"{opts.min_variety:.2f} (transcription is "
                     f"deterministic; retries would not change it)")
        return _finish_chart(opts, progress, check, y, sr, tempo, expert, best,
                             engine, audio, duration_s)

    progress(f"[2/6] loading {opts.model}")
    from .model import PitchCharter, load_charter

    model = load_charter(opts.model)
    conditioner_path = getattr(opts, "conditioner", None)
    if conditioner_path:
        # A trained conditioner makes the model itself pitch-aware; wrap it so
        # generation feeds pitch into the audio memory.
        pitched = PitchCharter(model)
        pitched.conditioner.load_state_dict(
            torch.load(conditioner_path, map_location="cpu"))
        model = pitched
        progress("      pitch conditioner loaded")
    check()

    fret_mode = getattr(opts, "fret_mode", "pitch")
    pitch_future, f0, voiced = None, None, None
    if fret_mode == "pitch":
        # Computed once per song, reused across attempts — and in a thread,
        # because the ~30s of CQT/harmonic-separation CPU work can hide
        # entirely under the GPU's generation time.
        from concurrent.futures import ThreadPoolExecutor

        progress("      extracting pitch features (overlapped with generation)")
        pitch_pool = ThreadPoolExecutor(max_workers=1)
        pitch_future = pitch_pool.submit(
            pitchmod.pitch_features, y, sr, model.config.grid_ms)
        pitch_pool.shutdown(wait=False)

    progress("[3/6] generating Expert onsets")
    reverse_map = SimpleTokenizerGuitar().reverse_map
    seed = getattr(opts, "seed", None)
    expert, best = None, -1.0
    for attempt in range(1, opts.attempts + 1):
        check()
        if seed is not None:
            # Each attempt gets its own deterministic stream; otherwise every
            # retry would resample the identical chart it just rejected.
            torch.manual_seed(seed + attempt - 1)
        seqs = model.generate(str(audio), temperature=opts.temperature,
                              top_k=opts.top_k)
        candidate = chartio.notes_from_tokens(
            torch.cat(seqs).flatten().cpu().tolist(),
            model.config.grid_ms, tempo, reverse_map, subdiv=opts.subdiv,
        )
        if getattr(opts, "density", "onset") == "onset":
            before = len({t for t, _, _ in candidate})
            candidate = density.gate_by_onsets(candidate, y, sr, tempo)
            after = len({t for t, _, _ in candidate})
            if before > after:
                progress(f"      density: {before} -> {after} positions "
                         f"(dropped {before - after} without onset evidence)")
        if fret_mode == "pitch":
            if f0 is None:
                _, f0, voiced = pitch_future.result()
            raw = candidate
            candidate = frets.reassign_frets(candidate, tempo, f0, voiced,
                                             model.config.grid_ms)
            # Sparse/ambient songs can have so little pitch movement that the
            # quantile mapping collapses onto one fret — and being
            # deterministic, retries reproduce the same collapse (playtested:
            # a chart that was one repeated green sustain). The model's own
            # lanes are the better chart then.
            if (quality.variety_score(candidate) < 0.40
                    and quality.variety_score(raw) > quality.variety_score(candidate)):
                progress("      pitch range too narrow for fret mapping; "
                         "using model frets for this song")
                candidate = raw
        # Both matter: variety alone passed a chart that just walked the
        # fretboard, so score on the weaker of the two.
        score = min(quality.variety_score(candidate), quality.walk_score(candidate))
        progress(f"      attempt {attempt}/{opts.attempts}: {quality.describe(candidate)}")
        if score > best:
            expert, best = candidate, score
        if score >= opts.min_variety:
            break
    if best < opts.min_variety:
        progress(f"      warning: best score {best:.2f} is below {opts.min_variety:.2f}; "
                 f"chart may lean on too few frets or walk the fretboard")

    if not expert:
        raise ValueError(
            "model produced no notes. temperature 0 selects greedy decoding, "
            "which collapses to 'no note' at every step; use a positive value."
        )

    return _finish_chart(opts, progress, check, y, sr, tempo, expert, best,
                         engine, audio, duration_s)


def _finish_chart(opts, progress, check, y, sr, tempo, expert, best,
                  engine, audio, duration_s) -> dict:
    """Everything downstream of Expert-note production, shared by engines:
    difficulty target, reduction, expression, lyrics, art, rating, writing."""
    res = tempo.resolution
    from . import rating

    target_diff = getattr(opts, "target_diff", None)
    if target_diff is not None:
        natural = rating.rate_expert(expert, tempo)
        if natural > int(target_diff):
            expert, achieved = reduce.thin_to_rating(expert, res, tempo,
                                                     int(target_diff))
            progress(f"      target difficulty {target_diff}: thinned "
                     f"{natural}/6 -> {achieved}/6")
        elif natural < int(target_diff):
            progress(f"      target difficulty {target_diff}: chart rates "
                     f"{natural}/6 naturally; notes are never invented, so "
                     f"the target only thins (try --density model to keep "
                     f"every generated note)")

    name = _display_name(opts, audio)
    charted_by = ("basic-pitch" if engine == "basicpitch"
                  else str(opts.model).split("/")[-1])
    meta = {
        "name": name, "artist": opts.artist, "album": opts.album,
        "genre": opts.genre, "year": opts.year,
        "charter": f"chartgen ({charted_by})",
    }

    progress("[4/6] deriving Hard/Medium/Easy")
    check()
    if getattr(opts, "reducer", "chartgen") == "easygen":
        # The vendored reducer needs a chart skeleton to read beat positions.
        skeleton = chartio.write_chart({"ExpertSingle": expert}, tempo, meta, "song.ogg")
        tiers = {"ExpertSingle": expert, **reduce.derive_lower_tiers_vendored(skeleton)}
    else:
        tiers = {"ExpertSingle": expert,
                 **reduce.derive_tiers(expert, res, bpm=tempo.bpm)}

    progress("[5/6] adding sustains, star power, sections")
    end_tick = int(tempo.time_to_beat(duration_s) * res)
    if opts.no_sustains:
        tiers = {n: [(t, l, 0) for t, l, _ in notes] for n, notes in tiers.items()}
    elif engine == "basicpitch":
        # Expert already carries REAL sustains (transcribed note durations);
        # only propagate them down, never overwrite with the gap heuristic.
        from . import transcribe

        tiers = transcribe.propagate_sustains(tiers)
    else:
        tiers = expression.add_sustains_all_tiers(
            tiers, res, end_tick, min_gap_beats=opts.min_sustain_gap
        )
    star_power = () if opts.no_star_power else expression.star_power_phrases(
        expert, res, duration_s
    )
    events = () if opts.no_sections else expression.sections(y, sr, tempo)
    lyric_events = ()
    if getattr(opts, "lyrics", True):
        progress("      transcribing vocals for lyrics")
        try:
            from . import lyrics as lyricsmod

            lyric_events = lyricsmod.transcribe(str(audio), tempo, progress)
        except Exception as error:  # lyrics are a nice-to-have, never fatal
            progress(f"      lyrics skipped: {type(error).__name__}: {error}")
    solo_phrases = () if getattr(opts, "no_solos", False) else expression.solos(
        expert, list(events), list(lyric_events), res
    )
    progress(f"      {len(star_power)} star power phrase(s), {len(events)} section(s), "
             f"{len(solo_phrases)} solo(s), HOPOs {'on' if opts.hopos else 'off'}")
    check()

    progress("[6/6] writing song folder")
    # YouTube titles carry characters Windows forbids in folder names
    # (measured in the wild: Baldur's Gate 3 - "I Want To Live"). Only the
    # folder is sanitised; song.ini keeps the original text.
    song_dir = Path(opts.outdir) / _folder_name(opts.artist, name)
    song_dir.mkdir(parents=True, exist_ok=True)
    audio_filename = stage_audio(audio, song_dir)
    from . import art

    art.add_album_art(song_dir, audio, getattr(opts, "thumbnail_url", None), progress)
    chart_path = song_dir / "notes.chart"
    chart_path.write_text(
        chartio.write_chart(tiers, tempo, meta, audio_filename, star_power, events,
                            lyrics=lyric_events, solos=solo_phrases),
        encoding="utf-8",
    )
    from . import rating

    tier = rating.rate_expert(expert, tempo)
    progress(f"      difficulty rating: {tier}/6")
    # The menu preview should land on an interesting part, not the intro.
    # Star power already marks the densest bar-aligned windows, so its first
    # phrase is the best "hook" estimate we have; otherwise a quarter in.
    preview_ms = int(duration_s * 250)
    if star_power:
        preview_ms = int(tempo.beat_to_time(star_power[0][0] / res) * 1000)
    (song_dir / "song.ini").write_text(
        chartio.write_song_ini(meta, int(duration_s * 1000), hopos=opts.hopos,
                               diff_guitar=tier, preview_start_ms=preview_ms),
        encoding="utf-8",
    )

    summary = {}
    tier_lanes = {"MediumSingle": 4, "EasySingle": 3}  # RBN-style lane budgets
    for tier, (notes, sustains, sp) in chartio.summarize(chart_path).items():
        summary[tier] = {
            "notes": notes, "sustains": sustains, "star_power": sp,
            "variety": quality.variety_score(tiers[tier], tier_lanes.get(tier, 5)),
            "walk": quality.walk_score(tiers[tier]),
        }
    progress(f"done -> {song_dir}")
    return {
        "song_dir": song_dir, "summary": summary, "bpm": tempo.bpm,
        "tempo_events": len(tempo.sync_track()), "star_power": len(star_power),
        "sections": len(events), "score": best,
    }
