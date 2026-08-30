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

    engine = getattr(opts, "engine", "basicpitch")
    if engine == "basicpitch":
        from . import transcribe

        progress("[2/6] transcribing (Basic Pitch engine)")
        events = transcribe.transcribe(str(audio), progress)
        check()
        progress("[3/6] building Expert from transcription")
        if not getattr(opts, "no_bass_fallback", False):
            extra = []
            if transcribe.starved_runs(events, tempo):
                # Starved stretches first get the strong medicine: transcribe
                # the isolated stems there (separation removes the masking
                # that collapsed the mix transcription). Demucs results are
                # cached, so lyrics/solos reuse this same separation later.
                from . import stems as stemsmod

                st = stemsmod.separate(str(audio), progress)
                if st is not None:
                    extra = transcribe.stem_rescue_events(events, tempo, st,
                                                          progress)
                    if extra:
                        progress(f"      stem rescue: {len(extra)} note(s) "
                                 f"transcribed from isolated stems where the "
                                 f"mix transcription starves")
            if not extra:
                extra = transcribe.bass_fallback_events(events, tempo)
                if extra:
                    progress(f"      register fallback: {len(extra)} bass "
                             f"note(s) admitted where the melodic selection "
                             f"starves")
            if extra:
                events = sorted(events + extra)
        swing = set()
        if getattr(opts, "swing", False):
            # Opt-in: on real songs the detected beat grid's local phase
            # error (±40ms) exceeds the 55ms separating the two grids, so
            # per-onset classification misfires on straight songs (Faded
            # measured 61 wrongly-swung beats). Until beat tracking is that
            # precise, straight 16ths are the safe default.
            swing = transcribe.swung_beats(events, tempo)
            if swing:
                progress(f"      swing: {len(swing)} beat(s) quantized to "
                         f"the triplet grid")
        expert = transcribe.expert_from_notes(
            events, tempo, subdiv=opts.subdiv,
            min_sustain_beats=getattr(opts, 'min_sustain_beats', 0.5),
            allow_opens=getattr(opts, "opens", True),
            swing_beats=swing,
            ornaments=not getattr(opts, "no_ornaments", False))
        orn = sum(1 for t, _, _ in expert
                  if t % (res // 8) == 0 and t % (res // 4) != 0)
        if orn:
            progress(f"      {orn} 32nd grace note(s) recovered from the "
                     f"transcription")
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
                             engine, audio, duration_s, bp_events=events)

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
    best_key = (False, -1.0)
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
        # fretboard, so score on the weaker of the two. That pair is a FLOOR,
        # though — it cannot tell whether a roll caught the actual part.
        gate = min(quality.variety_score(candidate), quality.walk_score(candidate))
        fidelity = density.onset_fidelity(candidate, y, sr, tempo)
        progress(f"      attempt {attempt}/{opts.attempts}: "
                 f"{quality.describe(candidate)}  onset-F1 {fidelity:.2f}")
        # Sampling makes this engine high-variance — playtested as "bigger
        # wins but bigger losses". Rank rolls by how well they match the audio,
        # preferring any roll that clears the floor over one that does not.
        key = (gate >= opts.min_variety, fidelity)
        if key > best_key:
            expert, best, best_key = candidate, gate, key
        if key[0] and fidelity >= getattr(opts, "min_fidelity", 0.60):
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


def _group(notes):
    out: dict[int, list] = {}
    for tick, lane, sus in notes:
        out.setdefault(tick, []).append(lane)
    return out


def _finish_chart(opts, progress, check, y, sr, tempo, expert, best,
                  engine, audio, duration_s, bp_events=None) -> dict:
    """Everything downstream of Expert-note production, shared by engines:
    difficulty target, reduction, expression, lyrics, art, rating, writing."""
    res = tempo.resolution
    from . import rating

    # Human charts of the same songs jump 3+ lanes on 0-1% of consecutive
    # single notes; ours did it 9-30% of the time. Smooth before reduction so
    # every tier inherits playable hand movement.
    # Playtest: charts were unplayable inside 30 seconds because harmony was
    # charted as chords. Do this before the jump smoothing so it sees the
    # shapes that actually survive.
    if not getattr(opts, "keep_dense_chords", False):
        from . import texture

        before = sum(1 for _, g in _group(expert).items() if len(g) > 1)
        # Texture first, backstop second: smoothing shape flicker into
        # honest same-shape runs is what lets chord riffs SURVIVE the
        # rapid-change demotion (same-shape repeats were never the
        # playability problem), and narrowing wide chords keeps the chord
        # where the old rule deleted it.
        expert = texture.smooth_chord_shapes(expert, res)
        expert = texture.narrow_wide_chords(expert, res)
        n_before = len({t for t, _, _ in expert})
        expert = texture.consolidate_gallops(expert, res)
        dropped = n_before - len({t for t, _, _ in expert})
        if dropped:
            progress(f"      rhythm: {dropped} one-off 16th fragment(s) "
                     f"consolidated to eighths")
        # ponytail: a "chord-riff song" gate (keep strummed chords on
        # punk/strum-rock, library median 49% chords vs EDM 7%) was shipped
        # here and removed by audit: no chart-level signal separates the two.
        # Raw chord share measured 33% on BOTH Mary Jane (strummy rock) and
        # Faded (EDM), and same-shape-run share 79% vs 82% - full-mix
        # transcription makes strummed guitar and pad stabs look identical.
        # Needs the Demucs guitar stem (already computed later for solos)
        # moved earlier; until then, unconditional demotion is the version
        # that survived playtesting.
        expert = reduce.simplify_rapid_chords(expert, res)
        after = sum(1 for _, g in _group(expert).items() if len(g) > 1)
        if before > after:
            progress(f"      chords: {before} -> {after} positions "
                     f"(human charts use 5-13% on EDM)")

    if not getattr(opts, "no_motifs", False):
        from . import motifs

        # settle_pushes MOVES notes without changing position counts, so a
        # dropped-positions delta alone would hide it (the stairs no-op bug
        # taught that every feature needs a visible count).
        ticks_before = {t for t, _, _ in expert}
        expert = motifs.settle_pushes(expert, res)
        moved = len({t for t, _, _ in expert} - ticks_before)
        n_before = len({t for t, _, _ in expert})
        expert = motifs.break_machine_gun(expert, res)
        thinned = n_before - len({t for t, _, _ in expert})
        if moved or thinned:
            progress(f"      flow: {moved} stray push(es) settled onto the "
                     f"8th grid, {thinned} machine-gun note(s) thinned")

    # Nothing may be charted past the end of the audio: a note there is
    # simply unhittable. Cheap insurance against any tempo-map drift in an
    # outro, which is exactly where beat detection is least reliable.
    end_of_audio = int(tempo.time_to_beat(duration_s) * res)
    trimmed = [n for n in expert if n[0] <= end_of_audio]
    if len(trimmed) < len(expert):
        progress(f"      trimmed {len(expert) - len(trimmed)} note(s) past the "
                 f"end of the audio")
        expert = trimmed

    if not getattr(opts, "no_playability", False):
        from . import playability

        before = playability.analyse(expert, tempo)["impossible"]
        expert = playability.enforce(expert, tempo)
        after = playability.analyse(expert, tempo)["impossible"]
        if before > after:
            progress(f"      playability: {before:.1%} -> {after:.1%} of "
                     f"transitions too fast to play (human median 1.3%)")

    if not getattr(opts, "no_brightness_lanes", False):
        stretches, _ = frets.degenerate_stretches(expert, res)
        if stretches:
            from . import stems as stemsmod

            st = stemsmod.separate(str(audio), progress)
            if st is not None:
                import numpy as np

                melodic = (st.get("bass", 0) + st.get("other", 0))
                if isinstance(melodic, np.ndarray):
                    expert, relaned = frets.relane_by_brightness(
                        expert, tempo, melodic, stemsmod.SR)
                    if relaned:
                        progress(f"      brightness lanes: {relaned} stuck "
                                 f"stretch(es) re-laned from the filter/"
                                 f"timbre contour (humans chart 4 lanes "
                                 f"even in bass-led sections)")

    max_jump = int(getattr(opts, "max_fret_jump", 2))
    if max_jump < 4:
        before = frets.step_share(expert, 3)
        expert = frets.smooth_fret_jumps(expert, tempo, max_step=max_jump)
        after = frets.step_share(expert, 3)
        if before > after:
            progress(f"      fret jumps >=3 lanes: {before:.0%} -> {after:.0%}")

    if not getattr(opts, "no_motifs", False):
        from . import motifs

        # Lane-only reshapes, after jump smoothing so they see (and keep)
        # playable hand travel: shapeless fast runs become 3-4 note wrap
        # chunks, monotonic chord walks take the RBN rung ladder, and lone
        # pickups into a chord take the chord's root.
        lanes_before = {(t, l) for t, l, _ in expert}
        expert = motifs.chunk_rolls(expert, res)
        expert = motifs.shape_chord_walks(expert, res)
        expert = motifs.pickup_roots(expert, res)
        relaned = len({(t, l) for t, l, _ in expert} - lanes_before)
        if relaned:
            progress(f"      lanes: {relaned} note(s) reshaped into wrap "
                     f"chunks, chord-ladder rungs, or pickup roots")

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

    events = () if opts.no_sections else expression.sections(y, sr, tempo)
    if events and not getattr(opts, "no_section_reuse", False):
        from . import structure

        ticks = [tick for tick, _ in events]
        lengths = [b - a for a, b in zip(ticks, ticks[1:])] + [0]
        signatures = structure.section_signatures(y, sr, tempo, ticks)
        repeats = structure.find_repeats(signatures, lengths)
        if repeats:
            expert = structure.reuse_patterns(expert, ticks, repeats)
            progress(f"      reused patterns across {len(repeats)} repeated "
                     f"section(s)")

    if not getattr(opts, "no_riff_unify", False):
        from . import structure

        n_bars = max((t for t, _, _ in expert), default=0) // (4 * res) + 1
        profiles = structure.bar_profiles(y, sr, tempo, n_bars)
        expert, stamped = structure.unify_riff_bars(expert, profiles, res)
        if stamped:
            progress(f"      riffs: {stamped} bar(s) unified onto their "
                     f"cluster's consensus pattern (human median: 48% of "
                     f"bars repeat)")

    progress("[4/6] deriving Hard/Medium/Easy")
    check()
    if getattr(opts, "reducer", "chartgen") == "easygen":
        # The vendored reducer needs a chart skeleton to read beat positions.
        skeleton = chartio.write_chart({"ExpertSingle": expert}, tempo, meta, "song.ogg")
        tiers = {"ExpertSingle": expert, **reduce.derive_lower_tiers_vendored(skeleton)}
    else:
        tiers = {"ExpertSingle": expert,
                 **reduce.derive_tiers(expert, res, bpm=tempo.bpm)}
    tiers = reduce.enforce_chord_rules(tiers)

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
            tiers, res, end_tick, min_gap_beats=opts.min_sustain_gap,
            bpm=tempo.bpm,
        )
    if not opts.no_sustains and not getattr(opts, "no_motifs", False):
        from . import motifs
        from . import transcribe as transcribemod

        # Sustain stairs: hold each note of a stepped phrase to just before
        # its successor (78% of human charts, ~6 phrases per 100 bars; we had
        # none). Before top-up, so the holds count toward the calibrated
        # sustain share; re-propagated so lower tiers inherit the same holds.
        stepped = motifs.legato_stairs(tiers["ExpertSingle"], res,
                                       bpm=tempo.bpm)
        if stepped != tiers["ExpertSingle"]:
            grown = sum(1 for a, b in zip(sorted(tiers["ExpertSingle"]),
                                          sorted(stepped)) if a[2] != b[2])
            progress(f"      {grown} note(s) held as sustain stairs "
                     f"(78% of human charts, ~6 phrases per 100 bars)")
            tiers["ExpertSingle"] = stepped
            tiers = transcribemod.propagate_sustains(tiers)
    if not opts.no_sustains:
        # Human charts sustain MORE as tiers get easier (8.7% of Expert notes
        # up to 15.3% on Easy); propagating Expert's lengths down left every
        # tier flat at Expert's rate.
        tiers = {
            name: expression.top_up_sustains(
                notes, res, end_tick,
                expression.TIER_SUSTAIN_SHARE.get(name, 0.09))
            for name, notes in tiers.items()
        }
    star_power = () if opts.no_star_power else expression.star_power_phrases(
        expert, res, duration_s
    )
    lyric_events = ()
    if getattr(opts, "lyrics", True):
        progress("      looking up lyrics")
        try:
            from . import lyrics as lyricsmod

            lyric_events = lyricsmod.collect(
                str(audio), tempo, opts.artist,
                # The [chartgen] tag is ours; a lyrics database has never
                # heard of it.
                opts.name or audio.stem, duration_s,
                getattr(opts, "lyric_source", "auto"), progress, y, sr)
        except Exception as error:  # lyrics are a nice-to-have, never fatal
            progress(f"      lyrics skipped: {type(error).__name__}: {error}")
    tap_ticks = set()
    if getattr(opts, "taps", False):
        from . import taps as tapsmod

        tap_ticks = tapsmod.detect(expert, y, sr, tempo, progress,
                                   audio_path=str(audio),
                                   section_marks=list(events))
    if not getattr(opts, "keep_dense_chords", False) and events:
        from . import texture

        grown = texture.climax_chords(tiers["ExpertSingle"], list(events), res)
        if len(grown) > len(tiers["ExpertSingle"]):
            progress(f"      {len(grown) - len(tiers['ExpertSingle'])} section-entry "
                     f"chord(s) grown to three notes")
        tiers["ExpertSingle"] = grown
    if (bp_events and not opts.no_sustains
            and not getattr(opts, "no_motifs", False)):
        from . import transcribe as transcribemod

        # Extended-sustain ladders (a held lane rings while higher lanes
        # join) go on Expert ONLY, after every tier is derived: reduced
        # spacing dislikes overlaps, and the joins are Expert expression.
        laddered, made = transcribemod.extend_ladders(
            tiers["ExpertSingle"], bp_events, tempo)
        if made:
            tiers["ExpertSingle"] = laddered
            progress(f"      {made} sustain ladder(s): a held note rings "
                     f"while higher lanes join (39% of human charts)")
    solo_phrases = ()
    if not getattr(opts, "no_solos", False):
        from . import solo as solomod

        solo_phrases = solomod.detect(expert, list(events), list(lyric_events),
                                      y, sr, tempo, progress,
                                      audio_path=str(audio))
    forced_ticks: set[int] = set()
    if opts.hopos and not getattr(opts, "no_motifs", False):
        from . import motifs

        # Forcing only means anything when natural HOPOs are on: with them
        # off, song.ini's hopo_frequency=1 already makes every note a strum.
        # Taps already never strum; a force flag on one is dead markup.
        forced_ticks = motifs.legato_forcing(tiers["ExpertSingle"], res) - tap_ticks
        if forced_ticks:
            progress(f"      {len(forced_ticks)} note(s) forced to legato "
                     f"HOPO (human median 3.9 per 100 notes)")
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
                            lyrics=lyric_events, solos=solo_phrases,
                            taps=tap_ticks, forced=forced_ticks),
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
