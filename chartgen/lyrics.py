"""Transcribe vocals into Clone Hero lyric events.

CH renders lyrics from `[Events]` lines: `phrase_start` opens a line at the
top of the highway, `lyric <word>` fills it word by word, `phrase_end` closes
it. Timing comes from faster-whisper word timestamps mapped through the tempo
map — lyrics are display text, so they keep their raw ticks instead of being
snapped to the note grid.

Instrumentals need no special-casing: VAD plus the no-speech filter simply
yields nothing, and the chart is written without an empty lyric line.

The model (base, ~150MB) downloads from Hugging Face on first use and is
cached. GPU is tried first; any CUDA/cuDNN mismatch falls back to CPU int8,
which transcribes a 4-minute song in a couple of minutes.
"""
# Singing over instruments legitimately scores ~0.65 no_speech_prob (measured
# 0.658 on a clearly-sung chorus), so only near-certain non-speech is dropped;
# the VAD has already cut the actual silence.
QUIET = 0.9


_MODEL = None


def _load_model(progress):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from faster_whisper import WhisperModel

    try:
        import torch

        if torch.cuda.is_available():
            try:
                # small: the GPU affords it, and sung vocals need the accuracy.
                _MODEL = WhisperModel("small", device="cuda", compute_type="float16")
                return _MODEL
            except Exception:
                progress("      (GPU transcription unavailable, using CPU)")
    except ImportError:
        pass
    _MODEL = WhisperModel("base", device="cpu", compute_type="int8")
    return _MODEL


def _clean(word: str) -> str:
    # Inner double quotes would terminate the E "..." chart line early.
    return word.strip().replace('"', "").replace("=", "-")


def transcribe(audio_path: str, tempo, progress=lambda m: None) -> list[tuple[int, str]]:
    """[(tick, event_text), ...] lyric events for write_chart, maybe empty."""
    model = _load_model(progress)
    # No VAD: it eats heavily-produced choruses (Faded lost everything past
    # 48s with it on). condition_on_previous_text off stops one misheard
    # phrase from propagating through the whole song. The no-speech filter
    # below handles instrumental stretches instead.
    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False,
    )

    res = tempo.resolution

    def tick(t: float) -> int:
        return max(0, int(round(tempo.time_to_beat(float(t)) * res)))

    events: list[tuple[int, str]] = []
    words = 0
    for segment in segments:
        if segment.no_speech_prob > QUIET or not segment.words:
            continue
        clean_words = [(w, _clean(w.word)) for w in segment.words]
        clean_words = [(w, text) for w, text in clean_words if text]
        if not clean_words:
            continue
        events.append((tick(segment.words[0].start), "phrase_start"))
        for w, text in clean_words:
            events.append((tick(w.start), f"lyric {text}"))
            words += 1
        events.append((tick(segment.end), "phrase_end"))

    if words:
        progress(f"      {words} words in {sum(1 for _, e in events if e == 'phrase_start')} phrases"
                 f" (language: {info.language})")
    else:
        progress("      no vocals detected; skipping lyrics")
        return []
    return events
