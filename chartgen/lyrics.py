"""Lyric events for Clone Hero: real synced lyrics first, transcription second.

CH renders lyrics from `[Events]` lines: `phrase_start` opens a line at the
top of the highway, `lyric <word>` fills it word by word, `phrase_end` closes
it. Timing comes from the tempo map — lyrics are display text, so they keep
their raw ticks instead of being snapped to the note grid.

Two sources, tried in order:

1. LRCLIB (see lrclib.py). Hand-synced lines, real words, no model. Measured
   on the charts we had generated, it matched every song whose metadata was
   clean, which makes it the primary source rather than a nicety.
2. faster-whisper. The fallback for anything LRCLIB does not carry —
   unreleased tracks, remixes, anything the community has not synced.

Instrumentals need no special-casing: LRCLIB has no entry and the no-speech
filter yields nothing, so the chart is written without an empty lyric line.

The Whisper model (base, ~150MB) downloads from Hugging Face on first use and
is cached. GPU is tried first; any CUDA/cuDNN mismatch falls back to CPU int8,
which transcribes a 4-minute song in a couple of minutes.
"""
import re

# Reference-decoder thresholds (openai/whisper transcribe.py, inherited by
# faster-whisper). The gate that matters is a CONJUNCTION: a segment is only
# silence when it both scores high on no-speech AND decoded poorly. We used
# to test no_speech_prob > 0.9 alone, which is neither half of that rule —
# too lax at the fade-out where hallucinations live, and too strict on quiet
# singing (measured 0.658 on a clearly-sung chorus). Against LRCLIB as a
# reference, that filter left us at 80-88% of the real word count on three
# songs and 107-125% on three others.
NO_SPEECH = 0.6
LOG_PROB = -1.0
# gzip ratio of the decoded text; a repetition loop compresses far better
# than language does. The reference default is 2.4.
COMPRESSION = 2.4
# faster-whisper skips silent stretches longer than this while assigning word
# timestamps — the named remedy for text invented over a fade-out.
SILENCE_SKIP = 2.0

# Whisper was trained on YouTube captions, so on non-speech audio it falls
# back on caption boilerplate. This is not hypothetical here: a generated
# chart ended with "Thank you for watching, I'll see you in the next one"
# over its outro, and three others closed on a bare "you".
_BOILERPLATE = (
    "thank you for watching", "thanks for watching", "thank you so much for watching",
    "see you in the next", "see you next time", "please subscribe", "like and subscribe",
    "subscribe to", "subtitles by", "subtitles created by", "amara.org",
    "transcription by", "translated by", "captions by", "all rights reserved",
)
# A trailing scrap this short, stranded after a long instrumental gap, is
# never a real closing line.
ORPHAN_WORDS = 5
ORPHAN_GAP_S = 5.0

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


# In .chart, a handful of ASCII characters are lyric MARKUP rather than text
# (per the format spec): '-' and '=' join this syllable to the next one, '_'
# stands in for a space, and '#^*%$/<>' are pitch/display directives. Whisper
# writes ordinary punctuation — a transcribed "well-known" would silently
# swallow the following word on the highway. Double quotes would end the
# E "..." line outright.
_AS_SPACE = "-=_"
_DROP = '+#^*%$/<>"§'


def _clean(word: str) -> str:
    """Strip .chart lyric markup so transcribed words display as written."""
    text = word.strip()
    for char in _AS_SPACE:
        text = text.replace(char, " ")
    for char in _DROP:
        text = text.replace(char, "")
    return " ".join(text.split())


def _phrases_to_events(phrases, tempo) -> list[tuple[int, str]]:
    """[(tick, event)] for [(word_time, word), ...] groups, in reading order.

    Emission order is the output order: write_chart keeps it for events that
    land on the same tick, which is what stops a phrase from being scrambled
    when several words quantize together.
    """
    res = tempo.resolution
    events: list[tuple[int, str]] = []
    last_end = 0
    for words in phrases:
        if not words:
            continue
        first = max(0, int(round(tempo.time_to_beat(words[0][0]) * res)))
        # Open the phrase half a beat early — never before the previous one
        # closed — so the line is on screen by the time it is sung.
        events.append((max(last_end, first - res // 2, 0), "phrase_start"))
        tick = first
        for when, text in words:
            # Non-decreasing: a source with sloppy timing must not reorder
            # the line, and equal ticks stay in the order written here.
            tick = max(tick, int(round(tempo.time_to_beat(when) * res)))
            events.append((tick, f"lyric {text}"))
        last_end = max(tick + res // 4, first + 1)
        events.append((last_end, "phrase_end"))
    return events


def from_lines(lines, tempo, duration_s: float) -> list[tuple[int, str]]:
    """CH events from LRC lines: [(seconds, text)] with line-level timing.

    LRC is synced per line, not per word, so words are spread across the line
    at a plausible singing rate — weighted by length, because "the" is not
    held as long as "shadow". The spread is capped so a line before a long
    instrumental break does not crawl across it; the phrase timing, which is
    what the player actually reads, comes straight from the human sync.
    """
    phrases = []
    for i, (start, text) in enumerate(lines):
        words = [w for w in (_clean(part) for part in text.split()) if w]
        if not words:
            continue
        following = lines[i + 1][0] if i + 1 < len(lines) else duration_s
        room = max(0.3, following - start - 0.1)
        # ~2.6 words/sec is an ordinary sung line; never overrun the gap.
        span = min(room, 0.38 * len(words))
        weights = [len(w) + 1 for w in words]
        total = sum(weights)
        at, spread = start, []
        for word, weight in zip(words, weights):
            spread.append((min(at, start + span), word))
            at += span * weight / total
        phrases.append(spread)
    return _phrases_to_events(phrases, tempo)


def _boilerplate(text: str) -> bool:
    flat = re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()
    return any(phrase in flat for phrase in _BOILERPLATE)


def _drop_invented_tail(segments):
    """Remove the caption boilerplate Whisper strands in instrumental outros.

    Two structural signals, no phrase list needed for most of it: text that
    collapses onto a single timestamp is not something anyone sang, and a
    handful of words marooned after a long gap at the end of a song is the
    fade-out hallucination rather than a closing line. The phrase list only
    catches the longer, well-formed hallucinations that pass both.
    """
    kept = list(segments)
    while kept:
        start, end, words = kept[-1]
        gap = start - kept[-2][1] if len(kept) > 1 else 0.0
        stranded = gap >= ORPHAN_GAP_S and len(words) <= ORPHAN_WORDS
        collapsed = len(words) > 2 and len({round(w[0], 2) for w in words}) <= 1
        if stranded or collapsed or _boilerplate(" ".join(w for _, w in words)):
            kept.pop()
            continue
        break
    # Boilerplate anywhere, not just at the end — a mid-song instrumental
    # break invites the same invention.
    return [s for s in kept if not _boilerplate(" ".join(w for _, w in s[2]))]


def transcribe(audio_path: str, tempo, progress=lambda m: None) -> list[tuple[int, str]]:
    """[(tick, event_text), ...] lyric events from the audio itself."""
    model = _load_model(progress)
    # No VAD: it eats heavily-produced choruses (Faded lost everything past
    # 48s with it on). condition_on_previous_text off stops one misheard
    # phrase from propagating through the whole song. The thresholds below
    # are the reference decoder's, which handle instrumental stretches.
    segments, info = model.transcribe(
        str(audio_path), word_timestamps=True, vad_filter=False,
        condition_on_previous_text=False,
        no_speech_threshold=NO_SPEECH, log_prob_threshold=LOG_PROB,
        compression_ratio_threshold=COMPRESSION,
        hallucination_silence_threshold=SILENCE_SKIP,
    )

    collected = []
    previous = None
    for segment in segments:
        if not segment.words:
            continue
        words = [(float(w.start), _clean(w.word)) for w in segment.words]
        words = [(t, text) for t, text in words if text]
        if not words:
            continue
        flat = " ".join(text for _, text in words).lower()
        # A segment repeating the one before it is a decoder loop, not a
        # repeated chorus line — a real repeat is separated by other words.
        if flat == previous:
            continue
        previous = flat
        collected.append((float(segment.start), float(segment.end), words))

    collected = _drop_invented_tail(collected)
    events = _phrases_to_events([w for _, _, w in collected], tempo)

    count = sum(1 for _, e in events if e.startswith("lyric "))
    if not count:
        progress("      no vocals detected; skipping lyrics")
        return []
    progress(f"      {count} words in "
             f"{sum(1 for _, e in events if e == 'phrase_start')} phrases"
             f" (language: {info.language})")
    return events


def collect(audio_path, tempo, artist: str, title: str, duration_s: float,
            source: str = "auto", progress=lambda m: None) -> list[tuple[int, str]]:
    """Lyric events from the best source available.

    'auto' looks the song up on LRCLIB and only transcribes if that misses;
    'online' and 'transcribe' pin one source.
    """
    if source in ("auto", "online"):
        from . import lrclib

        lines = lrclib.find(artist, title, duration_s, progress=progress)
        if lines:
            events = from_lines(lines, tempo, duration_s)
            progress(f"      {sum(1 for _, e in events if e.startswith('lyric '))}"
                     f" words in {len(lines)} synced lines")
            return events
        if source == "online":
            progress("      no synced lyrics found; skipping lyrics")
            return []
    return transcribe(audio_path, tempo, progress)
