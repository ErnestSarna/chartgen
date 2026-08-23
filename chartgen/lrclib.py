"""Real synced lyrics from LRCLIB, tried before transcribing anything.

Measured against our own output on seven charted songs, Whisper covered
80-125% of the reference word count: it both drops verses (K.Flay 80%,
Summer 88%) and invents a tail (The Stroke 125%). Worse, the invented tail
is not random — it is Whisper's documented YouTube-caption hallucination,
and "Thank you for watching, I'll see you in the next one" was sitting in
the outro of a real chart.

LRCLIB is a community database of LRC files keyed on artist/title/duration.
It needs no API key, and it returned a hand-synced match for every one of
those seven songs. When a match exists it is simply the right answer: the
words are the real words and the line timings were synced by a person.

Licence note: the LRC files are user-contributed and the underlying lyrics
stay the songwriter's copyright, so this fetches at runtime, on the user's
own machine, for their own chart. Nothing is bundled or redistributed.

Only the lookup lives here; turning lines into CH events is lyrics.py's job.
"""
import json
import re
import urllib.error
import urllib.parse
import urllib.request

API = "https://lrclib.net/api/search"
# LRCLIB asks clients to identify themselves rather than pose as a browser.
USER_AGENT = "chartgen/1.0 (https://github.com/ErnestSarna/chartgen)"

# LRC timestamps are absolute, so a lyric sheet for a different cut of the
# song drifts further out the longer it plays. Duration is the only cheap
# evidence that we matched the same recording.
MAX_DURATION_DRIFT = 12.0

_LRC_LINE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)")
# Text a charter's metadata carries but a lyrics database does not:
# "(Official Video)", "[chartgen]", "(Remix)" and friends.
_NOISE = re.compile(r"\s*[\(\[][^\)\]]*[\)\]]", re.ASCII)


def parse_lrc(text: str) -> list[tuple[float, str]]:
    """[(seconds, line_text)] from an LRC body, blank lines dropped.

    Instrumental breaks are published as timestamped empty lines, which is
    real information — it says the singing stops here — but not something to
    put on the highway.
    """
    out = []
    for line in text.splitlines():
        found = _LRC_LINE.match(line.strip())
        if not found:
            continue
        body = found.group(3).strip()
        if body and body not in ("♪", "..."):
            out.append((int(found.group(1)) * 60 + float(found.group(2)), body))
    return sorted(out)


def _search(artist: str, title: str, timeout: float) -> list[dict]:
    query = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    request = urllib.request.Request(f"{API}?{query}",
                                     headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _clean(text: str) -> str:
    """Strip the parentheticals that keep a real title from matching."""
    return _NOISE.sub("", text).strip() or text.strip()


def find(artist: str, title: str, duration_s: float, timeout: float = 15.0,
         progress=lambda m: None) -> list[tuple[float, str]] | None:
    """Hand-synced lyric lines for this recording, or None if none fit.

    Two queries at most: as titled, then with parentheticals stripped —
    "Too sweet (Remix)" and "Faded (BP)" both fail as written and match once
    trimmed. Candidates are then filtered on duration, because an LRC synced
    against a different edit is worse than no lyrics at all.
    """
    seen = []
    for name in dict.fromkeys((title, _clean(title))):
        for who in dict.fromkeys((artist, _clean(artist))):
            if not name:
                continue
            try:
                seen += _search(who, name, timeout)
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as error:
                progress(f"      LRCLIB unavailable ({type(error).__name__}); "
                         f"transcribing instead")
                return None
            if seen:
                break
        if seen:
            break

    synced = [h for h in seen if h.get("syncedLyrics") and h.get("duration")]
    if not synced:
        return None
    best = min(synced, key=lambda h: abs(float(h["duration"]) - duration_s))
    drift = abs(float(best["duration"]) - duration_s)
    if drift > MAX_DURATION_DRIFT:
        progress(f"      LRCLIB has \"{best['trackName']}\" but it runs "
                 f"{drift:.0f}s from this audio; transcribing instead")
        return None

    lines = parse_lrc(best["syncedLyrics"])
    if len(lines) < 4:
        return None
    # A sheet whose last line lands past the end of this audio is synced to
    # something else, whatever its stated duration says.
    if lines[-1][0] > duration_s + 5:
        return None
    progress(f"      LRCLIB: {best['artistName']} - {best['trackName']} "
             f"({len(lines)} synced lines)")
    return lines
