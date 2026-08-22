"""Paste a YouTube link, get a chartable audio file.

yt-dlp fetches the best audio stream (webm/m4a — containers libsndfile cannot
read), and imageio-ffmpeg's bundled static ffmpeg converts it to WAV, which
the rest of the pipeline already handles (stage_audio transcodes WAV to a
CH-playable song.opus). No system ffmpeg install, no PATH changes.

Local personal use only, like everything else here: the downloaded audio
stays on this machine next to the generated chart.
"""
import re
import subprocess
from pathlib import Path

URL_RE = re.compile(r"^https?://(www\.|m\.|music\.)?(youtube\.com|youtu\.be)/", re.I)
# Only a *pure* playlist URL expands to all its videos. A watch?v=…&list=…
# link means "this video, which happens to be in a playlist" — charting 200
# songs because of a stray URL parameter would be a nasty surprise.
PLAYLIST_RE = re.compile(
    r"^https?://(www\.|m\.|music\.)?youtube\.com/playlist\?", re.I)

# "Artist - Title (Official Video)" is the dominant convention; strip the
# bracketed noise and split once on the first dash variant.
NOISE_RE = re.compile(
    r"\s*[(\[][^)\]]*(official|video|audio|lyric|lyrics|hd|4k|visuali[sz]er|"
    r"remaster[^)\]]*|mv|m/v)[^)\]]*[)\]]\s*", re.I)


def is_youtube_url(value: str) -> bool:
    return bool(URL_RE.match(str(value).strip()))


def is_playlist_url(value: str) -> bool:
    return bool(PLAYLIST_RE.match(str(value).strip()))


def expand_inputs(items, progress=lambda m: None) -> list:
    """Flatten a mixed list of files/videos/playlists into chartable inputs.

    Playlist URLs become one entry per video (resolved without downloading);
    everything else passes through untouched. No silent caps — a 200-video
    playlist yields 200 inputs, and the count is announced so the scale of
    what was just queued is visible before hours get committed.
    """
    out: list = []
    for item in items:
        if not is_playlist_url(str(item)):
            out.append(item)
            continue
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError

        try:
            with YoutubeDL({"extract_flat": True, "quiet": True,
                            "no_warnings": True}) as ydl:
                info = ydl.extract_info(str(item), download=False)
        except DownloadError as error:
            raise ValueError(f"could not read playlist: {error}") from error
        entries = [e["url"] for e in (info.get("entries") or []) if e.get("url")]
        progress(f"      playlist \"{(info.get('title') or '?')[:50]}\": "
                 f"{len(entries)} videos queued")
        out.extend(entries)
    return out


def parse_title(title: str) -> tuple[str, str]:
    """(artist, song) guessed from a video title; artist may come back ''."""
    clean = NOISE_RE.sub(" ", title).strip(" -–—|")
    for dash in (" - ", " – ", " — ", " | "):
        if dash in clean:
            artist, song = clean.split(dash, 1)
            return artist.strip(), song.strip()
    return "", clean.strip()


# Tried in order when YouTube's bot detection 403s the media stream: the
# android and tv API clients are frequently not enforced when the default web
# client is. Enforcement is per-video and per-session, which is why half a
# batch can fail while the rest sails through.
CLIENT_FALLBACKS = (None, ["android"], ["tv"])


def download(url: str, dest_dir: Path, progress=lambda m: None,
             cookies_from: str | None = None) -> dict:
    """Fetch audio for a YouTube URL -> {'audio', 'artist', 'name', 'title'}.

    Raises ValueError with a readable message on failure (bad URL, private or
    region-locked video, no network) so the GUI can show it as-is.

    cookies_from ('firefox', 'chrome', 'edge', …) makes requests carry the
    browser's YouTube session — the heavy remedy for persistent 403s.
    """
    import time

    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    dest_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,   # a pasted video link should never pull a playlist
        "quiet": True,
        "no_warnings": True,
    }
    if cookies_from:
        options["cookiesfrombrowser"] = (cookies_from,)

    progress("      fetching audio from YouTube")
    info = None
    for attempt, clients in enumerate(CLIENT_FALLBACKS):
        opts = dict(options)
        if clients:
            opts["extractor_args"] = {"youtube": {"player_client": clients}}
        try:
            with YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except DownloadError as error:
            blocked = "403" in str(error) or "Forbidden" in str(error)
            last = attempt == len(CLIENT_FALLBACKS) - 1
            if not blocked or last:
                hint = (" (YouTube bot detection; try again later, or use "
                        "--cookies-from <your browser>)" if blocked else "")
                raise ValueError(f"YouTube download failed: {error}{hint}") from error
            progress(f"      blocked (403), retrying via "
                     f"{CLIENT_FALLBACKS[attempt + 1][0]} client")
            time.sleep(3 * (attempt + 1))

    raw = Path(ydl.prepare_filename(info))
    if not raw.is_file():
        raise ValueError(f"yt-dlp reported success but {raw.name} is missing")

    wav = raw.with_suffix(".wav")
    if not wav.is_file():
        progress("      converting to wav")
        import imageio_ffmpeg

        result = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw),
             "-vn", "-ac", "2", "-ar", "48000", str(wav)],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not wav.is_file():
            raise ValueError(
                f"ffmpeg could not convert the download: {result.stderr[-400:]}")

    title = info.get("title") or raw.stem
    artist, name = parse_title(title)
    return {
        "audio": wav,
        "artist": artist or (info.get("uploader") or "Unknown"),
        "name": name or title,
        "title": title,
        "thumbnail": info.get("thumbnail"),
    }
