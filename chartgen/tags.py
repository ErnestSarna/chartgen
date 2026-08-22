"""Guess artist/title for a local audio file.

Order of trust: embedded tags (ID3/Vorbis/MP4 — what music libraries write),
then the "Artist - Title.ext" filename convention. Returns empty strings for
whatever it cannot determine, so callers only overwrite fields the user left
blank.
"""
from pathlib import Path

from .youtube import parse_title


def _from_tags(path: Path) -> tuple[str, str]:
    try:
        import mutagen

        audio = mutagen.File(path, easy=True)
    except Exception:
        return "", ""
    if not audio or not audio.tags:
        return "", ""

    def first(key):
        values = audio.tags.get(key) or []
        return str(values[0]).strip() if values else ""

    return first("artist"), first("title")


def guess_metadata(path) -> tuple[str, str]:
    """(artist, name); either may be '' when nothing trustworthy exists."""
    path = Path(path)
    artist, name = _from_tags(path)
    if artist and name:
        return artist, name
    # Filename fallback shares the YouTube title parser: same conventions,
    # same bracketed-noise stripping.
    file_artist, file_name = parse_title(path.stem)
    return artist or file_artist, name or file_name or path.stem
