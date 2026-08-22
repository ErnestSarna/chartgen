"""Album art for the song folder (Clone Hero reads album.png).

Priority per song: art embedded in the audio file's tags (what a ripped MP3
or FLAC carries), else the YouTube thumbnail for downloaded songs. Thumbnails
are 16:9, so everything is center-cropped square and sized to 512 — CH's
song list shows square art, and letterboxed thumbnails look broken there.

Art is a nice-to-have: every function returns None/False on failure and the
pipeline never fails a chart over it.
"""
import io
from pathlib import Path

ART_SIZE = 512


def embedded_art(audio_path) -> bytes | None:
    """Cover image bytes from ID3/FLAC/MP4 tags, or None."""
    try:
        import mutagen

        audio = mutagen.File(str(audio_path))
        if audio is None:
            return None
        # FLAC / OGG carry a picture list; ID3 carries APIC frames; MP4 covr.
        pictures = getattr(audio, "pictures", None)
        if pictures:
            return pictures[0].data
        if audio.tags:
            for key, value in audio.tags.items():
                if key.startswith("APIC"):
                    return value.data
                if key == "covr" and value:
                    return bytes(value[0])
    except Exception:
        pass
    return None


def fetch(url: str) -> bytes | None:
    """Download an image URL (YouTube thumbnail), or None."""
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=20) as response:
            return response.read()
    except Exception:
        return None


def write_album(song_dir, image_bytes: bytes) -> bool:
    """Center-crop square, resize, save as album.png. False on bad image."""
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = image.size
        side = min(w, h)
        image = image.crop(((w - side) // 2, (h - side) // 2,
                            (w + side) // 2, (h + side) // 2))
        image = image.resize((ART_SIZE, ART_SIZE), Image.LANCZOS)
        image.save(Path(song_dir) / "album.png")
        return True
    except Exception:
        return False


def add_album_art(song_dir, audio_path, thumbnail_url: str | None,
                  progress=lambda m: None) -> None:
    """Best-effort album.png: embedded tag art first, then the thumbnail."""
    art = embedded_art(audio_path)
    source = "embedded tags"
    if art is None and thumbnail_url:
        art = fetch(thumbnail_url)
        source = "YouTube thumbnail"
    if art and write_album(song_dir, art):
        progress(f"      album art from {source}")
