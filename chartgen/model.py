"""Pitch-conditioned inference on top of audio2chart's Charter.

Charter.generate builds `audio_emb` once (encode -> embed -> norm -> compress)
and then passes that same tensor to transformer.forward on every decode step. So
adding a pitch bias to audio_emb once is enough to condition the whole run.

Rather than reimplement the generation loop — which would duplicate ~50 lines and
drift from upstream — this temporarily wraps transformer.forward to add the bias.
vendor/ stays untouched, which matters because audio2chart has no license and we
do not want our changes tangled up in it.
"""
import contextlib
import json
from pathlib import Path

import torch

from .conditioning import (
    PitchConditioner,
    align_frames,
    pack_features,
    windows_for_chunks,
)
from .pitch import pitch_features

ENCODEC_RATE = 24000  # Charter resamples to this internally
CHUNK_SECONDS = 30


_CHARTER_CACHE: dict[str, object] = {}


def load_charter(repo_or_dir: str):
    """Charter from a hub repo id or a local export directory, cached.

    Charter.from_pretrained only speaks hf_hub; tools/finetune.py exports to a
    local folder with the same two files, so accept both. The cache matters
    for batches: reloading the 910MB M checkpoint per song wasted 15-30s each.
    """
    key = str(repo_or_dir)
    if key in _CHARTER_CACHE:
        return _CHARTER_CACHE[key]

    from inference.engine import Charter, TransformerConfig

    path = Path(repo_or_dir)
    if not path.is_dir():
        model = Charter.from_pretrained(repo_or_dir)
    else:
        config = TransformerConfig(**json.loads(
            (path / "config.json").read_text(encoding="utf-8")))
        model = Charter(config)
        state = torch.load(path / "pytorch_model.bin", map_location="cpu")
        model.transformer.load_state_dict(state)
    _CHARTER_CACHE[key] = model
    return model


@contextlib.contextmanager
def _audio_bias(transformer, bias):
    """Add `bias` to the audio memory for the duration of the block."""
    original = transformer.forward

    def patched(input_ids, audio_emb, *args, **kwargs):
        return original(input_ids, audio_emb + bias, *args, **kwargs)

    transformer.forward = patched
    try:
        yield
    finally:
        transformer.forward = original


class PitchCharter:
    """Wraps a Charter and conditions it on pitch features.

    Holds the base model rather than subclassing it: Charter.from_pretrained is a
    classmethod that constructs its own instance, so composition is simpler than
    fighting the constructor.
    """

    def __init__(self, charter, conditioner: PitchConditioner | None = None):
        self.charter = charter
        d_model = charter.transformer.d_model
        self.conditioner = conditioner or PitchConditioner(d_model)

    @classmethod
    def from_pretrained(cls, repo_id: str, conditioner_path: str | None = None):
        from inference.engine import Charter

        model = cls(Charter.from_pretrained(repo_id))
        if conditioner_path:
            state = torch.load(conditioner_path, map_location="cpu")
            model.conditioner.load_state_dict(state)
        return model

    @property
    def config(self):
        return self.charter.config

    def pitch_bias(self, audio_path: str, n_chunks: int, frames_per_chunk: int,
                   starts, device) -> torch.Tensor:
        """[B, frames_per_chunk, d_model] bias, from the audio's pitch content."""
        import librosa

        grid_ms = self.charter.config.grid_ms
        y, sr = librosa.load(audio_path, mono=True)
        salience, f0, voiced = pitch_features(y, sr, grid_ms)
        packed = pack_features(salience, f0, voiced)
        windows = windows_for_chunks(packed, starts, frames_per_chunk,
                                     ENCODEC_RATE, grid_ms)
        return self.conditioner(windows.to(device))

    def generate(self, audio_path: str, **kwargs):
        """Same contract as Charter.generate, with pitch conditioning applied."""
        device = kwargs.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.charter.to(device)
        self.conditioner.to(device).eval()

        # Reproduce the chunk layout generate() will use, so the pitch windows
        # line up with the audio windows it encodes.
        import librosa

        duration = librosa.get_duration(path=audio_path)
        total_samples = int(duration * ENCODEC_RATE)
        chunk_samples = CHUNK_SECONDS * ENCODEC_RATE
        starts = list(range(0, total_samples, chunk_samples))
        if starts[-1] + chunk_samples > total_samples:
            starts[-1] = max(0, total_samples - chunk_samples)
        frames_per_chunk = int(CHUNK_SECONDS * 1000 / self.charter.config.grid_ms)

        with torch.no_grad():
            bias = self.pitch_bias(audio_path, len(starts), frames_per_chunk,
                                   starts, device)
            with _audio_bias(self.charter.transformer, bias):
                return self.charter.generate(audio_path, **kwargs)

    def trainable_parameters(self):
        """Only the conditioner trains; the pretrained decoder stays frozen."""
        return self.conditioner.parameters()

    def save_conditioner(self, path: str) -> None:
        torch.save(self.conditioner.state_dict(), path)
