"""Pitch conditioning for the audio2chart transformer.

`tools/pitch_probe.py` showed the model is pitch-blind (-0.04 correlation on a
falling scale) while `chartgen/pitch.py` extracts pitch at r=1.0. This is the
bridge: project the pitch features to d_model and add them to the audio memory
the decoder cross-attends to.

Why *add* to audio_emb rather than concatenate or widen the encoder:

- audio_emb is already [B, T', d_model], and at compression 3 a 30s window gives
  750 frames — the same 40ms grid the pitch features use. They line up 1:1 with no
  resampling.
- The projection is **zero-initialised**, so an untrained conditioner is exactly
  the identity. The pretrained checkpoints load unchanged and the model starts at
  its current quality rather than from a damaged input layer, then learns to use
  pitch. Concatenation would have changed tensor shapes and forced
  reinitialisation.
- If pitch turns out not to help, the projection simply stays near zero, so the
  experiment cannot make things worse than the baseline.
"""
import torch
import torch.nn as nn

from .pitch import N_BINS

# salience bins + dominant pitch + voiced flag
N_PITCH_FEATURES = N_BINS + 2


class PitchConditioner(nn.Module):
    """Projects per-frame pitch features into the audio memory.

    Zero-init is load-bearing, not tidiness: it makes adding this module a no-op
    at step 0, which is what lets a pretrained checkpoint keep its quality.
    """

    def __init__(self, d_model: int, n_features: int = N_PITCH_FEATURES,
                 dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.proj = nn.Linear(n_features, d_model)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        """[B, T, n_features] -> [B, T, d_model] bias to add to audio_emb."""
        return self.drop(self.proj(self.norm(pitch)))

    @property
    def is_identity(self) -> bool:
        """True while the projection is still all zeros (i.e. untrained)."""
        return bool(torch.all(self.proj.weight == 0) and torch.all(self.proj.bias == 0))


def pack_features(salience, f0, voiced) -> torch.Tensor:
    """Stack chartgen.pitch outputs into one [T, N_PITCH_FEATURES] tensor."""
    salience = torch.as_tensor(salience, dtype=torch.float32)
    f0 = torch.as_tensor(f0, dtype=torch.float32).reshape(-1, 1)
    voiced = torch.as_tensor(voiced, dtype=torch.float32).reshape(-1, 1)
    if not (salience.shape[0] == f0.shape[0] == voiced.shape[0]):
        raise ValueError(f"frame mismatch: {salience.shape[0]}, {f0.shape[0]}, "
                         f"{voiced.shape[0]}")
    return torch.cat([salience, f0, voiced], dim=-1)


def align_frames(pitch: torch.Tensor, n_frames: int) -> torch.Tensor:
    """Trim or edge-pad a [.., T, F] tensor to exactly n_frames.

    Encodec's frame count for a window and our CQT frame count are computed by
    different code paths, so they can disagree by a frame or two at the edges.
    Padding with the last frame is better than zeros, which would read as
    "silence" and fight the voiced flag.
    """
    frames = pitch.shape[-2]
    if frames == n_frames:
        return pitch
    if frames > n_frames:
        return pitch[..., :n_frames, :]
    pad = pitch[..., -1:, :].expand(*pitch.shape[:-2], n_frames - frames,
                                    pitch.shape[-1])
    return torch.cat([pitch, pad], dim=-2)


def windows_for_chunks(pitch: torch.Tensor, starts, frames_per_chunk: int,
                       sample_rate: int, grid_ms: int) -> torch.Tensor:
    """Slice a whole-song [T, F] tensor into a [B, frames_per_chunk, F] batch.

    `starts` are the audio-sample offsets Charter.generate uses to cut 30s chunks,
    so each converts to a frame index exactly: samples -> seconds -> frames.
    """
    out = []
    for start in starts:
        begin = int(round(start / sample_rate * 1000.0 / grid_ms))
        window = pitch[begin:begin + frames_per_chunk]
        out.append(align_frames(window.unsqueeze(0), frames_per_chunk))
    return torch.cat(out, dim=0)
