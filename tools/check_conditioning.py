"""Verify the pitch-conditioning plumbing before spending GPU time on it.

Two things must hold, and they are opposites, which is what makes them a real test:

  1. Zero-initialised, the conditioner is EXACTLY the identity — same tokens as
     the unmodified model, given the same seed. That proves adding it does not
     disturb the pretrained weights, so a checkpoint keeps its current quality
     and training starts from the baseline rather than from damage.
  2. With non-zero weights, the output CHANGES. That proves pitch actually
     reaches the decoder, rather than the plumbing silently dropping it.

A test that only checked (1) would pass just as well if pitch were ignored.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402  (vendor paths)
from chartgen.conditioning import PitchConditioner, align_frames, pack_features  # noqa: E402
from chartgen.model import PitchCharter  # noqa: E402

AUDIO = "work/test_rich.wav"
MODEL = "3podi/charter-v1.0-40-S-best-acc"


def tokens(model, seed):
    torch.manual_seed(seed)
    seqs = model.generate(AUDIO, temperature=0.5, top_k=32)
    return torch.cat(seqs).flatten().cpu().tolist()


def main():
    from inference.engine import Charter

    print("--- unit checks ---")
    cond = PitchConditioner(d_model=512)
    assert cond.is_identity, "fresh conditioner must be zero"
    x = torch.randn(2, 750, cond.proj.in_features)
    out = cond(x)
    assert out.shape == (2, 750, 512), out.shape
    assert torch.count_nonzero(out) == 0, "zero-init must produce an exact zero bias"
    print(f"zero-init bias is exactly zero, shape {tuple(out.shape)}  ok")

    packed = pack_features(torch.rand(100, 48), torch.rand(100), torch.ones(100))
    assert packed.shape == (100, 50), packed.shape
    assert align_frames(packed.unsqueeze(0), 120).shape == (1, 120, 50)
    assert align_frames(packed.unsqueeze(0), 80).shape == (1, 80, 50)
    print("pack_features / align_frames  ok")

    print("\n--- generation: zero-init must be a no-op ---")
    base = Charter.from_pretrained(MODEL)
    plain = tokens(base, seed=1234)

    conditioned = PitchCharter(base)
    assert conditioned.conditioner.is_identity
    same = tokens(conditioned, seed=1234)
    identical = plain == same
    print(f"baseline tokens      : {len(plain)}")
    print(f"zero-init conditioned: {len(same)}  identical={identical}")
    if not identical:
        diff = sum(1 for a, b in zip(plain, same) if a != b)
        print(f"  MISMATCH in {diff} positions — plumbing perturbs the base model")
        return 1

    print("\n--- generation: trained weights must change the output ---")
    torch.manual_seed(0)
    with torch.no_grad():
        # Stand in for a trained conditioner: any non-zero projection will do.
        torch.nn.init.normal_(conditioned.conditioner.proj.weight, std=0.5)
    assert not conditioned.conditioner.is_identity
    changed = tokens(conditioned, seed=1234)
    differs = sum(1 for a, b in zip(plain, changed) if a != b)
    print(f"non-zero conditioned : {differs} of {len(plain)} positions differ")
    if differs == 0:
        print("  FAILED: pitch has no effect — the bias is not reaching the decoder")
        return 1

    print("\nBoth hold: the conditioner is a no-op untrained, and pitch reaches "
          "the decoder once it has weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
