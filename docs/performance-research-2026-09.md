# Performance research: where chartgen's time goes and how to get it back

*2026-09-07. Measured on this RTX 3060 Ti / Ryzen desktop and on the same box
with the GPU hidden; web research by five Sonnet agents (separator, Basic
Pitch, Whisper + solo detector, pipeline concurrency, weak-hardware
coverage). Numbers are for Alan Walker – Faded (216 s) unless noted; the
10-song median run is 111 s.*

## 1. Measured budget (GPU box, one song, everything sequential today)

| stage | seconds | notes |
| :-- | --: | :-- |
| BS-RoFormer SW separation (subprocess) | 39 | **~19 s is fixed overhead**: a 30 s clip costs 22 s. Fresh process each song: torch import + CUDA init 5.8 s, 700 MB checkpoint load, temp wav write/read. Actual GPU compute ≈ 0.6 s per 6.7 s chunk, batch size 1, overlap already at the minimum (2), autocast fp16 already on. |
| Solo detector (`solo.py`) | 30 | `librosa.effects.harmonic` 9.4 s + `pyin` 21.3 s on the full mix at 22.05 kHz, on CPU, run last, alone. |
| Basic Pitch on 5 stems | 16.5 | 2.8–4.1 s each. Running the five in threads: 17.1 s, i.e. **no gain** — onnxruntime already saturates every core per call. |
| Basic Pitch on the mix | 5.4 | independent of everything except loading. |
| load + beat tracking | 5.2 | independent. |
| lyrics: LRCLIB lookup + alignment | 3–8 | Whisper fallback 10 s GPU / 40 s CPU. |
| repeated librosa on the mix | 3 | `chroma_cqt` 2–3 times (2.5 s), `onset_strength` 4–5 times (0.7 s). |
| per-window prominence features | 1.7 | pyloudnorm + rms + density, 21 windows × 6 stems. |
| chart-shaping passes | ~5 | pure Python. |

CPU-only (same song): separation **1207 s**, everything else ~80 s. The
engine choice never mattered on CPU; the separator is the whole problem.

## 2. What the research settled

- **Separator knobs already maxed.** The SW config ships `num_overlap: 2`
  (the fastest legal value; >8 is a known UVR bug) and `torch.cuda.amp.autocast`
  is on. Chunk size changes trade artefacts for little speed. Mono input does
  not halve cost: the model is stereo and upmixes mono. Remaining GPU levers:
  resident model (no per-song reload), batching several chunks per forward,
  `torch.compile` regional compilation (warm-up cost, batch runs only).
- **bs-roformer-infer's API is folder-in/folder-out only**, but exposes
  `BSRoformerSession` (load once, `device=`) and `get_model_from_config()` +
  `torch.load` for an in-process loop with numpy I/O.
- **Alternatives for the same six stems.** MVSep leaderboards: BS-RoFormer SW
  is the best public guitar (9.01 SDR) and piano (7.80) model; htdemucs_6s is
  much faster on CPU but its piano stem is widely reported as poor; no
  SCNet/Apollo/BandIt 6-stem checkpoints exist. Community artefacts of *this*
  model: `studio409/BSRoformer-SW-6Stem-GGUF` for
  [BSRoformer.cpp](https://github.com/chenmozhijin/BSRoformer.cpp) (C++/GGML,
  CPU + CUDA + **Vulkan**, FP16/Q8_0/Q4 quantization, no published timings)
  and an MIT ONNX export (`elicwhite/bs-roformer-sw-6stem-onnx`) usable with
  ONNX Runtime DirectML or OpenVINO (STFT/ISTFT stay host-side; fixed sequence
  length).
- **Basic Pitch**: the CQT front end is inside the ONNX graph; windows run one
  at a time at batch 1; providers are hardcoded to CPU; `predict()` accepts a
  reusable `Model` instance (we rebuild the session every call). GPU would
  need window batching to pay off. Cheapest substitute for the feature-only
  stems (vocals/bass/piano/other feed only densities and pitch stats): librosa
  onset detection + pyin/PESTO/SwiftF0 — but the prominence model was trained
  on Basic Pitch onset densities, so swapping features means retraining it
  (the 299-song dump tooling exists).
- **Concurrency**: onnxruntime `run()` and torch CUDA ops release the GIL, so
  threads overlap CPU and GPU work in one process; Windows `spawn` re-imports
  torch per worker (~6 s + CUDA init), so use one resident process, not
  per-song subprocesses. `librosa.onset_strength`, `chroma`, `rms`,
  `spectral_centroid` all accept `S=` a precomputed spectrogram.
- **Whisper is near its floor** (CTranslate2 int8 on CPU, fp16 on GPU; turbo/
  distil are not cheaper per minute; tiny/base already degrade on singing).
  The onset-envelope alignment (3–8 s) is cheaper than any neural forced
  aligner. whisper.cpp + Vulkan is 10–12x its own CPU path on iGPUs — only
  relevant if we adopt a GGML runtime anyway.
- **Beat tracking**: madmom is broken on Python 3.13+; beat_this (ISMIR 2024)
  is the modern accurate option; not a bottleneck at 5 s.
- **Non-NVIDIA hardware** (2026): torch-directml and Microsoft DirectML are
  in maintenance mode; ONNX Runtime DirectML EP is alive; **OpenVINO is the
  only path with a shipping separator** (Intel's Demucs IR in the Audacity
  OpenVINO plugin, NNCF int8 2–3x); ROCm-on-Windows is a preview for RDNA3/4
  only; Apple Silicon: MPS breaks on Demucs-family models, MLX ports are
  ~30–60x realtime; NPUs have no documented separator runs.

## 3. Roadmap

### A. Same output, less waiting (GPU and multicore, no model changes)

*Items 1, 2 and 5 shipped 2026-09-08 (commit "Speed: resident separator, CPU
prefetch, on-disk cache"). Acceptance test: the eight comparison songs'
charts are byte-identical to the pre-change deterministic charts, cold and
warm. One landmine found on the way: setting `cudnn.benchmark` process-wide
(as the separator CLI does) changed the lyric aligner's cuDNN algorithm
choice and moved a word by 0.1 s, so the flag is now scoped to the
separation call. Measured on this box, first song of a process / later
songs / cached rerun: Faded 114 → 86 / ~80 / 29 s; The Chain 124 → 83 / – /
29 s. Items 3 (batching) and 4 are open.*

1. **Resident separator, in-process.** Load the SW model once per app session
   via the package's model API, feed numpy, keep chunk loop + fade windows.
   Saves ~19 s of the 39 s per song, more in batches. Also removes the temp
   wav round trip and lets us run **at below-normal priority** without
   starving the UI.
2. **Overlap CPU and GPU.** While the GPU separates: beat tracking (5 s),
   Basic Pitch on the mix (5 s), the LRCLIB lookup, and the solo detector's
   HPSS + pyin (30 s, needs only the mix). After stems: Basic Pitch on the
   stems overlaps the lyric alignment. Expected: ~111 s → ~60–65 s with
   identical charts. Threads suffice (GIL is released by both runtimes); keep
   the existing cancel hook per stage.
3. **Batch chunks in the separator** (4–8 chunks per forward, fp16): the
   remaining ~20 s of GPU compute should drop noticeably; measure, since the
   model is attention-heavy and may already be saturating the card.
4. **Compute each spectrogram once.** One STFT/CQT per song shared by
   `expression.sections`, `structure`, `density`, `taps`, `lyrics`: ~3 s.
5. **Stem cache on disk** keyed by (audio sha256, separator id). Shipped as
   float32 npz (~230 MB per song: float16 or FLAC would quantise the stems
   and change the transcriptions), LRU-capped at 6 GB, plus the Basic Pitch
   transcriptions as JSON. Re-charting with different options never
   separates or transcribes again (−39 s GPU, −20 min CPU).

### B. Cheaper analysis (small quality risk, measure before shipping)

6. **Solo detector off the full-mix HPSS.** The guitar stem already exists;
   pyin on the guitar stem at 11.025 kHz / hop 2048, or SwiftF0/PESTO, replaces
   `harmonic()` + pyin (30 s → a few seconds). Re-validate the 82%p rule on the
   cached solo calibration set before switching.
7. **Reuse one Basic Pitch `Model`** across the six calls; skip the vocals
   stem transcription (its class was useless in the prominence model); trim
   leading/trailing silence. A few seconds.
8. **Lazy stem transcription**: transcribe piano/other/bass only when a
   starved window follows them; prominence densities from librosa onsets —
   requires retraining `prominence_model.joblib` on the new feature (dump
   tooling exists). Saves ~10 s.

### C. Weak machines (CPU-only, iGPU, Apple Silicon)

9. **Fast preset** (auto-selected when no CUDA device): htdemucs 4-stem on
   CPU — **measured 135 s on Faded vs 1207 s for SW, 9x faster** (6 torch
   threads) — `--no-prominence`, taps via the four-stem rule,
   solos via the no-stem lead rule, Whisper base int8. Charts lose the
   guitar/piano-keyed features; that is the honest trade at 20 minutes.
10. **BSRoformer.cpp + GGUF Q8_0** as the six-stem path for CPU and for
    AMD/Intel GPUs through Vulkan: same weights, one static binary, no torch.
    Unbenchmarked publicly — worth one afternoon of measurement (quality vs
    the PyTorch stems on the 12-song stem A/B harness, speed on CPU and iGPU).
11. **OpenVINO int8 of the ONNX export** for Intel CPU/iGPU/NPU users, on
    the Intel/Audacity template. Only if 10 disappoints.
12. **Apple Silicon**: MLX port of the separator (Demucs ports exist; RoFormer
    would be new work). Defer until there is a Mac to test on.
13. Distribution: the CPU torch wheel is ~600 MB vs ~2.5 GB CUDA; a GGML
    separator plus ONNX Basic Pitch plus CTranslate2 Whisper would make the
    torch dependency optional for non-NVIDIA installs.

### Not worth doing

- Threading the five Basic Pitch calls (measured: no gain).
- Swapping Whisper models or the lyric aligner.
- Lower overlap or mono input for the separator (already minimal / no effect).
- DirectML for torch (maintenance mode), ROCm on Windows (preview).
