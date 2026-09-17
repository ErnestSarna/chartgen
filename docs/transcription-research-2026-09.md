# Transcription research, September 2026

Research only. Nothing in the pipeline was changed. The question: the tool
has shipped, the transcription stage is still the thing holding features
back (swing detection, riff repetition, chugs, missing notes, instrument
confusion, sustains). What could replace or improve it?

Two sources of evidence went into this:

1. **Local measurements** on the song library, using the Harmonix-style
   multitrack folders (an isolated `guitar.opus` plus an official Rock Band
   chart; 61 have a guitar stem, 40 have a full stem set to sum into a mix)
   and the community `.chart` songs. The isolated stems let us measure
   Basic Pitch's own ceiling with *perfect* separation, independent of
   BS-RoFormer. Scripts are in `tools/bp_research/`
   (`bp_sweep.py`, `bp_stability.py`, `bp_ensemble.py`, `bp_speedup.py`,
   `bp_timing.py`, `bp_summary.py`); they read the library and write to
   `out/bp_research/` (git-ignored; the sweep's raw JSONL and the web pass's
   report and extracted claims are kept there too), nothing else.
2. **A 104-agent web research pass** (search, fetch, claim extraction,
   3-vote adversarial verification) over replacement models, guitar-specific
   transcription, Basic Pitch tuning, onset/beat trackers and separation.
   Verified claims are marked as such below; claims the pass extracted but
   did not get to verify are marked *unverified*.

Metric used throughout: **coverage** of the human Expert chart, i.e. the
fraction of human note positions that have *some* transcribed onset within
±80 ms (recall80), plus precision80 (fraction of our onsets within 80 ms of a
human note). A human chart is a stylistic selection, not ground truth, so
precision is expected to be well under 1; recall80 is the "can the
transcription even see this note" ceiling that everything downstream
(selection, density, motifs) is bounded by.

---

## 1. The headline local finding: a 128 ms minimum-note filter we never set

`chartgen/transcribe.py` calls `basic_pitch.inference.predict(path)` with
library defaults. One of those defaults is `minimum_note_length = 127.7 ms`
(11 frames at 86 fps). In `note_creation.output_to_notes_polyphonic` any
note whose frame activation dies out within 11 frames of its onset is
**discarded before it is ever returned** (`if i - note_start_idx <= min_note_len: continue`).
Every cached transcription confirms it: the shortest note in any cached
event list is exactly 0.129 s.

What that means musically:

| grid          | 128 ms is longer than one slot above |
|---------------|--------------------------------------|
| 16th notes    | 117 BPM                              |
| 8th triplets  | 156 BPM                              |
| 8th notes     | 234 BPM                              |

Sixteenth-note passages above 117 BPM only survive when the *frame*
activation stays lit past the next onset (ringing chords, sustained lead);
anything staccato, palm-muted or damped is removed. That is precisely the
chug / fast-run / "missing positions not lanes" symptom. It is not a model
limitation; it is a decode parameter.

Measured on isolated guitar stems (official chart as reference), default
decode "D" (0.5 / 0.3 / 127.7 ms) against "B" (0.5 / 0.3 / 23 ms):

| song (isolated guitar stem)      | recall80 D | recall80 B | precision80 D | precision80 B |
|----------------------------------|-----------:|-----------:|--------------:|--------------:|
| Van Halen – Panama               | 0.41       | 0.59       | 0.64          | 0.62          |
| Metallica – Enter Sandman        | 0.73       | 0.90       | 0.65          | 0.63          |
| Judas Priest – Halls of Valhalla | 0.72       | 0.94       | 0.76          | 0.75          |
| The Red Chord – Hour of Rats     | 0.35       | 0.66       | 0.74          | 0.80          |
| Ozzy Osbourne – Miracle Man      | 0.46       | 0.73       | 0.69          | 0.68          |
| Creed – Higher                   | 0.85       | 0.95       | 0.74          | 0.72          |

Coverage rises 10 to 31 points at essentially unchanged precision. The
library-wide sweep (section 2) shows the same pattern across genres.

**This is the single cheapest, highest-yield change available**, and it is
zero new dependencies. The caveat is real, though: the selection stage
(`expert_from_notes`, density, playability, motif code) was calibrated on
event lists that never contained a note under 129 ms. Relaxing the filter
roughly doubles the raw event count (short notes become 30 to 60 percent of
events), so it is a recalibration project for the downstream stages, not a
one-line flip. The right first step is an A/B of the full pipeline on the
calibration set with `minimum_note_length` at 23 ms, 35 ms and 58 ms, scoring
against human charts with `tools/evaluate.py` and the paired per-song test.

The other decode parameters matter less: dropping `onset_threshold` to 0.3
or 0.2 buys recall only by flooding precision (config F reaches recall ≈ 1.0
at precision 0.3 to 0.6); `frame_threshold` interacts with note length only.

---

## 2. Library-wide coverage sweep

62 songs with human Expert charts: 40 Harmonix-style multitrack folders
(isolated guitar stem + summed-stem mix, official chart) and 22 community
`.chart` songs (mix only). One Basic Pitch model pass per source, then six
decodes plus two ceilings. Medians across songs; "wins" counts songs where
recall80 improved by more than 0.01 over the default decode D.

Decodes: D = 0.5 / 0.3 / 127.7 ms (today), A = 0.5 / 0.3 / 58 ms,
B = 0.5 / 0.3 / 23 ms, C = 0.3 / 0.3 / 58 ms, E = 0.3 / 0.2 / 23 ms,
F = 0.2 / 0.2 / 23 ms (onset threshold / frame threshold / minimum note
length). "onset peaks ≥ 0.1" = local maxima of the raw onset posteriorgram
at any pitch; "SuperFlux" = librosa spectral-flux onsets on the same audio.

**Isolated guitar stem, official charts (n = 40)**

| decode            | recall50 | recall80 | precision80 | onsets per human note | median \|err\| | wins  |
|-------------------|---------:|---------:|------------:|----------------------:|---------------:|------:|
| D (today)         | 0.665    | 0.758    | 0.739       | 1.4                   | 17.6 ms        | –     |
| A (58 ms)         | 0.839    | 0.921    | 0.714       | 2.0                   | 15.4 ms        | 40/40 |
| B (23 ms)         | 0.874    | 0.943    | 0.708       | 2.4                   | 14.1 ms        | 40/40 |
| C (onset 0.3)     | 0.901    | 0.959    | 0.592       | 3.3                   | 14.8 ms        | 40/40 |
| F (0.2/0.2/23)    | 0.990    | 0.998    | 0.561       | 7.8                   | 8.0 ms         | 40/40 |
| onset peaks ≥ 0.1 | 0.931    | 0.997    | 0.528       | 3.9                   | 18.7 ms        | 40/40 |
| SuperFlux         | 0.496    | 0.626    | 0.947       | 0.7                   | 34.3 ms        | 8/40  |

Paired per-song gain in recall80, B versus D: median +0.15, interquartile
+0.09 to +0.23, positive on every song.

**Mix, same 40 songs**: D 0.746 → A 0.888 → B 0.915 recall80, precision80
0.651 → 0.639 → 0.632.

**Mix, 22 community charts**: D 0.585 → A 0.733 → B 0.780 → C 0.866
recall80, precision80 0.502 → 0.496 → 0.496 → 0.444. Community charts sit
lower because they follow synths and vocals more freely; the worst cases
(Tristam, Grandson, Kid Cudi: EDM and hip-hop) start at 0.31 to 0.42 and
gain 20 points.

Three conclusions the sweep supports beyond section 1:

- **The network sees essentially every human note.** Raw onset-posteriorgram
  peaks at threshold 0.1 cover 99.5 percent or more of human positions on
  both stems and mixes. Every miss chartgen suffers today is introduced in
  the decode (length filter, greedy note assignment, thresholds), not in the
  CNN. Replacing the network cannot raise coverage; it can only raise
  precision, timing and instrument attribution.
- **Separation is not the coverage bottleneck.** On the same 40 songs the
  perfect guitar stem beats the mix by a median of only +0.03 recall80 at D
  and +0.004 at B (stem better on 17 songs, mix better on 17). Stems earn
  their cost for *which instrument* and for precision, not for seeing notes.
  A better guitar separator would not fix missing notes.
- **The remaining hard cases are decode-limited too, but not by length.**
  After B, the low tail on isolated stems is funk and fast metal: Kung Fu
  Fighting 0.40 (even F reaches only 0.65: the chart follows a part the
  stem barely carries), Fortunate Son 0.51, Panama 0.59, Hour of Rats 0.66.
  Those are where a guitar-trained model (section 6) or the half-speed pass
  (section 3) would have to earn their keep.

---

## 3. Half-speed transcription helps distorted guitar beyond the filter fix

Playing the guitar stem back at half speed (same samples, half the sample
rate: one octave down, twice as slow, no phase vocoder) and mapping times ×2
afterwards:

| song (isolated guitar stem) | 1x D | 0.5x D | 1x B | 0.5x B |
|-----------------------------|-----:|-------:|-----:|-------:|
| Panama                      | 0.41 | 0.54   | 0.59 | 0.71   |
| Enter Sandman               | 0.73 | 0.80   | 0.90 | 0.90   |
| Halls of Valhalla           | 0.72 | 0.90   | 0.94 | 0.98   |
| Hour of Rats                | 0.35 | 0.50   | 0.66 | 0.69   |
| Miracle Man                 | 0.46 | 0.61   | 0.73 | 0.79   |
| Creed – Higher              | 0.85 | 0.87   | 0.95 | 0.94   |

Most of the gain at D is the minimum-note filter again (notes are twice as
long). But at B there is still +4 to +12 points on the metal songs, because
half speed doubles the effective frame rate (172 fps) and moves fast low
chugs into a register the CQT resolves better. Precision drops 3 to 8
points. Double speed (octave up) is strictly worse. A union of 1x and 2x
gains a little recall at a little precision cost; not worth it.

Verdict: a candidate second pass for guitar-followed windows only (cost: one
extra Basic Pitch call on the guitar stem, about 4 s CPU), to be tested
after the filter fix, since the two overlap.

---

## 4. Riff instability: Basic Pitch is ~10 percent non-deterministic under alignment

Shifting the same guitar stem by 50 ms of leading silence (which only moves
Basic Pitch's internal 2 s window boundaries) changes the onset set:

| song            | onset agreement, 50 ms shift | 130 ms | 1000 ms | pitch agreement |
|-----------------|-----------------------------:|-------:|--------:|----------------:|
| Enter Sandman   | 0.91                         | 0.89   | 0.87    | 0.95            |
| Panama          | 0.88                         | 0.88   | 0.76    | 0.98            |
| Mr. Brightside  | 0.91                         | 0.92   | 0.91    | 0.95            |
| Learn to Fly    | 0.94                         | 0.93   | 0.91    | 0.93            |
| Say It Ain't So | 0.94                         | 0.93   | 0.93    | 0.94            |
| Creed – Higher  | 0.93                         | 0.92   | 0.91    | 0.94            |

So 6 to 16 percent of notes appear or vanish depending on where a riff falls
relative to the window grid, and about 5 percent of matched notes change
pitch. Two identical bars in a song will land at different window offsets
almost always. This accounts for part, not all, of the 80 to 90 percent
per-bar position variance measured earlier (the rest is acoustic: vocals,
fills, mix changes between passes).

Averaging posteriorgrams over three window shifts (test-time augmentation)
raised agreement only from ~0.93 to ~0.95 and did not change coverage.
Verdict: **park**. The instability is mostly in the greedy note decode, not
the network, so the fix is downstream: riff unification by consensus (which
already ships) is the right tool, and the decoder relaxation in section 1
should reduce flicker because fewer notes sit on the length threshold.

---

## 5. Onset timing precision: the jitter is Basic Pitch's, and cheap fixes fail

Against official charts (which are hand-placed to the recording on an
accurate tempo map, so they are a fair timing reference to within a few ms),
after removing each detector's constant bias:

| song (guitar stem)   | BP notes: jitter p50 / p90 | ≤10 ms | SuperFlux p50 / p90 | SuperFlux recall80 |
|----------------------|---------------------------:|-------:|--------------------:|-------------------:|
| Learn to Fly         | 7 / 20 ms                  | 63 %   | 11 / 29 ms          | 0.53               |
| Mr. Brightside       | 12 / 34 ms                 | 45 %   | 13 / 31 ms          | 0.16               |
| Creed – Higher       | 12 / 38 ms                 | 45 %   | 12 / 33 ms          | 0.33               |
| Enter Sandman        | 20 / 50 ms                 | 27 %   | 16 / 53 ms          | 0.56               |
| Panama               | 24 / 60 ms                 | 24 %   | 11 / 28 ms          | 0.26               |
| Halls of Valhalla    | 24 / 69 ms                 | 22 %   | 37 / 105 ms         | 0.17               |

Findings:

- Basic Pitch's onsets are early by a constant 2 to 23 ms per song (the
  per-pitch onset peak leads the perceived attack on distorted chords).
  That bias is correctable per song, and correcting it alone would move a
  quarter of notes into the ±10 ms band.
- Residual jitter is 7 to 24 ms median, 20 to 70 ms p90. A 16th slot is
  separated from a triplet slot by ~55 ms at 120 BPM, so swing detection
  needs p90 well under 25 ms. Basic Pitch does not deliver that on rock
  material and no decode trick changes it: parabolic sub-frame refinement
  on the onset posteriorgram did nothing (the posteriorgram itself is
  smeared over 2 to 3 frames), and snapping to SuperFlux onsets did nothing.
- SuperFlux (librosa, spectral flux) on the isolated stem has *similar*
  jitter and far worse recall on these stems (0.16 to 0.56), so "pitch from
  Basic Pitch, timing from a flux detector" is dead on arrival for guitar.
  The dedicated neural onset detectors (madmom CNN) that would be better
  are CC BY-NC-SA weights (section 7), which rules them out.
- Beat grid: the other half of the swing problem is librosa's ±40 ms beat
  phase. Beat This! (section 7) is the credible fix for that half and is
  MIT code and weights; it does not need madmom.

Verdict on symptom 1 (swing): fix the beat grid with Beat This!, correct
the per-song onset bias, and re-measure. If p90 jitter is still above
~25 ms, swing stays opt-in. Do not expect a transcriber swap to fix timing:
none of the candidates below is evaluated tighter than ±50 ms.

---

## 6. Web research: replacement and complement models

Verified by primary source unless stated. Numbers are the papers' own.

### Ranked shortlist

1. **Keep Basic Pitch, fix the decode** (section 1). Zero cost, zero
   license change, largest measured effect. Do first; every other
   comparison should be re-based on it.
2. **Beat This!** (CPJKU, ISMIR 2024) for the beat grid. MIT code *and*
   weights (verified in the README), pure PyTorch 2.x, no TensorFlow, 78 MB
   main / 8 MB small checkpoint, CPU or GPU, DBN post-processing optional so
   madmom is not needed. Addresses the beat-phase half of the timing
   problem and would make the sync track more trustworthy for every
   downstream rule. Cheap experiment: run it on the 61 official charts and
   compare beat-phase error against their tempo maps versus librosa.
3. **TimbreAMT** (Li & Zhu, TISMIR 2026, github.com/madderscientist/timbreAMT).
   The only license-clean, drop-in-shaped alternative: Apache-2.0 code and
   checked-in ONNX weights, ~65 k parameters, waveform in / 84-bin onset and
   frame posteriorgrams out, decoded by the *same* `output_to_notes_polyphonic`
   we already use, and faster than Basic Pitch on CPU (17 s vs 39 s for a
   301 s duet in their in-browser test). Caveats verified: it only *ties* a
   focal-loss Basic Pitch on classical data (Bach10 75.2 vs 75.2 note F1,
   URMP 71.4 vs 71.5), has no guitar evaluation at all, and the code is
   notebook-level. Its second branch produces a per-note timbre embedding
   that clusters notes by instrument, which is a possible instrument-
   commitment signal. Cheap experiment: run its ONNX on the same 40 stems
   and mixes with `bp_sweep.py`; adopt only if coverage and timing match or
   beat Basic Pitch at config B.
4. **A guitar-specialised retrain of a Basic-Pitch-sized model** (medium
   term). Three independent papers agree the distorted-guitar failure is a
   training-data problem, not an architecture problem: the Tone-informed
   Transformer work (arXiv 2504.07406) gets 84 / 82 / 79 onset F1 on
   low-gain / crunch / high-gain re-amped guitar versus 74 / 48 / 43 for a
   zero-shot MAESTRO hFT-Transformer, and a plain hFT trained on the same
   tone-diverse data lands within ~4 points; the DAFx 2024 TabCNN study
   lifts multi-pitch F1 on real electric recordings from 0.64 to 0.74 just
   by adding real-amp and effects renders to the training data; naive
   fine-tuning of a MAESTRO model made low-gain *worse*. Redistributable
   data exists: **Guitar-TECHS** (ICASSP 2025, CC BY 4.0, Zenodo 14963133,
   5.2 h, three players, DI + amp + mics, per-string hexaphonic MIDI, but
   flat-EQ clean tones and up to 100 ms label misalignment to fix) plus
   synthetic amp and effect renders. Training code: `basic-pitch-torch`
   (gudgud96, PyTorch port with converted Spotify weights, GPU-able, exposes
   the posteriorgrams; license not verified) or TimbreAMT's training code.
   This is weeks of work and is the only path that moves the metal-song
   ceiling (Panama 0.59, Hour of Rats 0.66 on a *perfect* stem even after
   the filter fix).
5. **MT3 family via `mt3-infer`** (openmirlab, MIT wrapper, Nov 2025 /
   v0.2.0 Jul 2026), specifically **MR-MT3** (Sony, MIT code, HF weights,
   176 MB, ~57× real time on a 4090) or **YourMT3+** (Apache-2.0 in the
   HF Space release that mt3-infer vendors; the author's GitHub repo is
   GPL-3.0; 536 MB, 30 to 46 M params, ~36× real time fp16 on a T4). These
   give per-instrument note streams, which is the attractive part for
   instrument confusion. Verified limits: gains over MT3 are small (MR-MT3
   Slakh onset F1 63 → 67; YourMT3+ beats the strongest prior baseline on
   GuitarSet by ~0.5 point, and the "substantially beats MT3 on Slakh"
   claim was refuted), program-level F1 stays under 40 percent, Slakh
   domain overfit is documented (19 to 24 F1 on NSynth), nothing is
   evaluated on distorted guitar, CPU cost is unreported anywhere, and the
   MT3-PyTorch backend has no license at all. Verdict: **test, low
   priority**, as an instrument-labelling oracle rather than a note source.

### Everything examined

| candidate | license (code / weights) | verdict | reason |
|-----------|--------------------------|---------|--------|
| Basic Pitch decode parameters | Apache-2.0 | **adopt (test)** | +10 to +31 coverage on stems, measured locally |
| Half-speed second pass on guitar stem | – | test | +4 to +12 on metal after filter fix, measured locally |
| Window-shift posteriorgram ensemble | – | park | +0.02 stability, no coverage change, 3× cost |
| Beat This! | MIT / MIT | **test** | fixes beat-phase half of timing; no madmom |
| TimbreAMT | Apache-2.0 / Apache-2.0 (ONNX) | test | drop-in shape, faster, but only ties BP on classical, no guitar eval |
| Guitar-specialised retrain (Guitar-TECHS + amp renders) | CC BY 4.0 data | plan | only path past the distorted-guitar ceiling; weeks |
| MR-MT3 via mt3-infer | MIT / HF (MIT) | test, low | instrument streams; modest gains; Slakh overfit; CPU unknown |
| YourMT3+ via mt3-infer | Apache-2.0 (Space) / same; GitHub GPL-3.0 | test, low | as above; licence depends on which release is vendored |
| MT3-PyTorch backend | none | reject | no license upstream |
| Harmonica (BandLab, arXiv 2609.04640, Sep 2026) | none released | watch | largest claimed gains over BP (GuitarSet OnP .78 → .90, OnPOff .55 → .76 → better offsets); no code, weights or license yet; 12-day-old preprint |
| Timbre-Trap (Sony) | MIT / HF | park | URMP-trained classical, frame-level multipitch only |
| hFT-Transformer, Transkun, ByteDance piano | MIT | reject | piano-only training; distorted-guitar zero-shot collapse documented (74 → 43 F1) |
| madmom CNN onset / RNN-DBN beat | BSD / **CC BY-NC-SA 4.0** | reject | weights non-commercial; PyPI 0.16.1 (2018) does not import on Python ≥ 3.10 (`collections.MutableSequence`), issue open since 2022 |
| BeatNet | CC BY 4.0 | reject | hard dependency on madmom 0.16.1 |
| librosa SuperFlux onsets as timing source | ISC | reject | measured: similar jitter, 0.16 to 0.56 recall on stems |
| "Snapping Matters" onset refinement (arXiv 2606.11903) | paper | park | per-pitch bipartite peak assignment on the onset posteriorgram; our posteriorgram is too smeared for sub-frame gains (parabolic test) |
| Guitar velocity prediction (QMUL, arXiv 2606.24912) | none | reject | validated on synthetic renders only; no real-guitar velocity labels exist; no release |
| Newer guitar-stem separators (ZFTurbo zoo, MVSEP) | per checkpoint | none found | *unverified*: the zoo lists only HTDemucs 6-stem with a guitar stem and no SDR for it; no RoFormer/SCNet guitar checkpoint documented |
| Jointist / Cerberus joint separation+transcription | research code | reject | Slakh-synthetic, no distorted guitar, not packaged |
| Mel-RoFormer fine-tuned for transcription (ByteDance, ISMIR 2024) | paper | park | the "separation backbone → transcription head" recipe is the modern route if we ever fine-tune the SW model; no guitar release |
| Tone-informed Transformer / EGDB-PG | none | evidence only | code, weights and data not released |
| TabCNN / FretNet / Fretting-Transformer | mixed | reject | solve string/fret disambiguation we do not need; GuitarSet-only training |
| Commercial (Moises, Klangio, AnthemScore, Melodyne, Logic) | proprietary | reject | not local or not redistributable |
| Large audio LLMs as oracle | API | reject | not local; no tick-level output |

### Two things the web pass turned up that are not about transcription

- **BS-RoFormer SW weight provenance** (*unverified* beyond what the agents
  fetched): the original Hugging Face repo `jarredou/BS-ROFO-SW-Fixed`
  returns 401 and the account appears deleted; community mirrors serve the
  byte-identical 699 MB checkpoint but disagree on its license ("unknown"
  vs "mit"), and no model card states training data or SDR.
  `bs-roformer-infer` (MIT) downloads it from the TRvlvr/model_repo GitHub
  release used by UVR. This does not change anything today, but the README
  should not imply the *weights* are MIT-licensed, and a fallback download
  location is worth having.
- **Evaluation tolerance.** Every published transcription number, including
  Spotify's, is at ±50 ms onset tolerance (confirmed in spotify/basic-pitch
  issue 96). Nothing in the literature speaks to tick-level timing, which is
  why section 5 had to be measured locally.

---

## 7. Ideas rejected as highly uncertain (kept brief)

- **Chord-recognition models as a chord-vs-single oracle** (madmom chord is
  NC-licensed; Chordino/BTC are beat-level, far too coarse for chug ticks).
- **Drum-transcription-locked grids**: would inherit drum-stem transcription
  errors and only helps where guitar follows the kick; Beat This! covers the
  useful part.
- **Community tab / Guitar Pro / Lakh MIDI as a prior via alignment**:
  coverage is spotty, licensing of scraped tabs is worse than the retired
  neural charter's, and alignment to a wrong tab is worse than no tab.
- **A learned correction model from (Basic Pitch output → human chart)**:
  technically distinct from the retired end-to-end charter but carries the
  same data-statement problem for community charts; the 61 official
  Harmonix charts are too few and not redistributable as training data.
- **Transcribe once per repeated segment and copy**: riff unification by
  consensus already does the useful version of this; the acoustic variation
  between passes is real, so copying would erase legitimate differences.

---

## 8. Suggested order of work (none started)

1. A/B `minimum_note_length` 23 / 35 / 58 ms through the full pipeline on
   the calibration set; recalibrate density and chord gates as needed.
2. Per-song onset bias correction (median of Basic Pitch onset minus nearest
   beat-grid slot, or minus SuperFlux onset where both fire) before
   quantisation.
3. Beat This! beat grid behind a flag; measure beat-phase error on the 61
   official tempo maps; re-test swing votes.
4. Half-speed second pass on guitar-followed starved windows.
5. TimbreAMT ONNX through `bp_sweep.py` on the same 40 stems.
6. If and only if the metal ceiling still matters after 1 to 4: plan the
   Guitar-TECHS + synthetic-amp retrain of a Basic-Pitch-sized model.
