"""Followed-instrument timeline: which stem a charter would follow, per
4-bar window (the Route B prominence prior).

Why a model and not a rule: the 95-song ownership study measured stem
ENERGY and found it predicts nothing about which line humans chart, and
the 299-song Route B dump (2026-09-05) confirmed it - the loudest stem by
energy names the followed instrument in 32% of contested windows, by
K-weighted loudness 35%. But a small gradient-boosting model on the same
level shares reaches 81%, and 91% once per-stem onset density joins in
(92.4% over all labeled windows, song-level 5-fold, precision 96% while
committing on 90% of windows). Labels came from onset matching between
each SW stem's transcription and the human chart, per-song offset
corrected, coincidence split (tools/prominence_pilot.py); the model is
trained by tools/prominence_train.py --save and stored next to this file.

Runtime cost on top of the SW separation every song already pays: four
extra stem transcriptions (~7 s each on CPU; the guitar stem's is reused
when the chord-texture pass already made it) and the features below.

Consumers read the timeline as merged RUNS of consecutive same-stem
windows: taps decide per run, note selection may commit per run. The
window is a 4-bar unit on the chart's own beat grid, the granularity the
charters' switches happen at; single-window blips between two agreeing
neighbours are smoothed away first.
"""
from pathlib import Path

import numpy as np

STEMS = ("vocals", "bass", "guitar", "piano", "sw_other")
FEATS = ("energy", "loudness", "density", "pitch", "mono")
LETTER = {"vocals": "V", "bass": "B", "guitar": "G", "piano": "P", "sw_other": "S"}
BARS_PER_WINDOW = 4
MODEL_PATH = Path(__file__).resolve().parent / "prominence_model.joblib"
# Windows below this confidence carry no commitment: consumers treat them
# as "no opinion" (taps fall back to the keyed-share rule, selection stays
# loudness-following). 0.6 sits where held-out precision is ~96%.
MIN_CONFIDENCE = 0.60

_MODEL = None


def window_features(mono, sr, meter, stem_events, t0, t1):
    """Per-stem features of one (t0, t1) span. IDENTICAL to the training
    extractor (tools/prominence_pilot.py imports this): energy share and
    K-weighted loudness share over the six true stems, onset density,
    median pitch and monophonic share from the stem's transcription."""
    a, b = int(t0 * sr), int(t1 * sr)
    feats = {}
    energy, loud = {}, {}
    for name in STEMS + ("drums",):
        seg = mono[name][a:b]
        if len(seg) < sr // 2:
            return None
        energy[name] = float(np.sqrt((seg ** 2).mean()))
        try:
            lufs = meter.integrated_loudness(seg.astype(np.float64))
            loud[name] = 10 ** (lufs / 10) if np.isfinite(lufs) else 0.0
        except Exception:
            loud[name] = 0.0
    et, lt = sum(energy.values()), sum(loud.values())
    for name in STEMS:
        ev = [(s, p) for s, e, p, amp in stem_events[name] if t0 <= s < t1 and amp >= 0.20]
        by = {}
        for s, p in ev:
            by.setdefault(round(s, 2), []).append(p)
        feats[name] = {
            "energy": energy[name] / et if et else 0.0,
            "loudness": loud[name] / lt if lt else 0.0,
            "density": len(by) / (t1 - t0),
            "pitch": float(np.median([p for _, p in ev])) if ev else 0.0,
            "mono": (sum(1 for v in by.values() if len(v) == 1) / len(by)) if by else 0.0,
        }
    feats["drums_energy"] = energy["drums"] / et if et else 0.0
    return feats


def feature_vector(feats, prev_feats, next_feats):
    """Model input for one window: its own features plus the energy and
    loudness shares of the neighbouring windows (zeros at the edges)."""
    x = []
    for s in STEMS:
        x += [feats[s][k] for k in FEATS]
    x.append(feats["drums_energy"])
    for ctx in (prev_feats, next_feats):
        if ctx:
            x += [ctx[s][k] for s in STEMS for k in ("energy", "loudness")]
        else:
            x += [0.0] * (2 * len(STEMS))
    return x


def load_model():
    global _MODEL
    if _MODEL is None and MODEL_PATH.is_file():
        import joblib

        _MODEL = joblib.load(MODEL_PATH)
    return _MODEL


def windows_for(tempo, duration_s: float):
    """4-bar windows on the chart's beat grid, as (t0, t1, beat0, beat1)."""
    step = BARS_PER_WINDOW * 4
    out = []
    beat = 0
    while True:
        t0 = tempo.beat_to_time(beat)
        if t0 >= duration_s:
            break
        t1 = min(duration_s, tempo.beat_to_time(beat + step))
        if t1 - t0 >= 1.0:
            out.append((t0, t1, beat, beat + step))
        beat += step
    return out


def transcribe_stems(mono, sr, progress=lambda m: None, known=None):
    """Basic Pitch on each true melodic stem; `known` supplies already
    transcribed stems (the chord-texture pass makes the guitar's)."""
    import tempfile

    import soundfile as sf

    from . import transcribe

    out = dict(known or {})
    for name in STEMS:
        if name in out:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
            path = fh.name
        try:
            sf.write(path, mono[name], sr)
            out[name] = transcribe.transcribe(path)
        finally:
            Path(path).unlink(missing_ok=True)
    return out


def smooth(labels):
    """A single window disagreeing with two agreeing neighbours takes
    their label: charters switch instruments at phrase boundaries, not
    for one bar."""
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i - 1] == out[i + 1] != out[i]:
            out[i] = out[i - 1]
    return out


def runs(windows):
    """Merge consecutive same-stem windows: [(t0, t1, beat0, beat1, stem,
    mean confidence)]. Windows without an opinion (stem None) form their
    own runs so consumers can fall back there."""
    out = []
    for w in windows:
        if out and out[-1]["stem"] == w["stem"]:
            last = out[-1]
            last["t1"], last["beat1"] = w["t1"], w["beat1"]
            last["conf"].append(w["conf"])
        else:
            out.append({"t0": w["t0"], "t1": w["t1"], "beat0": w["beat0"], "beat1": w["beat1"],
                        "stem": w["stem"], "conf": [w["conf"]]})
    for r in out:
        r["conf"] = float(np.mean(r["conf"]))
    return out


def letters(windows) -> str:
    return "".join(LETTER.get(w["stem"], "?") if w["stem"] else "?" for w in windows)


def timeline(audio_path, tempo, duration_s, progress=lambda m: None, known_events=None):
    """Per-window followed stem with confidence, or None when the model or
    the SW stems are unavailable. Each entry: {t0, t1, beat0, beat1, stem,
    conf, probs}; stem is None below MIN_CONFIDENCE."""
    model = load_model()
    if model is None:
        return None
    from . import stems as stemsmod

    mono = stemsmod.separate(str(audio_path), progress, backend="sw")
    if not mono or any(k not in mono for k in STEMS + ("drums",)):
        return None
    import pyloudnorm as pyln

    meter = pyln.Meter(stemsmod.SR)
    stem_events = transcribe_stems(mono, stemsmod.SR, progress, known_events)
    spans = windows_for(tempo, duration_s)
    feats = [window_features(mono, stemsmod.SR, meter, stem_events, t0, t1) for t0, t1, _, _ in spans]
    rows, keep = [], []
    for i, f in enumerate(feats):
        if f is None:
            continue
        prev = feats[i - 1] if i > 0 else None
        nxt = feats[i + 1] if i + 1 < len(feats) else None
        rows.append(feature_vector(f, prev, nxt))
        keep.append(i)
    if not rows:
        return None
    clf = model["model"]
    proba = np.zeros((len(rows), len(STEMS)))
    p = clf.predict_proba(np.array(rows))
    proba[:, list(clf.classes_)] = p
    out = []
    labels = []
    for j, i in enumerate(keep):
        t0, t1, b0, b1 = spans[i]
        k = int(proba[j].argmax())
        conf = float(proba[j][k])
        stem = STEMS[k] if conf >= MIN_CONFIDENCE else None
        labels.append(stem)
        out.append({"t0": t0, "t1": t1, "beat0": b0, "beat1": b1, "stem": stem, "conf": conf,
                    "probs": {s: float(proba[j][n]) for n, s in enumerate(STEMS)}})
    for w, lab in zip(out, smooth(labels)):
        w["stem"] = lab
    return out
