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
MIN_AUDIBLE = 0.05

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
        if out[i - 1] is not None and out[i - 1] == out[i + 1] != out[i]:
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
    # Training labels required the followed stem to hold >= MIN_AUDIBLE of
    # the window's energy; enforce the same at inference, or a silent
    # stem with a few hallucinated onsets can win a window (Dubstep Is
    # Dead drop: "guitar 1.00" at 1% guitar energy).
    for j, i in enumerate(keep):
        f = feats[i]
        for n, stem in enumerate(STEMS):
            if f[stem]["energy"] < MIN_AUDIBLE:
                proba[j][n] = 0.0
        total = proba[j].sum()
        if total > 0:
            proba[j] /= total
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


# Rhythm rescue. In EDM drops the followed stem is synth or bass but its
# PITCHED transcription starves: the wub is a filter sweep on one note,
# and Basic Pitch on the mix recovers 9-12% of the human onsets there
# (Dubstep Is Dead drops, Go Beyond drop; 2026-09-05 go/no-go test),
# while onset detection on the followed stem recovers 45-82% at 93-100%
# precision. Charters chart the wub's RHYTHM, laned by its filter
# contour - the same thing brightness lanes do for stuck stretches, but
# here the notes have to be supplied first.
RESCUE_STEMS = ("sw_other", "bass")
RESCUE_MIN_ONSETS_PER_S = 3.0     # a real rhythm, not a pad
RESCUE_STARVED_RATIO = 0.5        # chart holds < half the stem's onsets
RESCUE_ONSET_DELTA = 0.07         # librosa peak-pick threshold (tested)
RESCUE_NOTE_WINDOW_S = 0.12
RESCUE_MIN_SPAN = 1.3             # centroid p90/p10 below this: static
RESCUE_MIN_SHARE = 0.10           # stem must carry this much of the window


def _centroids(signal, sr, times):
    out = []
    for t in times:
        a = int(t * sr)
        seg = signal[a:min(len(signal), a + int(RESCUE_NOTE_WINDOW_S * sr))]
        if len(seg) < 256:
            out.append(0.0)
            continue
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
        power = spec.sum()
        out.append(float((spec * freqs).sum() / power) if power > 0 else 0.0)
    return np.asarray(out)


def _lanes_from_contour(values, strengths):
    """Lanes 0-4 from the log-centroid contour between its p10 and p90,
    median-filtered; a spectrally static run lanes by hit strength
    instead (accents climb), so a monotone drop is not one lane."""
    v = np.asarray(values, dtype=float)
    voiced = v[v > 0]
    if len(voiced) >= 4:
        p10, p90 = np.percentile(voiced, 10), np.percentile(voiced, 90)
        if p10 > 0 and p90 / p10 >= RESCUE_MIN_SPAN:
            logc = np.log(np.maximum(v, p10 * 0.5))
            lanes = np.clip(((logc - np.log(p10)) / (np.log(p90) - np.log(p10)) * 5).astype(int), 0, 4)
            sm = lanes.copy()
            for k in range(1, len(lanes) - 1):
                sm[k] = sorted(lanes[k - 1:k + 2])[1]
            return [int(x) for x in sm]
    st = np.asarray(strengths, dtype=float)
    if len(st) and st.max() > st.min():
        ranks = (st - st.min()) / (st.max() - st.min())
        return [int(min(4, r * 5)) for r in ranks]
    return [1] * len(v)


def rhythm_rescue(expert, followed, mono, sr, tempo):
    """Add notes from the followed stem's onsets in starved synth/bass
    runs. Returns (notes, added, runs_touched)."""
    import librosa

    res = tempo.resolution
    by_tick = {t for t, _, _ in expert}
    open_ticks = sorted({t for t, l, _ in expert if l == 7})
    added = []
    relaned_opens = {}
    touched = 0
    # Onsets are picked on the WHOLE stem once: peak-picking normalises
    # within the signal it is given, so a run segment of a near-silent
    # stem would manufacture dozens of "onsets" out of noise.
    peaks = {}
    for stem in RESCUE_STEMS:
        env = librosa.onset.onset_strength(y=mono[stem], sr=sr)
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, units="frames",
                                            delta=RESCUE_ONSET_DELTA, backtrack=False)
        peaks[stem] = (librosa.frames_to_time(frames, sr=sr), env[frames])
    hop = int(0.05 * sr)
    envs = {}
    for stem in STEMS + ("drums",):
        n = len(mono[stem]) // hop
        envs[stem] = np.sqrt((mono[stem][:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    for r in followed:
        if r["stem"] not in RESCUE_STEMS:
            continue
        t0, t1 = r["t0"], r["t1"]
        sig = mono[r["stem"]]
        all_times, all_strengths = peaks[r["stem"]]
        sel = (all_times >= t0) & (all_times < t1)
        if not sel.any():
            continue
        times, strengths = all_times[sel], all_strengths[sel]
        ticks = [tempo.quantize(t, subdiv=4) for t in times]
        # Starvation is judged per 4-bar WINDOW inside the run: a song-long
        # synth run is well charted on average while its drop is empty
        # (Go Beyond: 405 chart positions vs 527 human, drop 51 vs 211).
        step = BARS_PER_WINDOW * 4 * res
        keep = []
        seen = set()
        fired = []
        for lo in range(int(r["beat0"] * res), int(r["beat1"] * res), step):
            hi = min(lo + step, int(r["beat1"] * res))
            idx = [i for i, tk in enumerate(ticks) if lo <= tk < hi]
            secs = tempo.beat_to_time(hi / res) - tempo.beat_to_time(lo / res)
            if secs <= 0 or len(idx) / secs < RESCUE_MIN_ONSETS_PER_S:
                continue
            # the rescued stem must be audible in this window
            fa, fb = int(tempo.beat_to_time(lo / res) / 0.05), int(tempo.beat_to_time(hi / res) / 0.05)
            fb = max(fa + 1, min(fb, min(len(e) for e in envs.values())))
            means = {st: float(e[fa:fb].mean()) for st, e in envs.items()}
            tot = sum(means.values())
            if tot <= 0 or means[r["stem"]] / tot < RESCUE_MIN_SHARE:
                continue
            present = sum(1 for t in by_tick if lo <= t < hi)
            if present >= RESCUE_STARVED_RATIO * len(idx):
                continue
            fired.append((lo, hi))
            for i in idx:
                tk = ticks[i]
                if tk in seen:
                    continue
                if any(abs(tk - e) <= res // 4 for e in (by_tick & {tk - res // 4, tk, tk + res // 4})):
                    continue
                seen.add(tk)
                keep.append(i)
        if not keep:
            continue
        # The mix transcription's own notes in a fired window are the
        # wub heard as a sub-register pitch, charted as opens; they were
        # fretted before only because brightness relaning fired on the
        # sparse, stuck stretch. Lane them from the same contour as the
        # rescued notes (humans chart 0-5% opens on these songs).
        opens_here = [t for t in open_ticks if any(lo <= t < hi for lo, hi in fired)
                      and t not in relaned_opens]
        open_times = np.array([tempo.beat_to_time(t / res) for t in opens_here])
        all_times = np.concatenate([times[keep], open_times]) if len(open_times) else times[keep]
        fill = float(np.median(strengths[keep])) if len(keep) else 0.0
        all_str = np.concatenate([strengths[keep], np.full(len(open_times), fill)]) \
            if len(open_times) else strengths[keep]
        lanes = _lanes_from_contour(_centroids(sig, sr, all_times), all_str)
        for i, lane in zip(keep, lanes[:len(keep)]):
            added.append((ticks[i], lane, 0))
            by_tick.add(ticks[i])
        for t, lane in zip(opens_here, lanes[len(keep):]):
            relaned_opens[t] = lane
        touched += 1
    if not added and not relaned_opens:
        return expert, 0, 0
    out = [(t, relaned_opens.get(t, l) if l == 7 else l, sus) for t, l, sus in expert]
    return sorted(set(out + added)), len(added), touched
