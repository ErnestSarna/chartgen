"""Do the pitch features carry what the model does not?

Same three fixtures as tools/pitch_probe.py, which measured the model at
-0.04 correlation on a falling scale. If these features score strongly and
symmetrically on the same audio, the signal exists and the gap is that it never
reaches the model — which is what makes pitch conditioning worth training.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartgen.pitch import frets_from_pitch, pitch_features  # noqa: E402
from tools.pitch_probe import ALTERNATING, FALLING, RISING, render  # noqa: E402

GRID_MS = 40


def main():
    import librosa

    Path("work").mkdir(exist_ok=True)
    print(f"{'fixture':<13} {'corr(pitch,f0)':>15} {'corr(pitch,fret)':>18}  fret span")
    for label, freqs in (("rising", RISING), ("falling", FALLING),
                         ("alternating", ALTERNATING)):
        path = f"work/feat_{label}.wav"
        render(freqs, path)
        y, sr = librosa.load(path, mono=True)
        _, f0, voiced = pitch_features(y, sr, GRID_MS)

        # One onset per eighth note. Sample just *after* each onset: truncating
        # to the frame below lands on the previous note's ring-out, which with
        # 400ms decay is still the louder of the two.
        eighth = 60.0 / 120.0 / 2
        n = int((len(y) / sr) / eighth)
        idx = [int(round(i * eighth * 1000 / GRID_MS)) + 1 for i in range(n)]
        idx = [i for i in idx if i < len(f0)]
        true_pitch = np.log([freqs[i % len(freqs)] for i in range(len(idx))])
        measured = f0[idx]
        frets = frets_from_pitch(f0, voiced, idx)

        c_f0 = np.corrcoef(true_pitch, measured)[0, 1]
        c_fr = np.corrcoef(true_pitch, frets)[0, 1] if len(set(frets)) > 1 else float("nan")
        print(f"{label:<13} {c_f0:>+15.3f} {c_fr:>+18.3f}  {frets.min()}-{frets.max()}")
        print(f"              first 24 frets: {list(frets[:24])}")

    print("\nmodel, same fixtures (tools/pitch_probe.py): rising +0.44/+0.58, "
          "falling -0.04/-0.05, leaps +0.09/+0.40")


if __name__ == "__main__":
    main()
