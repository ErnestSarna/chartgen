"""Separate what the model emits from what our processing does to it.

Answers "is a bad lane distribution the model's fault or ours?" by reporting the
raw token histogram alongside the final chart lanes. Tokens are cached to
work/tokens-*.json so the analysis can be rerun without paying for generation.

    python tools/diagnose.py work/song.wav --model 3podi/charter-v1.0-40-S-best-acc
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chartgen  # noqa: F401,E402  (puts vendor trees on sys.path)
from chartgen import chart as chartio  # noqa: E402
from chartgen import tempo as tempomod  # noqa: E402

LANE_NAMES = {0: "G", 1: "R", 2: "Y", 3: "B", 4: "O", 7: "open"}


def get_tokens(audio: Path, model_name: str, temperature: float, top_k: int, runs: int):
    cache = Path("work") / f"tokens-{model_name.split('/')[-1]}-t{temperature}-n{runs}.json"
    if cache.exists():
        print(f"(reusing {cache})")
        return json.loads(cache.read_text()), None

    import torch
    from inference.engine import Charter

    model = Charter.from_pretrained(model_name)
    out = []
    for i in range(runs):
        seqs = model.generate(str(audio), temperature=temperature, top_k=top_k)
        out.append(torch.cat(seqs).flatten().cpu().tolist())
        print(f"  run {i + 1}/{runs} done")
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out))
    return out, model.config.grid_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--model", default="3podi/charter-v1.0-40-S-best-acc")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_k", type=int, default=32)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--grid-ms", type=int, default=40)
    ap.add_argument("--subdiv", type=int, default=4)
    args = ap.parse_args()

    from chart.tokenizer import SimpleTokenizerGuitar

    reverse_map = SimpleTokenizerGuitar().reverse_map
    y, sr = tempomod.load(str(args.audio))
    tempo = tempomod.detect(y, sr)

    all_tokens, grid_ms = get_tokens(
        args.audio, args.model, args.temperature, args.top_k, args.runs
    )
    grid_ms = grid_ms or args.grid_ms

    for run, tokens in enumerate(all_tokens):
        note_frames = [(i, t) for i, t in enumerate(tokens) if t in reverse_map]
        raw_lanes = collections.Counter(
            lane for _, t in note_frames for lane in reverse_map[t]
        )
        raw_chords = collections.Counter(len(reverse_map[t]) for _, t in note_frames)

        notes = chartio.notes_from_tokens(
            tokens, grid_ms, tempo, reverse_map, subdiv=args.subdiv
        )
        by_tick = collections.defaultdict(list)
        for tick, lane, _ in notes:
            by_tick[tick].append(lane)
        final_lanes = collections.Counter(lane for _, lane, _ in notes)
        final_chords = collections.Counter(len(v) for v in by_tick.values())

        # How often did several model frames collapse onto one chart tick?
        frames_per_tick = collections.Counter(
            tempo.quantize(i * grid_ms / 1000.0, subdiv=args.subdiv)
            for i, _ in note_frames
        )
        merged = sum(1 for c in frames_per_tick.values() if c > 1)

        def pct(counter):
            tot = sum(counter.values()) or 1
            return "  ".join(
                f"{LANE_NAMES.get(k, k)}={v}({v / tot * 100:.0f}%)"
                for k, v in sorted(counter.items())
            )

        print(f"\n--- run {run + 1} ---")
        print(f"note frames emitted by model : {len(note_frames)} of {len(tokens)}")
        print(f"model lanes  : {pct(raw_lanes)}")
        print(f"chart lanes  : {pct(final_lanes)}")
        print(f"model chord sizes : {dict(sorted(raw_chords.items()))}")
        print(f"chart chord sizes : {dict(sorted(final_chords.items()))}")
        print(f"ticks built from >1 model frame : {merged} of {len(by_tick)}")
        top = max(final_lanes.values()) / max(1, sum(final_lanes.values())) * 100
        print(f"most-used lane : {top:.0f}% "
              f"{'<-- DEGENERATE' if top > 60 else ''}")


if __name__ == "__main__":
    main()
