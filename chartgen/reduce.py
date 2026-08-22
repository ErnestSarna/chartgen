"""Derive Hard/Medium/Easy from Expert.

This replaces driving the vendored EasyChartGenerator (still available as
`derive_lower_tiers_vendored`), which had two measured defects on real songs:

- Its Hard rule is a pure grid filter — keep beats and 8ths, delete every 16th.
  Natural HOPOs only occur under a 162-tick gap, i.e. between 16ths, so Hard
  came out all strums: 3 HOPOs where Expert had 650. Real Hard charts lean on
  HOPOs; an all-strum Hard is needlessly tiring and feels wrong.
- Its lane remap collapses orange AND open onto green (4->0, 7->0, 3->1),
  piling notes onto one lane: Medium measured 0.72 variety and Easy 0.41
  against the 0.80 gate.

The rules here, per tier:

Hard    keep beats and 8ths; keep short 16th figures whole; thin long 16th
        runs by dropping only the "e" (first 16th after the beat), so every
        beat keeps a HOPO pair and density drops instead of rhythm dying.
        Chords capped at 2 notes.
Medium  quarter notes, with 8th-position notes kept only where a bar would
        otherwise go quiet. Chords capped at 2. No orange (order-preserving
        remap to G/R/Y/B, not a collapse).
Easy    on-beat singles at least 2 beats apart (1 beat where the song would
        otherwise go silent for a bar). G/R/Y only, order-preserving.
"""
from types import SimpleNamespace

import re

NOTE_RE = re.compile(r"(\d+)\s*=\s*N\s+(\d+)\s+(\d+)")
OPEN = 7

# Order-preserving compressions: relative pitch motion survives, one adjacent
# pair merges (which reads as a repeat — what real parts do anyway). Opens map
# to green below Expert; sub-Expert opens are unusual in real charts.
MEDIUM_LANES = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, OPEN: 0}
EASY_LANES = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, OPEN: 0}

Note = tuple[int, int, int]


def _by_tick(notes: list[Note]) -> dict[int, list[Note]]:
    out: dict[int, list[Note]] = {}
    for note in notes:
        out.setdefault(note[0], []).append(note)
    return out


def _sixteenth_chains(ticks: list[int], resolution: int) -> dict[int, int]:
    """tick -> length of the dense run it belongs to.

    A "dense run" is consecutive positions closer than an 8th — 16ths on the
    straight grid, triplet 8ths on the triplet grid. Run length decides whether
    a figure is kept whole (short lick) or thinned (wall of 16ths).
    """
    chain_len: dict[int, int] = {}
    run = [ticks[0]] if ticks else []
    for prev, cur in zip(ticks, ticks[1:]):
        if cur - prev < resolution // 2:
            run.append(cur)
        else:
            for t in run:
                chain_len[t] = len(run)
            run = [cur]
    for t in run:
        chain_len[t] = len(run)
    return chain_len


def _cap_chord(group: list[Note], max_notes: int) -> list[Note]:
    """Keep the lowest lanes: the root carries the motion of a chord line."""
    fretted = sorted((n for n in group if n[1] != OPEN), key=lambda n: n[1])
    opens = [n for n in group if n[1] == OPEN]
    kept = (opens[:1] + fretted)[:max_notes] if not fretted else fretted[:max_notes]
    return kept or group[:max_notes]


def _remap(notes: list[Note], table: dict[int, int]) -> list[Note]:
    return [(t, table[lane], s) for t, lane, s in notes]


def _dedupe(notes: list[Note]) -> list[Note]:
    """Remapping can land two chord notes on one lane; keep the longer."""
    best: dict[tuple[int, int], Note] = {}
    for note in notes:
        key = (note[0], note[1])
        if key not in best or note[2] > best[key][2]:
            best[key] = note
    return sorted(best.values())


HARD_MAX_NPS = 4.5  # real Hard charts cap around here regardless of genre


def _cap_bar_density(ticks: list[int], resolution: int, per_bar_cap: int) -> list[int]:
    """Enforce a per-bar position budget: fine positions go first, then every
    other off-8th. Beats are never dropped."""
    bar = 4 * resolution
    capped: list[int] = []
    i = 0
    while i < len(ticks):
        bar_start = (ticks[i] // bar) * bar
        bar_ticks = [t for t in ticks[i:] if t < bar_start + bar]
        i += len(bar_ticks)
        over = len(bar_ticks) - per_bar_cap
        if over > 0:
            fine = [t for t in bar_ticks if t % resolution not in (0, resolution // 2)]
            drop = set(fine[:over])
            over -= len(drop)
            if over > 0:
                off8 = [t for t in bar_ticks
                        if t % resolution == resolution // 2 and t not in drop]
                drop.update(off8[1::2][:over])
            bar_ticks = [t for t in bar_ticks if t not in drop]
        capped.extend(bar_ticks)
    return capped


def thin_to_rating(expert: list[Note], resolution: int, tempo,
                   target: int) -> tuple[list[Note], int]:
    """Thin an Expert track until its predicted tier drops to `target`.

    Tightens a per-bar position budget one notch at a time, re-rating after
    each cut, so the chart loses exactly as much as the target demands and no
    more. Returns (notes, achieved_rating); achieved may sit above target
    when even a beats-only chart still rates higher (short fast songs).
    """
    from . import rating

    groups = _by_tick(expert)
    current = expert
    achieved = rating.rate_expert(current, tempo)
    per_bar = 16
    while achieved > target and per_bar > 2:
        per_bar -= 1
        kept = _cap_bar_density(sorted(groups), resolution, per_bar)
        current = [n for n in expert if n[0] in set(kept)]
        achieved = rating.rate_expert(current, tempo)
    return current, achieved


def reduce_hard(expert: list[Note], resolution: int,
                bpm: float | None = None) -> list[Note]:
    groups = _by_tick(expert)
    ticks = sorted(groups)
    chains = _sixteenth_chains(ticks, resolution)

    kept: list[int] = []
    for tick in ticks:
        pos = tick % resolution
        fine = pos not in (0, resolution // 2)  # not a beat, not an 8th
        # Long saturated runs: drop the fine positions in the first half of
        # the beat. Straight grid: the "e" goes, the "a" stays -> the beat
        # keeps one 120-tick HOPO pair. Short figures are kept whole.
        if fine and chains.get(tick, 1) > 4 and 0 < pos < resolution // 2:
            continue
        kept.append(tick)

    if bpm:
        # Playtest: on dense EDM the rule above barely reduces (92% of Expert
        # survived) because everything is beats and 8ths. Enforce a
        # notes-per-second ceiling per bar.
        per_bar_cap = max(4, int(round(HARD_MAX_NPS * 4 * 60.0 / bpm)))
        kept = _cap_bar_density(kept, resolution, per_bar_cap)

    out: list[Note] = []
    for tick in kept:
        out.extend(_cap_chord(groups[tick], 2))
    return sorted(out)


def reduce_medium(expert: list[Note], resolution: int) -> list[Note]:
    groups = _by_tick(expert)
    out: list[Note] = []
    last_kept = -resolution * 8
    for tick in sorted(groups):
        pos = tick % resolution
        # Beats and 8ths, deliberately generous: the ratio cap in derive_tiers
        # trims whatever this produces down to the measured 55% of Expert. A
        # strict rule cannot be trimmed UP, and on songs with sparse downbeats
        # it left Medium sitting on top of Easy (33% vs 32%) — the scanner's
        # difficultyNotReduced error.
        if pos not in (0, resolution // 2):
            continue
        out.extend(_cap_chord(groups[tick], 2))
        last_kept = tick
    return _dedupe(_remap(out, MEDIUM_LANES))


def reduce_easy(expert: list[Note], resolution: int) -> list[Note]:
    groups = _by_tick(expert)
    out: list[Note] = []
    last_kept = -resolution * 8
    for tick in sorted(groups):
        pos = tick % resolution
        on_beat = pos == 0
        gap = tick - last_kept
        # Downbeats carry Easy; an 8th position is allowed only to break a
        # silence of two beats or more. Like Medium this is deliberately
        # generous and trimmed to the measured 39% of Expert afterwards.
        #
        # Half-note pacing came from the RBN guideline ("leave half note
        # spaces between strums") and measured far too sparse against real
        # Clone Hero charts — 0.5 notes/sec where they average ~1.8.
        keep = on_beat or (gap >= 2 * resolution and pos == resolution // 2)
        if not keep:
            continue
        out.extend(_cap_chord(groups[tick], 1))
        last_kept = tick
    return _dedupe(_remap(out, EASY_LANES))


def enforce_chord_rules(tiers: dict[str, list[Note]]) -> dict[str, list[Note]]:
    """Remove chord shapes the community standards call unplayable.

    The Chorus Encore scanner errors on a Green+Orange two-note chord on Hard,
    and the RBN guidelines keep three-note chords spanning green to orange off
    Expert as well ("as sparingly as possible" even as a pair). Medium adds no
    Green+Blue. These are hand contortions rather than musical choices, so the
    outer note moves inward one lane and keeps its sustain.
    """
    out: dict[str, list[Note]] = {}
    for name, notes in tiers.items():
        groups = _by_tick(notes)
        fixed: list[Note] = []
        for tick in sorted(groups):
            held = {lane: sus for _, lane, sus in groups[tick]}
            fretted = sorted(l for l in held if l != OPEN)

            spans_neck = 0 in held and 4 in held
            if spans_neck and (
                (name == "HardSingle" and len(fretted) == 2)
                or (name == "ExpertSingle" and len(fretted) >= 3)
            ):
                sus = held.pop(4)
                held.setdefault(3, sus)
            if name == "MediumSingle" and 0 in held and 3 in held:
                sus = held.pop(3)
                held.setdefault(2, sus)

            fixed.extend((tick, lane, held[lane]) for lane in sorted(held))
        out[name] = fixed
    return out



# Median position counts relative to Expert across 49 human charts in the
# local library that carry a full four-tier ladder. Grid rules alone drifted
# well off these (Hard landed at 96% of Expert on one song), so the ratios are
# enforced as a cap rather than hoped for.
TIER_RATIO = {"HardSingle": 0.78, "MediumSingle": 0.55, "EasySingle": 0.39}


def _thin_to_ratio(ticks: list[int], resolution: int, target: int) -> list[int]:
    """Drop the least structural positions until at most `target` remain.

    Fine positions go before 8ths and 8ths before beats, so the pulse of the
    song survives even a heavy cut, and within each class the drops are spread
    evenly rather than taken from one stretch.
    """
    if len(ticks) <= target:
        return ticks

    def rank(tick: int) -> int:
        pos = tick % resolution
        if pos == 0:
            return 2                     # downbeats are the last to go
        return 1 if pos == resolution // 2 else 0

    keep = set(range(len(ticks)))
    need = len(ticks) - target
    for level in (0, 1, 2):
        if need <= 0:
            break
        idx = [i for i in sorted(keep) if rank(ticks[i]) == level]
        if not idx:
            continue
        take = min(need, len(idx))
        step = len(idx) / take
        for j in range(take):
            keep.discard(idx[int(j * step)])
        need -= take
    return [t for i, t in enumerate(ticks) if i in keep]


def derive_tiers(expert: list[Note], resolution: int,
                 bpm: float | None = None) -> dict[str, list[Note]]:
    """{'HardSingle': ..., 'MediumSingle': ..., 'EasySingle': ...}"""
    tiers = {
        "HardSingle": reduce_hard(expert, resolution, bpm=bpm),
        "MediumSingle": reduce_medium(expert, resolution),
        "EasySingle": reduce_easy(expert, resolution),
    }
    expert_positions = len({t for t, _, _ in expert})
    out: dict[str, list[Note]] = {}
    # Walk down the ladder: each tier is capped at its measured share of
    # Expert AND forced strictly below the tier above it. Without the second
    # clause a uniformly dense song collapses Medium and Easy onto the same
    # "every beat" grid, which is the scanner's `difficultyNotReduced` error.
    previous = expert_positions
    for name in ("HardSingle", "MediumSingle", "EasySingle"):
        notes = tiers[name]
        target = max(1, min(int(expert_positions * TIER_RATIO[name]), previous - 1))
        kept = set(_thin_to_ratio(sorted(_by_tick(notes)), resolution, target))
        out[name] = [n for n in notes if n[0] in kept]
        previous = len(kept)
    return out


def derive_lower_tiers_vendored(expert_chart_text: str) -> dict[str, list[Note]]:
    """The original EasyChartGenerator path, kept for A/B comparison."""
    import easygen

    options = SimpleNamespace(
        force=True, bpm_multiplier=1, doublekick=0,
        easy=False, medium=False, hard=False,
    )
    parser = easygen.Parser(options)
    # It expects pre-stripped lines with any BOM already removed.
    parser.parse_file([ln.strip().replace("﻿", "") for ln in expert_chart_text.splitlines()])

    tiers = {}
    for part_name, lines in parser.new_parts.items():
        notes = [
            (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            for m in (NOTE_RE.match(ln) for ln in lines) if m
        ]
        tiers[part_name.strip("[]")] = sorted(notes)
    return tiers
