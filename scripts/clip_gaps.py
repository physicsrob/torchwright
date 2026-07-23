"""Measure clip-memory same-column write-gap and look-back distributions.

CPU only. Uses pydoom.expected_ar_tokens on the E1M1 start-room fixture.

Clip WRITE = a SCREEN_RANGE token whose previous token is CLIP_UPDATE
(matches the graph's `screen_range_after_clip_update` = range_active).
The written column is the most-recent SET_CURSOR_X column
(x // PIXEL_WIDTH), mirroring ClipMemory.publish's cursor_x_scalar_pub.
"""

from __future__ import annotations

import itertools
import statistics
from collections import defaultdict

from torchwright_doom.model.constants import COLUMN_COUNT, PIXEL_WIDTH, VIEW_HEIGHT
from torchwright_doom.model.vocab import CLIP_UPDATE, SCREEN_RANGE, SET_CURSOR_X
from torchwright_doom.prompt import scenes as S
from torchwright_doom.pydoom import GameState as PyGameState
from torchwright_doom.pydoom import Scene as PyScene
from torchwright_doom.pydoom import expected_ar_tokens


def build_scene():
    md, state = S.load(S.E1M1_START_ROOM)
    py_scene = PyScene.model_validate(
        {
            "map_data": md.model_dump(),
            "test_poses": [{"x": state.x, "y": state.y, "angle": state.angle}],
        }
    )
    py_state = PyGameState(x=state.x, y=state.y, angle=state.angle)
    return py_scene, py_state


def main() -> None:
    py_scene, py_state = build_scene()
    tokens = expected_ar_tokens(py_scene, py_state)
    print(
        f"PIXEL_WIDTH={PIXEL_WIDTH} COLUMN_COUNT={COLUMN_COUNT} VIEW_HEIGHT={VIEW_HEIGHT}"
    )
    print(f"total tokens in frame: {len(tokens)}")

    cur_col = None
    prev_type = None
    # list of (position, column) for every clip write
    writes: list[tuple[int, int]] = []
    n_clip_update = 0
    n_screen_range_total = 0
    n_screen_range_after_clip = 0

    for pos, tok in enumerate(tokens):
        t = tok.type
        if t is SET_CURSOR_X:
            x = int(tok.values.get("x", 0))
            cur_col = x // PIXEL_WIDTH
        elif t is CLIP_UPDATE:
            n_clip_update += 1
        elif t is SCREEN_RANGE:
            n_screen_range_total += 1
            if prev_type is CLIP_UPDATE:
                n_screen_range_after_clip += 1
                writes.append((pos, cur_col if cur_col is not None else -1))
        prev_type = t

    print(f"CLIP_UPDATE tokens: {n_clip_update}")
    print(f"SCREEN_RANGE tokens total: {n_screen_range_total}")
    print(f"SCREEN_RANGE after CLIP_UPDATE (clip writes): {n_screen_range_after_clip}")
    print(f"distinct columns written: {len({c for _, c in writes})}")

    # (a) gaps in token positions between CONSECUTIVE writes to the SAME column
    by_col: dict[int, list[int]] = defaultdict(list)
    for pos, col in writes:
        by_col[col].append(pos)

    same_col_gaps: list[int] = []
    writes_per_col: list[int] = []
    for col, positions in by_col.items():
        positions.sort()
        writes_per_col.append(len(positions))
        for a, b in itertools.pairwise(positions):
            same_col_gaps.append(b - a)

    # The same set, reinterpreted for RoPE recency resolvability (R13).
    #
    # Recency picks the MOST-RECENT matching key (here: the most-recent prior
    # write to the read column). The dangerous competitor is the RUNNER-UP (the
    # second-most-recent write): the recency signal must separate winner from
    # runner-up. At any read the winner and runner-up are two CONSECUTIVE
    # same-column writes, so {winner-vs-runner-up gaps} is a SUBSET of
    # {same_col_gaps}; hence min(same_col_gap) is a conservative LOWER BOUND on
    # the binding recency gap -- recency never resolves two positions closer
    # than this. The MIN (not the median) is what matters.
    winner_runnerup_gaps = same_col_gaps

    def summ(name, xs) -> None:
        if not xs:
            print(f"  {name}: (none)")
            return
        xs = sorted(xs)
        n = len(xs)

        def p(q):
            return xs[min(n - 1, int(q * n))]

        print(
            f"  {name}: n={n} min={xs[0]} median={statistics.median(xs):.0f} "
            f"mean={statistics.mean(xs):.0f} p90={p(0.90)} p95={p(0.95)} "
            f"p99={p(0.99)} max={xs[-1]}"
        )

    print("\n=== (a) position-gap between consecutive writes to the SAME column ===")
    summ("same_col_gap", same_col_gaps)

    print("\n=== writes per column ===")
    summ("writes_per_col", writes_per_col)
    # columns written exactly once never produce a same-col gap; report how many.
    once = sum(1 for c in writes_per_col if c == 1)
    print(f"  columns written exactly once (no gap): {once} / {len(writes_per_col)}")

    print("\n=== (b) recency winner-vs-runner-up gap (R13: MIN is binding) ===")
    summ("winner_vs_runnerup_gap", winner_runnerup_gaps)

    if winner_runnerup_gaps:
        min_gap = min(winner_runnerup_gaps)
        # A coarse monotone recency signal (a single rotary plane read relative
        # to BOS) resolves the winner from the runner-up when the per-position
        # phase step times the gap exceeds the readout noise. The best plane is
        # the FASTEST that does not wrap over the rollout: theta = 2*pi/N over
        # N=64000 positions (one full turn across the whole rollout, monotone).
        # Then the winner-runner-up phase separation is theta * min_gap, and the
        # readout-noise budget (allowing noise on both) is half of that.
        import math

        N_ROLLOUT = 64000
        theta_rec = 2.0 * math.pi / N_ROLLOUT
        budget_rad = 0.5 * theta_rec * min_gap
        print(
            f"\n=== (c) implied recency readout budget (R13 verdict) ===\n"
            f"  binding gap (min) = {min_gap} positions\n"
            f"  best recency plane theta = 2*pi/{N_ROLLOUT} = {theta_rec:.3e} rad/pos "
            f"(one turn over the rollout, monotone)\n"
            f"  phase readout noise budget < {budget_rad:.3e} rad "
            f"(~{math.degrees(budget_rad):.3f} deg) to keep winner > runner-up\n"
            f"  => a COARSE monotone signal suffices iff the readout (R12) lands "
            f"inside this budget; recency need NOT resolve adjacent positions."
        )

    if writes:
        first_pos = writes[0][0]
        last_pos = writes[-1][0]
        print(
            f"\nclip-write phase spans token positions [{first_pos}, {last_pos}] "
            f"(span={last_pos - first_pos})"
        )

    # CAVEAT: this fixture writes only ~30 of 160 production columns (a near
    # wall in one room); a denser frame could tighten the min gap. Re-run on a
    # fuller production frame before treating the min as final (R13 / R5).


if __name__ == "__main__":
    main()
