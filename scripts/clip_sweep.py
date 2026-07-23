"""Sweep player angle to characterize clip-write gap distribution range.

CPU only. Production resolution (set env before running):
  TORCHWRIGHT_DOOM_RENDER_SCALE=1 TORCHWRIGHT_DOOM_DETAIL=low
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from torchwright_doom.prompt import scenes as S
from torchwright_doom.model.constants import PIXEL_WIDTH, COLUMN_COUNT
from torchwright_doom.model.vocab import SET_CURSOR_X, CLIP_UPDATE, SCREEN_RANGE
from torchwright_doom.pydoom import Scene as PyScene, GameState as PyGameState
from torchwright_doom.pydoom import expected_ar_tokens


def writes_for(angle: int):
    md, state = S.load(S.E1M1_START_ROOM)
    py_scene = PyScene.model_validate(
        {
            "map_data": md.model_dump(),
            "test_poses": [{"x": state.x, "y": state.y, "angle": angle}],
        }
    )
    py_state = PyGameState(x=state.x, y=state.y, angle=angle)
    tokens = expected_ar_tokens(py_scene, py_state)
    cur_col = None
    prev = None
    writes = []
    for pos, tok in enumerate(tokens):
        if tok.type is SET_CURSOR_X:
            cur_col = int(tok.values.get("x", 0)) // PIXEL_WIDTH
        elif tok.type is SCREEN_RANGE and prev is CLIP_UPDATE:
            writes.append((pos, cur_col if cur_col is not None else -1))
        prev = tok.type
    return len(tokens), writes


def gaps(writes):
    by = defaultdict(list)
    for pos, col in writes:
        by[col].append(pos)
    g = []
    for col, ps in by.items():
        ps.sort()
        g += [b - a for a, b in zip(ps, ps[1:])]
    return g


print(f"PIXEL_WIDTH={PIXEL_WIDTH} COLUMN_COUNT={COLUMN_COUNT}")
print(
    f"{'angle':>5} {'ntok':>6} {'nwrite':>6} {'ncol':>4} {'ngap':>5} "
    f"{'med':>5} {'p95':>6} {'max':>6}"
)
all_gaps = []
for angle in range(0, 256, 16):
    ntok, writes = writes_for(angle)
    g = sorted(gaps(writes))
    all_gaps += g
    ncol = len(set(c for _, c in writes))
    if g:
        med = int(statistics.median(g))
        p95 = g[min(len(g) - 1, int(0.95 * len(g)))]
        mx = g[-1]
    else:
        med = p95 = mx = 0
    print(
        f"{angle:>5} {ntok:>6} {len(writes):>6} {ncol:>4} {len(g):>5} "
        f"{med:>5} {p95:>6} {mx:>6}"
    )

ag = sorted(all_gaps)
n = len(ag)
print(
    f"\nAGGREGATE over all angles: n={n} min={ag[0]} "
    f"median={statistics.median(ag):.0f} p95={ag[int(0.95 * n)]} "
    f"p99={ag[int(0.99 * n)]} max={ag[-1]}"
)
