"""Walls-as-tokens attention prototype — derisking the mechanism.

This test exists to answer one question before investing in the full
walls-as-runtime-data proposal: can the existing ``Attn`` node, with
hand-designed Q/K/V/O matrices, implement angular-similarity +
distance-penalty wall selection accurately enough to gather a single
wall's value vector cleanly?

Scope (deliberately narrow):

* No rendering, no intersection math, no texture lookup, no DOOM
  geometry.  Wall parameters are fed directly as input tensors — the
  test is not about computing cos/sin/dist from raw ``(x1, y1, x2, y2)``
  and a player position, just about whether attention can *select*
  the right wall given them.
* A single full-width rotary content head (``d_head = 256``, the production
  grid), single layer.  The 3 geometry scalars ride the three slowest planes
  via the production ``rotary_content_head`` builder; the global positional
  rotation is ≈ identity over the small wall/query token offsets here, so the
  selection logits are preserved.  See the RoPE note at the end.
* Walls live at input positions ``0..N-1``, queries at ``N..N+M-1``.
  Causal masking plus zero-K on the non-matching role is what
  prevents cross-role contamination; the tests include explicit
  checks for both directions.

Input row layout (each slot is its own 1-wide :func:`create_input`):

    is_wall        1 at wall rows, 0 at query rows
    is_query       0 at wall rows, 1 at query rows
    ray_cos        0 at wall rows, cos(ray_angle) at query rows
    ray_sin        0 at wall rows, sin(ray_angle) at query rows
    wall_cos_mid   cos(wall_midpoint_angle) at wall rows, 0 at query rows
    wall_sin_mid   sin(wall_midpoint_angle) at wall rows, 0 at query rows
    wall_dist      player-to-wall distance at wall rows, 0 at query rows
    wall_id        wall identifier at wall rows, 0 at query rows

Attn matrices (hand-designed, not trained):

    Q = logit_scale * (ray_cos, ray_sin, is_query)
    K = (wall_cos_mid, wall_sin_mid, is_wall * wall_bias - dist_scale * wall_dist)
    V = (wall_id, wall_cos_mid, wall_sin_mid)

    Q·K at query@wall = logit_scale * (cos(ray_angle - wall_midpoint)
                                        + wall_bias
                                        - dist_scale * wall_dist)

Because ``is_query`` is zero on wall rows and the wall slots are zero
on query rows, both Q and K collapse to ``(0, 0, 0)`` on the wrong
role.  A query's logits against wall rows carry the full geometric
score; its logits against prior query rows are exactly zero, so
softmax deprioritises them unconditionally as long as the query
rows' zero logit is comfortably below the wall logits.  And since V
is also ``(0, 0, 0)`` on query rows, any residual attention weight
that leaks onto a query row contributes zero to the gathered output.

``wall_bias`` is the fix for a subtle failure mode: a wall whose
``cos_diff - dist_scale * wall_dist`` happens to be ≤ 0 would tie
with the query's zero self-attention logit, splitting mass 50/50
with the V=0 self-row and producing a halved output.  Routing
``is_wall * wall_bias`` into K dim 2 pushes every wall logit up by a
fixed positive offset at wall rows (but not at query rows, where the
``is_wall`` slot is zero), keeping the gap to the zero-logit
baseline wide enough that any realistic ``wall_dist`` still yields
a strongly positive logit.

RoPE form (universal full-width rotary).  Every head in the end-state compiler
is full-width rotary on one global grid, so this head is built via
``rotary_content_head`` at ``d_head = 256`` with the three content scalars
(angular cos, angular sin, role/distance) relocated onto the three slowest
planes.  Two distinct cos/sin pairs are in play and must NOT be conflated: the
angular ``cos``/``sin`` above are a *content* dot product (geometry fed as input
numbers — ``cos(ray - wall_midpoint)``); RoPE's positional ``cos``/``sin`` is a
rotation by *token position* applied on top.  Slow planes make that positional
rotation quasi-static (``cos(Δ·θ) ≈ 1`` for the tiny token offsets here, error
~1e-9), so it does not corrupt the angular content and the un-rotated analytical
reference below stays exact within the tests' tolerances.

Open questions for the Phase-6 ``torchwright_doom`` port.  This prototype proves
the *mechanism* (rotary attention can do angular + distance selection on slow
planes); the following need the walls-as-tokens token-stream spec, which does
not exist yet, so they are NOT derisked here:

* **Distance dynamic range.**  The distance penalty rides a slow plane that
  attenuates ~0.8% at the 42k rollout — 0.8% of the *absolute* penalty, which
  can swamp a small Δ-distance tiebreak.  Real wall distances must be normalised
  to a bounded score (cf. ``_MAX_SCORE_ABS`` in the Phase-3 content tests), not
  raw map units.
* **Angular resolution.**  Real DOOM adjacent-column separation (~0.36° at full
  width) is ~1000× tighter than this toy's ~11°, forcing a much larger
  ``logit_scale`` (~3e5); calibrate to the real per-column ``cos`` gap.
* **Δ-regime.**  Whether walls re-emit per frame (token offset Δ ≈ 300, easy) or
  persist (Δ ≈ 42k, the Phase-3-hard regime) decides the whole difficulty class.
* **Geometric fidelity.**  Real DOOM assigns a column to the wall whose angular
  *interval* contains the ray (with occlusion), not by midpoint-cosine
  similarity.  Midpoint-cos is a derisking proxy for "can rotary attention do
  angular selection at all", not how DOOM picks walls; a faithful version needs
  an in-interval test in the MLP feeding a validity head (a larger graph).
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch

from torchwright.debug.probe import probe_graph, reference_eval
from torchwright.graph import Concatenate
from torchwright.graph.rope import ROPE_BASE, rotary_content_head
from torchwright.ops.inout_nodes import create_input

# ---------------------------------------------------------------------------
# Slot layout
# ---------------------------------------------------------------------------


_SLOT_IS_WALL = 0
_SLOT_IS_QUERY = 1
_SLOT_RAY_COS = 2
_SLOT_RAY_SIN = 3
_SLOT_WALL_COS = 4
_SLOT_WALL_SIN = 5
_SLOT_WALL_DIST = 6
_SLOT_WALL_ID = 7

D_INPUT = 8
WALL_D_HEAD = 256  # rotary grid (production d_head); content rides the 3 slowest planes
D_OUTPUT = 3  # (attended_wall_id, attended_wall_cos_mid, attended_wall_sin_mid)

INPUT_NAMES = (
    "is_wall",
    "is_query",
    "ray_cos",
    "ray_sin",
    "wall_cos_mid",
    "wall_sin_mid",
    "wall_dist",
    "wall_id",
)

_OUT_WALL_ID = 0
_OUT_WALL_COS = 1
_OUT_WALL_SIN = 2


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_wall_attention_graph(
    logit_scale: float = 100.0,
    dist_scale: float = 1.0,
    wall_bias: float = 30.0,
):
    """Build the wall-attention head as a full-width rotary content head.

    The three geometric content scalars are fed to ``rotary_content_head`` (the
    production slow-plane content builder), which relocates them onto the three
    slowest planes of a ``d_head = WALL_D_HEAD`` rotary grid and builds the
    ``Attn``.  Over the small wall/query token offsets here the positional
    rotation is ≈ identity (see the module docstring), so the logits are the
    same un-rotated ``logit_scale * (cos_diff + wall_bias - dist_scale*wall_dist)``
    the analytical reference computes.

    Q/K read the 7 geometry slots (everything but ``wall_id``); the payload
    ``value_in`` is a separate ``(wall_id, wall_cos_mid, wall_sin_mid)`` node
    because ``rotary_content_head`` uses identity V/O.  ``wall_id`` is therefore
    NOT in the shared Q/K residual (it is read only by V), which also keeps the
    Q/K ``Concatenate`` free of dead children.

    ``wall_bias`` is a positive constant routed into the role/distance content
    column via ``is_wall``.  It keeps every wall logit comfortably above the
    zero-logit query-self baseline — see the module docstring for why.
    """
    inputs = [create_input(name, 1) for name in INPUT_NAMES]

    # Shared Q/K source: the 7 geometry slots (wall_id excluded — it is payload,
    # read only by V).  Every slot here is read by Q or K (no dead children).
    qk_residual = Concatenate(inputs[:_SLOT_WALL_ID])

    # Q content (compact, W=3): logit_scale * (ray_cos, ray_sin, is_query).
    # Zero on wall rows because ray_cos/ray_sin/is_query are all zero there.
    q_content = torch.zeros(_SLOT_WALL_ID, D_OUTPUT)
    q_content[_SLOT_RAY_COS, 0] = logit_scale
    q_content[_SLOT_RAY_SIN, 1] = logit_scale
    q_content[_SLOT_IS_QUERY, 2] = logit_scale

    # K content (compact, W=3): (wall_cos_mid, wall_sin_mid,
    #   is_wall*wall_bias - dist_scale*wall_dist).  Zero on query rows.
    k_content = torch.zeros(_SLOT_WALL_ID, D_OUTPUT)
    k_content[_SLOT_WALL_COS, 0] = 1.0
    k_content[_SLOT_WALL_SIN, 1] = 1.0
    k_content[_SLOT_IS_WALL, 2] = wall_bias
    k_content[_SLOT_WALL_DIST, 2] = -dist_scale

    # Payload (identity V/O): (wall_id, wall_cos_mid, wall_sin_mid).  Zero on
    # query rows (all wall slots are zero there), so role-gating is preserved.
    payload = Concatenate(
        [inputs[_SLOT_WALL_ID], inputs[_SLOT_WALL_COS], inputs[_SLOT_WALL_SIN]]
    )

    return rotary_content_head(
        qk_residual,
        qk_residual,
        payload,
        query_matrix=q_content,
        key_matrix=k_content,
        d_head=WALL_D_HEAD,
        base=ROPE_BASE,
    )


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _wall_on_circle(k: int, n: int, radius: float, wall_id: float) -> dict:
    """A single wall at angle ``2π * k / n`` on a circle of ``radius``."""
    angle = 2.0 * math.pi * k / n
    return {
        "wall_cos_mid": math.cos(angle),
        "wall_sin_mid": math.sin(angle),
        "wall_dist": radius,
        "wall_id": float(wall_id),
        "_angle": angle,
    }


def _walls_on_circle(n_walls: int, radius: float = 5.0) -> List[dict]:
    """``n_walls`` walls spaced evenly on a circle around the player.

    ``wall_id`` is ``k + 1`` so the "no wall selected" output of 0
    can never be mistaken for a real wall.
    """
    return [_wall_on_circle(k, n_walls, radius, wall_id=k + 1) for k in range(n_walls)]


def _query_from_angle(ray_angle: float) -> dict:
    """A query ray at a given angle (radians)."""
    return {
        "ray_cos": math.cos(ray_angle),
        "ray_sin": math.sin(ray_angle),
        "_angle": ray_angle,
    }


def _build_input_dict(
    walls: List[dict],
    queries: List[dict],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Pack wall + query records into ``{name: (n_pos, 1)}`` input tensors.

    Wall rows come first, queries after.  Every slot gets zero by
    default; each role fills in only its own slots.
    """
    n_walls = len(walls)
    n_queries = len(queries)
    n_pos = n_walls + n_queries

    inputs = {name: torch.zeros(n_pos, 1) for name in INPUT_NAMES}

    for i, wall in enumerate(walls):
        inputs["is_wall"][i, 0] = 1.0
        inputs["wall_cos_mid"][i, 0] = wall["wall_cos_mid"]
        inputs["wall_sin_mid"][i, 0] = wall["wall_sin_mid"]
        inputs["wall_dist"][i, 0] = wall["wall_dist"]
        inputs["wall_id"][i, 0] = wall["wall_id"]

    for j, query in enumerate(queries):
        pos = n_walls + j
        inputs["is_query"][pos, 0] = 1.0
        inputs["ray_cos"][pos, 0] = query["ray_cos"]
        inputs["ray_sin"][pos, 0] = query["ray_sin"]

    return inputs, n_pos


# ---------------------------------------------------------------------------
# Reference softmax
# ---------------------------------------------------------------------------


def _expected_softmax_weights(
    walls: List[dict],
    queries: List[dict],
    query_idx: int,
    logit_scale: float,
    dist_scale: float,
    wall_bias: float,
) -> np.ndarray:
    """Reference softmax attention weights for the ``query_idx``-th query.

    Walls occupy rows ``0..n_walls-1``, queries occupy rows
    ``n_walls..``.  The query at row ``n_walls + query_idx`` attends
    causally to rows ``0..n_walls + query_idx`` inclusive.

    * Wall rows contribute logit
      ``= logit_scale * (cos_diff + wall_bias - dist_scale * wall_dist)``.
    * Query rows (self and prior) have ``K = (0, 0, 0)``, so their
      logits are exactly zero.

    Returns a ``(query_row + 1,)`` array of softmax weights.
    """
    n_walls = len(walls)
    query_row = n_walls + query_idx
    q = queries[query_idx]
    ray_cos = q["ray_cos"]
    ray_sin = q["ray_sin"]

    logits = np.zeros(query_row + 1, dtype=np.float64)
    for i, w in enumerate(walls):
        cos_diff = w["wall_cos_mid"] * ray_cos + w["wall_sin_mid"] * ray_sin
        logits[i] = logit_scale * (cos_diff + wall_bias - dist_scale * w["wall_dist"])
    # Query rows (positions n_walls..query_row inclusive) have K=0, logit=0.
    # Those positions are already zero in the `logits` array.

    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def _expected_output(
    walls: List[dict],
    queries: List[dict],
    query_idx: int,
    logit_scale: float,
    dist_scale: float,
    wall_bias: float,
) -> np.ndarray:
    """The ``(wall_id, wall_cos_mid, wall_sin_mid)`` vector the query
    should see given the expected softmax weights.
    """
    weights = _expected_softmax_weights(
        walls,
        queries,
        query_idx,
        logit_scale,
        dist_scale,
        wall_bias,
    )
    n_walls = len(walls)
    out = np.zeros(D_OUTPUT, dtype=np.float64)
    for i, w in enumerate(walls):
        out[_OUT_WALL_ID] += weights[i] * w["wall_id"]
        out[_OUT_WALL_COS] += weights[i] * w["wall_cos_mid"]
        out[_OUT_WALL_SIN] += weights[i] * w["wall_sin_mid"]
    # Query rows contribute zero V (all wall-slot extractions hit zeros).
    return out


# ---------------------------------------------------------------------------
# Test 1: single wall, single query — mechanism sanity
# ---------------------------------------------------------------------------


def test_single_wall_single_query():
    """One wall, one query pointed straight at it.

    This is a pure sanity check: does the attention mechanism even
    compose in this IR?  With just one wall the softmax degenerates
    (a single wall logit against a single query logit of zero) — so
    long as ``logit_scale`` is a few units above zero, the wall
    dominates and the output is the wall's V up to softmax noise.
    """
    attn = _build_wall_attention_graph(logit_scale=100.0, dist_scale=1.0)

    walls = [
        {
            "wall_cos_mid": 1.0,
            "wall_sin_mid": 0.0,
            "wall_dist": 0.0,  # zero distance so the dist term doesn't suppress the logit
            "wall_id": 42.0,
        }
    ]
    queries = [_query_from_angle(0.0)]  # ray points at wall_cos=1, wall_sin=0

    inputs, n_pos = _build_input_dict(walls, queries)
    cache = reference_eval(attn, inputs, n_pos)
    out = cache[attn]

    # Query is at row n_pos - 1.
    query_row = n_pos - 1
    assert out.shape == (n_pos, D_OUTPUT)
    assert abs(out[query_row, _OUT_WALL_ID].item() - 42.0) < 1e-4, (
        f"attended wall_id was {out[query_row, _OUT_WALL_ID].item()}, expected 42.0"
    )
    assert abs(out[query_row, _OUT_WALL_COS].item() - 1.0) < 1e-4
    assert abs(out[query_row, _OUT_WALL_SIN].item() - 0.0) < 1e-4


# ---------------------------------------------------------------------------
# Test 2: N walls on a circle, query at each wall's midpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_walls", [2, 4, 8, 16])
def test_n_walls_each_midpoint(n_walls):
    """``n_walls`` walls evenly spaced on a circle; one query per wall,
    each pointed exactly at its wall's midpoint.  Every query must
    retrieve its own wall's V.

    The softmax weight on the correct wall is whatever logit
    separation falls out of the geometry.  For walls on a unit circle
    the adjacent-wall suppression is ``logit_scale * (1 - cos(2π/n))``
    in logit units — large ``n`` needs a larger ``logit_scale`` for
    the softmax to remain sharp.  This test uses ``logit_scale=200``,
    which handles ``n_walls <= 16`` with ≥ 0.99 weight on the correct
    wall (verified by ``test_softmax_weights_match_reference`` below).
    """
    logit_scale = 200.0
    dist_scale = 1.0
    wall_bias = 30.0
    attn = _build_wall_attention_graph(
        logit_scale=logit_scale,
        dist_scale=dist_scale,
        wall_bias=wall_bias,
    )

    walls = _walls_on_circle(n_walls, radius=0.0)  # dist=0 to isolate angular selection
    queries = [_query_from_angle(w["_angle"]) for w in walls]

    inputs, n_pos = _build_input_dict(walls, queries)
    cache = reference_eval(attn, inputs, n_pos)
    out = cache[attn]

    assert out.shape == (n_pos, D_OUTPUT)

    for q_idx in range(n_walls):
        query_row = n_walls + q_idx
        expected = _expected_output(
            walls,
            queries,
            q_idx,
            logit_scale,
            dist_scale,
            wall_bias,
        )
        got = out[query_row].numpy()
        # Correct wall dominates the softmax but adjacent walls leak a
        # small amount; use a tolerance wide enough for the blend and
        # assert the wall_id rounds to the right integer.
        assert np.allclose(got, expected, atol=1e-4), (
            f"n_walls={n_walls}, query {q_idx}: got {got}, expected {expected}"
        )
        # And the actual wall_id rounds to the right integer.
        assert round(got[_OUT_WALL_ID]) == walls[q_idx]["wall_id"], (
            f"n_walls={n_walls}, query {q_idx}: attended wall_id "
            f"{got[_OUT_WALL_ID]} does not round to {walls[q_idx]['wall_id']}"
        )


# ---------------------------------------------------------------------------
# Test 3: softmax sharpness sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_walls,logit_scale,min_correct_weight",
    [
        # Small scenes at a moderate scale are easy: the adjacent logit gap
        # widens with angular separation, so the softmax saturates fast.
        (2, 50.0, 0.99),
        (4, 50.0, 0.99),
        (4, 100.0, 0.999),
        (8, 50.0, 0.97),
        (8, 100.0, 0.99),
        # For 16 walls the angular gap shrinks to ~22.5° which gives an
        # adjacent-wall cos_diff of ~0.076.  At logit_scale=100 that's a
        # 7.6 logit gap → adjacent weight ≈ 5e-4 → correct weight ≈ 0.999.
        (16, 100.0, 0.99),
        (16, 200.0, 0.999),
        # At 32 walls and scale=200 the adjacent gap is ~0.019 cos units,
        # logit gap ~3.9 → correct weight ≈ 0.95.  Scale=500 gets us to ≈ 0.99993.
        (32, 500.0, 0.99),
    ],
)
def test_softmax_weights_match_reference(n_walls, logit_scale, min_correct_weight):
    """Sweep ``(n_walls, logit_scale)`` and verify two things:

    1. The expected softmax weight on the correct wall (the one the
       query ray is pointed at) is at least ``min_correct_weight`` —
       i.e., the hand-designed Q/K actually achieves the target
       sharpness for this scene density and scale.
    2. The Attn node's oracle output matches our reference softmax
       computation to within numerical noise — i.e., the mechanism
       is doing what we think it is.

    The (scale, min_weight) thresholds come from the analytical
    adjacent-wall-gap calculation; if they ever drift it means the
    Attn compute changed semantically.
    """
    wall_bias = 30.0
    attn = _build_wall_attention_graph(
        logit_scale=logit_scale,
        dist_scale=1.0,
        wall_bias=wall_bias,
    )

    walls = _walls_on_circle(n_walls, radius=0.0)
    # Query at wall 0's midpoint — by symmetry, what we learn about
    # wall 0 applies to every wall on the circle.
    queries = [_query_from_angle(walls[0]["_angle"])]

    weights = _expected_softmax_weights(
        walls,
        queries,
        query_idx=0,
        logit_scale=logit_scale,
        dist_scale=1.0,
        wall_bias=wall_bias,
    )
    correct_weight = weights[0]
    assert correct_weight >= min_correct_weight, (
        f"n_walls={n_walls}, logit_scale={logit_scale}: correct-wall "
        f"weight {correct_weight:.6f} below threshold {min_correct_weight}; "
        f"all weights = {weights.tolist()}"
    )

    inputs, n_pos = _build_input_dict(walls, queries)
    cache = reference_eval(attn, inputs, n_pos)
    out = cache[attn]

    expected = _expected_output(
        walls,
        queries,
        query_idx=0,
        logit_scale=logit_scale,
        dist_scale=1.0,
        wall_bias=wall_bias,
    )
    query_row = n_walls  # first query
    got = out[query_row].numpy()
    assert np.allclose(got, expected, atol=1e-4), (
        f"attended output {got} disagrees with reference softmax "
        f"output {expected} at n_walls={n_walls}, scale={logit_scale}"
    )


# ---------------------------------------------------------------------------
# Test 4: distance tiebreak
# ---------------------------------------------------------------------------


def test_distance_tiebreak():
    """Two walls at the same angular midpoint but different distances.
    The closer one should win the softmax.

    This test was the one that originally caught the "logit ties with
    query-self baseline when cos_diff == dist_scale * wall_dist" bug;
    ``wall_bias`` exists specifically to handle this regime, so we
    keep the distances realistic (well above 1.0) and rely on the
    bias to keep both wall logits positive.
    """
    logit_scale = 100.0
    dist_scale = 1.0
    wall_bias = 30.0
    attn = _build_wall_attention_graph(
        logit_scale=logit_scale,
        dist_scale=dist_scale,
        wall_bias=wall_bias,
    )

    # Both walls sit at angle 0, so angular score is identical for both.
    walls = [
        {
            "wall_cos_mid": 1.0,
            "wall_sin_mid": 0.0,
            "wall_dist": 1.0,  # near
            "wall_id": 10.0,
        },
        {
            "wall_cos_mid": 1.0,
            "wall_sin_mid": 0.0,
            "wall_dist": 3.0,  # far
            "wall_id": 20.0,
        },
    ]
    queries = [_query_from_angle(0.0)]

    inputs, n_pos = _build_input_dict(walls, queries)
    cache = reference_eval(attn, inputs, n_pos)
    out = cache[attn]

    query_row = len(walls)
    got = out[query_row].numpy()

    # Logit gap = logit_scale * dist_scale * (3 - 1) = 200 → softmax
    # weight on the far wall ≈ exp(-200) → essentially zero.  The
    # attended wall_id should round to the near wall's id.
    assert round(got[_OUT_WALL_ID]) == 10.0, (
        f"distance tiebreak failed: attended wall_id was {got[_OUT_WALL_ID]}, "
        f"expected 10.0 (near wall)"
    )

    # Also verify the reference softmax gives essentially all weight to
    # the near wall, so we know the test is checking what it thinks.
    weights = _expected_softmax_weights(
        walls,
        queries,
        query_idx=0,
        logit_scale=logit_scale,
        dist_scale=dist_scale,
        wall_bias=wall_bias,
    )
    assert weights[0] > 0.9999, f"near wall weight was {weights[0]}, expected > 0.9999"


# ---------------------------------------------------------------------------
# Test 5: multi-query independence
# ---------------------------------------------------------------------------


def test_multi_query_independence():
    """Four walls + three queries at different angles in one forward
    pass.  Each query must return its own wall, unaffected by the
    presence of the other queries or by the order they appear in.

    This is where causal masking plus zero-K on query rows matters:
    later queries attend to earlier queries with logit=0 and V=(0,0,0),
    which must not contaminate the gathered output.
    """
    logit_scale = 200.0
    dist_scale = 1.0
    wall_bias = 30.0
    attn = _build_wall_attention_graph(
        logit_scale=logit_scale,
        dist_scale=dist_scale,
        wall_bias=wall_bias,
    )

    n_walls = 4
    walls = _walls_on_circle(n_walls, radius=0.0)
    # Queries target walls 0, 2, 1 (intentionally out of order).
    target_walls = [0, 2, 1]
    queries = [_query_from_angle(walls[k]["_angle"]) for k in target_walls]

    inputs, n_pos = _build_input_dict(walls, queries)
    cache = reference_eval(attn, inputs, n_pos)
    out = cache[attn]

    assert out.shape == (n_pos, D_OUTPUT)

    for q_idx, target in enumerate(target_walls):
        query_row = n_walls + q_idx
        got = out[query_row].numpy()
        expected_id = walls[target]["wall_id"]
        assert round(got[_OUT_WALL_ID]) == expected_id, (
            f"query {q_idx} (target wall {target}): attended wall_id "
            f"{got[_OUT_WALL_ID]}, expected {expected_id}"
        )
        # Also check the returned (cos, sin) matches.
        assert abs(got[_OUT_WALL_COS] - walls[target]["wall_cos_mid"]) < 1e-3
        assert abs(got[_OUT_WALL_SIN] - walls[target]["wall_sin_mid"]) < 1e-3


# ---------------------------------------------------------------------------
# Test 6: compiled module matches oracle (probe_graph)
# ---------------------------------------------------------------------------


def test_probe_compiled_matches_oracle():
    """Run the full graph through ``probe_graph`` on a canonical
    4-wall, 2-query config.  The probe compiles the graph into a
    ``HeadlessTransformer`` and verifies every materialised node's
    compiled residual-stream value matches the oracle
    ``Node.compute`` output.

    We use this test sparingly — compilation takes seconds — but it
    catches classes of bugs the oracle alone can't see: bad
    Q/K/V/O packing, residual assignment errors, and the slow-plane
    rotary content head wiring through the compiler at full-width d_head.
    """
    logit_scale = 200.0
    dist_scale = 1.0
    wall_bias = 30.0
    attn = _build_wall_attention_graph(
        logit_scale=logit_scale,
        dist_scale=dist_scale,
        wall_bias=wall_bias,
    )

    n_walls = 4
    walls = _walls_on_circle(n_walls, radius=0.0)
    queries = [
        _query_from_angle(walls[0]["_angle"]),
        _query_from_angle(walls[2]["_angle"]),
    ]
    inputs, n_pos = _build_input_dict(walls, queries)

    report = probe_graph(
        attn,
        input_values=inputs,
        n_pos=n_pos,
        d=512,
        d_head=WALL_D_HEAD,
        verbose=False,
        atol=1e-3,
    )
    assert report.first_divergent is None, (
        f"probe reported divergence:\n{report.format_short()}"
    )
