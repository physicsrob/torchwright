"""Predicate helpers that attach runtime checks to graph nodes.

A check is *node metadata* (``node.checks`` /
``node.claimed_type`` — see ``torchwright.graph.node.Node``), not a
wrapper node: attaching never changes the graph's topology, costs
nothing in the compiled transformer, and the claim survives every
bounds recompute (``refresh_node_caches`` re-applies it — the choke
point of docs/assert_metadata_plan.md).  Each helper attaches and
returns the same node, so graph construction reads naturally::

    from torchwright.graph.asserts import assert_01, assert_strictly_less

    position_onehot = assert_onehot(position_onehot)
    sentinel = assert_strictly_less(real_score, sentinel)

Checks run during reference evaluation (every ``compute`` of a checked
node runs its predicates) and against compiled values during
``debug=True`` forwards and ``probe_compiled``.

Attach-before-consumers convention
----------------------------------
Ops bake gating offsets from input ranges at graph-build time, so a
claim tightens only what is built *after* it is attached.  Every helper
here is meant to be called immediately after the node is constructed,
before anything consumes it (all existing call sites do).  Compile-time
bounds are always right regardless — ``lower()`` recomputes fresh in
topological order.

Naming convention (matches the rest of the codebase)
----------------------------------------------------
* ``bool`` values are ±1 — see ``ops/arithmetic_ops.bool_to_01`` at
  ``torchwright/ops/relu/arithmetic_ops.py:97``.
* ``01`` values are {0, 1} — matches ``bool_to_01``'s output.

Safe placement notes live in each helper's docstring.  The core split
is **pre-attention vs. post-attention**: attention layers average
values under a softmax, which blends one-hot encodings and introduces
PL2D-scale fuzz.  Invariants that only hold pre-attention (exactly-one
one-hots, distinct values) should be asserted before the attention
layer; invariants that hold on aggregated values (``{0, 1}`` bools
after broadcast) take a tolerance large enough to absorb that fuzz.
"""

import torch

from torchwright.graph.misc import Check, Concatenate, Predicate
from torchwright.graph.node import Node, _current_annotation
from torchwright.graph.value_type import NodeValueType, Range, tightened_with

# The midpoint between {0, 1} and ±1 validity encodings: both conventions'
# "valid" value (1.0 or +1.0) sits above this line and their "invalid"
# value (0.0 or -1.0) sits below it, so a single threshold reads either.
_BOOL_VALID_THRESHOLD = 0.5

# A pairwise-distinctness or gap check is vacuous with fewer than two rows
# to compare — there's no pair to violate the invariant.
_MIN_ROWS_FOR_A_PAIR = 2

# Row/column-shaped tensors this module checks against are always 2D:
# (n_pos, d).
_ROW_TENSOR_NDIM = 2


def attach_assert(
    node: Node,
    predicate: Predicate,
    message: str = "",
    *,
    claimed_type: NodeValueType | None = None,
    integer_claim: bool = False,
) -> Node:
    """Attach an assert-kind check (and optional range claim) to ``node``.

    The predicate is appended to ``node.checks``; ``claimed_type`` is
    intersected into the node's running claim and folded into its
    derived caches immediately (claims commute, so attach order is
    irrelevant).  Returns ``node`` for fluent construction.
    """
    node.checks.append(
        Check(
            predicate=predicate,
            message=message,
            kind="assert",
            annotation=_current_annotation.get(),
        )
    )
    changed = False
    if claimed_type is not None:
        node.claimed_type = (
            claimed_type
            if node.claimed_type is None
            else tightened_with(node.claimed_type, claimed_type)
        )
        changed = True
    if integer_claim and not node.integer_claim:
        node.integer_claim = True
        changed = True
    if changed:
        from torchwright.graph.affine_rules import refresh_node_caches

        refresh_node_caches(node)
    return node


def debug_watch(node: Node, predicate: Predicate, message: str = "") -> Node:
    """Attach a watch-kind check to ``node``: prints when ``predicate`` fires."""
    node.checks.append(
        Check(
            predicate=predicate,
            message=message,
            kind="watch",
            annotation=_current_annotation.get(),
        )
    )
    return node


def collect_asserts(output_node: Node) -> list[Node]:
    """Nodes reachable from ``output_node`` carrying assert-kind checks.

    Safe before or after compiling: compilation never mutates the
    source graph, and checks are node metadata, so they are always
    where they were attached.
    """
    return [n for n in _walk(output_node) if any(c.kind == "assert" for c in n.checks)]


def collect_watches(output_node: Node) -> list[Node]:
    """Nodes reachable from ``output_node`` carrying watch-kind checks."""
    return [n for n in _walk(output_node) if any(c.kind == "watch" for c in n.checks)]


def collect_debug_nodes(output_node: Node) -> list[Node]:
    """Nodes reachable from ``output_node`` carrying any checks."""
    return [n for n in _walk(output_node) if n.checks]


def _walk(output_node: Node) -> list[Node]:
    seen: set[int] = set()
    ordered: list[Node] = []
    stack: list[Node] = [output_node]
    while stack:
        node = stack.pop()
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        ordered.append(node)
        stack.extend(node.inputs)
    return ordered


def _format_bad(x: torch.Tensor, mask: torch.Tensor, *, max_show: int = 3) -> str:
    """Summarize the first ``max_show`` positions where ``mask`` is True.

    Indices are into the flattened tensor — the mask is flattened
    *before* ``nonzero`` (a multi-dim ``nonzero`` returns coordinate
    rows, which flattened would be coordinates misread as flat indices,
    pairing wrong values with wrong positions).
    """
    bad_indices = mask.flatten().nonzero(as_tuple=False).flatten().tolist()
    if not bad_indices:
        return "no bad entries"
    head = bad_indices[:max_show]
    more = (
        ""
        if len(bad_indices) <= max_show
        else f" (+{len(bad_indices) - max_show} more)"
    )
    # Show the actual offending values at those indices.
    flat = x.flatten()
    shown = ", ".join(f"[{i}]={flat[i].item():.4f}" for i in head)
    return f"bad at {shown}{more}"


def assert_in_range(
    node: Node,
    lo: float,
    hi: float,
    *,
    atol: float = 1e-3,
) -> Node:
    """Assert every element of ``node`` lies in ``[lo - atol, hi + atol]``.

    **Safe placement**: anywhere.  The ``atol`` default (1e-3) absorbs
    typical post-PL2D noise.  Used for bounds like ``|score| ≤ 100``
    on ``attend_argmin_unmasked`` scores.
    """

    def predicate(x: torch.Tensor) -> tuple:
        bad = (x < lo - atol) | (x > hi + atol)
        if not bad.any():
            return True, ""
        return False, f"expected [{lo}, {hi}] (atol={atol}); {_format_bad(x, bad)}"

    return attach_assert(
        node,
        predicate,
        message=f"values in [{lo}, {hi}]",
        claimed_type=NodeValueType.bounded(lo, hi),
    )


def assert_integer(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert every element rounds to an integer within ``atol``.

    Tightens the ``value_range`` to integer-rounded endpoints of the
    input's existing range (or leaves it unbounded if unknown), and
    marks the node integer-grained (``node.integer_claim`` — the
    univariate collapse pass gates on it).

    **Safe placement**: at any point where the caller can guarantee the
    tensor is numerically integer.
    """

    def predicate(x: torch.Tensor) -> tuple:
        bad = (x - x.round()).abs() > atol
        if not bad.any():
            return True, ""
        return False, f"expected integer (atol={atol}); {_format_bad(x, bad)}"

    # Preserve the input's known range (rounded to integer endpoints)
    # if finite; otherwise leave unbounded.
    r = node.value_type.value_range
    import math

    if math.isfinite(r.lo) and math.isfinite(r.hi):
        claimed = NodeValueType(value_range=Range(math.floor(r.lo), math.ceil(r.hi)))
    else:
        claimed = NodeValueType()
    return attach_assert(
        node,
        predicate,
        message="integer-valued",
        claimed_type=claimed,
        integer_claim=True,
    )


def assert_bool(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert every element is ≈ +1 or ≈ -1 (a "bool" in the codebase convention).

    **Safe placement**: outputs of ``compare``, ``equals_vector``,
    ``bool_any_true``, ``bool_all_true``, ``bool_not``, ``cond_gate``,
    and any ±1-valued node *before* it's averaged by attention.
    """

    def predicate(x: torch.Tensor) -> tuple:
        near_pos = (x - 1.0).abs() <= atol
        near_neg = (x + 1.0).abs() <= atol
        bad = ~(near_pos | near_neg)
        if not bad.any():
            return True, ""
        return False, f"expected ±1 (atol={atol}); {_format_bad(x, bad)}"

    return attach_assert(
        node,
        predicate,
        message="bool (±1)",
        claimed_type=NodeValueType(value_range=Range(-1.0, 1.0)),
    )


def assert_01(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert every element is ≈ 0 or ≈ 1.

    **Safe placement**: outputs of ``bool_to_01``, ``in_range`` flags,
    and post-broadcast {0, 1} vectors such as ``side_P_vec`` after the
    BSP ``attend_mean_where * max_bsp_nodes`` recovery.  Uses a looser
    ``atol`` than ``assert_bool`` because broadcast values accumulate
    more interpolation fuzz than raw ``compare`` outputs.
    """

    def predicate(x: torch.Tensor) -> tuple:
        near_zero = x.abs() <= atol
        near_one = (x - 1.0).abs() <= atol
        bad = ~(near_zero | near_one)
        if not bad.any():
            return True, ""
        return False, f"expected {{0,1}} (atol={atol}); {_format_bad(x, bad)}"

    return attach_assert(
        node,
        predicate,
        message="binary (0/1)",
        claimed_type=NodeValueType(value_range=Range(0.0, 1.0)),
    )


def assert_onehot(node: Node, *, atol: float = 1e-3) -> Node:
    """Assert each row is a one-hot vector: exactly one ≈ 1, rest ≈ 0.

    **Safe placement**: **pre-attention only.**  After an attention
    layer averages one-hot vectors across positions, the result is a
    *distribution*, not a one-hot — use ``assert_01`` on specific
    components instead.  Examples of safe sites: ``position_onehot`` at
    WALL positions, ``sort_rank_onehot`` at SORTED positions, *before*
    either is routed through an attention value.
    """

    def predicate(x: torch.Tensor) -> tuple:
        if x.ndim != _ROW_TENSOR_NDIM:
            return False, f"expected 2D (n_pos, d); got shape {tuple(x.shape)}"
        near_zero = x.abs() <= atol
        near_one = (x - 1.0).abs() <= atol
        elementwise_ok = near_zero | near_one
        rows_elem_ok = elementwise_ok.all(dim=-1)
        row_sums = x.sum(dim=-1)
        rows_sum_ok = (row_sums - 1.0).abs() <= atol
        rows_ok = rows_elem_ok & rows_sum_ok
        if rows_ok.all():
            return True, ""
        bad_rows = (~rows_ok).nonzero(as_tuple=False).flatten().tolist()
        max_show = 3
        head = bad_rows[:max_show]
        more = (
            "" if len(bad_rows) <= max_show else f" (+{len(bad_rows) - max_show} more)"
        )
        summary = ", ".join(f"row {i} sum={row_sums[i].item():.3f}" for i in head)
        return False, f"not one-hot (atol={atol}); {summary}{more}"

    return attach_assert(
        node,
        predicate,
        message="one-hot",
        claimed_type=NodeValueType(value_range=Range(0.0, 1.0)),
    )


def assert_matches_value_type(
    node: Node,
    vt: NodeValueType,
    *,
    atol: float = 1e-3,
) -> Node:
    """Assert the tensor satisfies the value_range of ``vt`` within ``atol``.

    Checks that no element lies outside [lo - atol, hi + atol].

    ``vt`` becomes the node's range claim, so downstream static
    analysis sees the promoted type.
    """
    import math

    r = vt.value_range

    def predicate(x: torch.Tensor) -> tuple:
        if math.isfinite(r.lo) or math.isfinite(r.hi):
            bad = torch.zeros_like(x, dtype=torch.bool)
            if math.isfinite(r.lo):
                bad = bad | (x < r.lo - atol)
            if math.isfinite(r.hi):
                bad = bad | (x > r.hi + atol)
            if bad.any():
                return False, f"range {r} (atol={atol}); {_format_bad(x, bad)}"
        return True, ""

    return attach_assert(
        node,
        predicate,
        message=f"matches {vt}",
        claimed_type=vt,
    )


def assert_strictly_less(
    a: Node,
    b: Node,
    *,
    margin: float = 0.0,
) -> Node:
    """Assert every element of ``a`` is strictly less than ``b``'s matching element.

    Both nodes must have the same width.  Returns a checked projection
    of ``b`` (the upper bound) — the natural thing to thread into
    downstream graph construction when the point is to guarantee the
    sentinel stays above the real values.

    The predicate needs both values at once, so the check attaches to a
    ``Concatenate([a, b])`` composite; a trailing ``Linear`` projects
    ``b``'s slice back out for downstream use (value-preserving, so the
    projection inherits ``b``'s static type as its own claim).

    **Safe placement**: anywhere.  Used for sentinel ordering (e.g.
    ``bsp_sentinel`` < ``nonwall_sentinel``, or real ranks <
    sentinel).  For pre-attention ordering between positions, gate
    inputs with ``select(is_wall, ...)`` first.
    """
    if len(a) != len(b):
        raise ValueError(
            f"assert_strictly_less: width mismatch (a={len(a)}, b={len(b)})"
        )
    width = len(a)

    def predicate(x: torch.Tensor) -> tuple:
        a_val = x[:, :width]
        b_val = x[:, width:]
        diff = b_val - a_val
        bad = diff <= margin
        if not bad.any():
            return True, ""
        i, j = bad.nonzero(as_tuple=False)[0].tolist()
        return False, (
            f"expected a < b (margin={margin}); "
            f"row={i} col={j}: a={a_val[i, j].item():.4f}, "
            f"b={b_val[i, j].item():.4f}"
        )

    composite = attach_assert(
        Concatenate([a, b]),
        predicate,
        message=f"strictly_less (margin={margin})",
    )
    from torchwright.graph import Linear

    proj = torch.zeros(len(a) + len(b), len(b))
    for i in range(len(b)):
        proj[len(a) + i, i] = 1.0
    projected = Linear(composite, proj, name="strictly_less_b")
    if b.value_type != NodeValueType.unknown():
        return assert_matches_value_type(projected, b.value_type)
    return projected


def assert_unique_values(
    node: Node,
    *,
    margin: float = 0.5,
) -> Node:
    """Assert every pair of distinct components in a row differs by at least ``margin``.

    Useful for sort-score vectors (like a stacked ``bsp_rank`` across
    WALL positions) where any two tied scores would make the downstream
    ``attend_argmin_unmasked`` softmax-blend their values.  ``margin``
    defaults to 0.5 — bigger than typical PL2D fuzz, small enough not
    to conflict with the real 1.0 rank spacing.

    **Safe placement**: **pre-attention only.**  The whole point is to
    catch ties *before* they can blend.
    """

    def predicate(x: torch.Tensor) -> tuple:
        if x.ndim != _ROW_TENSOR_NDIM:
            return False, f"expected 2D (n_pos, d); got shape {tuple(x.shape)}"
        _n_pos, d = x.shape
        if d < _MIN_ROWS_FOR_A_PAIR:
            return True, ""
        # Pairwise absolute differences within each row.
        # x.unsqueeze(-1) - x.unsqueeze(-2) has shape (n_pos, d, d).
        diffs = (x.unsqueeze(-1) - x.unsqueeze(-2)).abs()
        # Mask out the diagonal (self-pairs).
        eye = torch.eye(d, dtype=torch.bool, device=x.device).unsqueeze(0)
        bad = (diffs < margin) & ~eye
        if not bad.any():
            return True, ""
        row_idx, i, j = bad.nonzero(as_tuple=False)[0].tolist()
        return False, (
            f"duplicate values (margin={margin}); "
            f"row={row_idx}: [{i}]={x[row_idx, i].item():.4f} "
            f"≈ [{j}]={x[row_idx, j].item():.4f}"
        )

    return attach_assert(node, predicate, message=f"unique values (margin={margin})")


def assert_distinct_across(
    value: Node,
    where: Node,
    *,
    margin: float = 0.5,
) -> Node:
    """Assert per-position ``value`` is pairwise-distinct across ``where ≈ 1`` rows.

    Use this for cross-position uniqueness invariants like "``bsp_rank``
    values at WALL positions are all different" — the precondition that
    keeps a downstream ``attend_argmin_unmasked`` softmax concentrated
    on a single key.

    Rows with ``where.squeeze(-1) ≤ 0.5`` are ignored (the predicate
    sees only the valid subset).  This makes the check tolerant of both
    ±1 bool validity (where ``≥ 0.5`` maps to "valid") and {0, 1}
    indicator validity.

    Pairwise comparison uses L∞ distance on the ``d_value``-wide vectors
    (for the common 1-wide scalar case, L∞ is just absolute difference);
    any pair closer than ``margin`` triggers failure.  Default margin
    (0.5) is larger than typical PL2D fuzz and smaller than the natural
    spacing of integer ranks (1.0).

    **Safe placement**: pre- or post-attention.  Prefer upstream of the
    attention that consumes ``value`` so a tie is caught before it
    can blend.

    Implementation mirrors :func:`assert_strictly_less` — the check
    attaches to a ``Concatenate`` composite of the two inputs; a
    trailing ``Linear`` projects the checked composite back to
    ``value``'s width for downstream use.
    """
    d_value = len(value)
    d_where = len(where)

    def predicate(x: torch.Tensor) -> tuple:
        val = x[:, :d_value]
        valid = x[:, d_value : d_value + d_where]
        if d_where == 1:
            mask = valid.squeeze(-1) > _BOOL_VALID_THRESHOLD
        else:
            # Multi-wide validity collapses to "any slot ≥ 0.5".
            mask = (valid > _BOOL_VALID_THRESHOLD).any(dim=-1)
        rows = val[mask]
        if rows.shape[0] < _MIN_ROWS_FOR_A_PAIR:
            return True, ""
        diffs = (rows.unsqueeze(1) - rows.unsqueeze(0)).abs().max(dim=-1).values
        eye = torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)
        bad = (diffs < margin) & ~eye
        if not bad.any():
            return True, ""
        i, j = bad.nonzero(as_tuple=False)[0].tolist()
        return False, (
            f"valid-subset rows {i},{j} within margin={margin}: "
            f"{rows[i].tolist()} vs {rows[j].tolist()}"
        )

    composite = attach_assert(
        Concatenate([value, where]),
        predicate,
        message=f"distinct_across (margin={margin})",
    )
    from torchwright.graph import Linear

    proj = torch.zeros(d_value + d_where, d_value)
    for i in range(d_value):
        proj[i, i] = 1.0
    projected = Linear(composite, proj, name="distinct_across_value")
    if value.value_type != NodeValueType.unknown():
        return assert_matches_value_type(projected, value.value_type)
    return projected


def assert_score_gap_at_least(
    score: Node,
    where: Node,
    *,
    margin: float = 1.0,
) -> Node:
    """Assert the two smallest valid ``score`` values differ by at least ``margin``.

    Resolvability invariant for ``attend_argmin_unmasked``.  With
    ``_QUERY_GAIN = 8``, the softmax concentrates ≥99.9 % on the lowest
    score only when the rank-1 vs rank-2 gap exceeds
    ``ln(999) / 8 ≈ 0.864``.  Pair this assert with the score node
    immediately upstream of the argmin to catch precision regressions
    that close the gap below softmax resolution — the failure mode
    behind the historical angle-192 sort-concentration regression.

    The default margin of ``1.0`` matches the integer-score invariant
    that all current callers uphold (BSP rank, digit, slot index —
    gaps ≥ 1).  Tighten only if a caller deliberately produces
    sub-unit score gaps.

    Differs from :func:`assert_distinct_across`:
      * checks only the *tightest* pair, not all O(N²) pairs;
      * uses a tighter default margin calibrated to the ``_QUERY_GAIN``
        floor.

    Vacuous for <2 valid rows (mask sums to 0 or 1).
    """
    d_value = len(score)
    d_where = len(where)

    def predicate(x: torch.Tensor) -> tuple:
        val = x[:, :d_value]
        valid = x[:, d_value : d_value + d_where]
        mask = (
            valid.squeeze(-1) > _BOOL_VALID_THRESHOLD
            if d_where == 1
            else (valid > _BOOL_VALID_THRESHOLD).any(dim=-1)
        )
        rows = val[mask]
        if rows.shape[0] < _MIN_ROWS_FOR_A_PAIR:
            return True, ""
        flat = rows.squeeze(-1) if d_value == 1 else rows.min(dim=-1).values
        sorted_scores, order = torch.sort(flat)
        gap = (sorted_scores[1] - sorted_scores[0]).item()
        if gap >= margin:
            return True, ""
        i, j = order[0].item(), order[1].item()
        return False, (
            f"min-gap {gap:.5f} < margin={margin}; "
            f"rank-1 row [{i}]={sorted_scores[0].item():.5f}, "
            f"rank-2 row [{j}]={sorted_scores[1].item():.5f}"
        )

    composite = attach_assert(
        Concatenate([score, where]),
        predicate,
        message=f"score_gap_at_least (margin={margin})",
    )
    from torchwright.graph import Linear

    proj = torch.zeros(d_value + d_where, d_value)
    for i in range(d_value):
        proj[i, i] = 1.0
    projected = Linear(composite, proj, name="score_gap_value")
    if score.value_type != NodeValueType.unknown():
        return assert_matches_value_type(projected, score.value_type)
    return projected


def assert_picked_from(
    result: Node,
    values: Node,
    keys: Node,
    *,
    atol: float = 1e-2,
) -> Node:
    """Assert an attention output matches exactly one valid per-position value row.

    For a pick-style attention (``attend_argmin_unmasked`` or
    ``attend_argmax_dot``), the output at every query row should equal
    ``values[k]`` for some key row ``k`` where ``keys ≈ 1`` — a blend
    of two or more values indicates a softmax that failed to
    concentrate.  That's exactly the angle-192 rendering artifact in
    DOOM: the compiled softmax returned a weighted average of walls
    instead of one wall.

    ``result`` and ``values`` must share width ``d`` (the attention
    op's contract — output width equals value width).  ``keys`` is a
    (n_pos, 1) validity flag; rows with ``keys > 0.5`` are treated
    as valid candidate picks.

    Predicate: for every query row ``q``, compute the L∞ distance from
    ``result[q]`` to every valid ``values[k]`` row; fail if the minimum
    exceeds ``atol``.  Default ``atol`` (1e-2) is loose enough to
    absorb PL-softmax jitter on clean picks and tight enough to catch
    the ~3% blend we saw at angle 192 (which produced 0.15-unit drift
    on magnitude-5 values).

    **Safe placement**: **post-attention, compiled-side only.**
    Reference math at high match-gain is exact; only the compiled
    piecewise-linear softmax drifts.  Expect this assert to fire
    through :func:`torchwright.debug.probe.check_asserts_on_compiled`,
    not during :func:`reference_eval`.
    """
    d_r = len(result)
    d_v = len(values)
    d_k = len(keys)
    if d_r != d_v:
        raise ValueError(
            f"assert_picked_from: result/values width mismatch "
            f"(result={d_r}, values={d_v})"
        )

    def predicate(x: torch.Tensor) -> tuple:
        res = x[:, :d_r]
        vals = x[:, d_r : d_r + d_v]
        keys_t = x[:, d_r + d_v : d_r + d_v + d_k]
        mask = (
            keys_t.squeeze(-1) > _BOOL_VALID_THRESHOLD
            if d_k == 1
            else (keys_t > _BOOL_VALID_THRESHOLD).any(dim=-1)
        )
        n_pos = res.shape[0]
        valid_idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        if valid_idxs.numel() == 0:
            # No position in this batch is a valid key — which in the
            # caller's convention (``keys`` is the per-row is_wall /
            # is_render / is_tex_col boolean, same flag as query-row
            # activity) means every query row is itself inactive.  The
            # downstream consumer gates the result away (e.g. via
            # ``select(is_sorted, ...)``), so there's nothing to check.
            # This branch hits in decode mode at step positions whose
            # type boolean is 0 (e.g. a PLAYER_X step for a sort-stage
            # assert keyed on is_wall).
            print(
                f"  [picked_from] skip: no valid key positions in batch "
                f"of {n_pos} — every query inactive"
            )
            return True, ""
        valid_vals = vals[valid_idxs]
        # (n_pos, 1, d) - (1, n_keys, d) → (n_pos, n_keys, d); max over d = L∞.
        dists = (res.unsqueeze(1) - valid_vals.unsqueeze(0)).abs().max(dim=-1).values
        min_dists, argmin_k = dists.min(dim=-1)
        # Skip query rows that have no valid key in their causal window —
        # the softmax has nothing legitimate to pick from there, the
        # downstream consumer is expected to gate the result away (e.g.
        # via ``select(is_sorted, ...)``), and asserting against the
        # valid-key set fires on a value the consumer never reads.
        first_valid_idx = int(valid_idxs.min().item())
        query_rows = torch.arange(n_pos, device=x.device)
        has_visible_valid = query_rows >= first_valid_idx
        # Also skip query rows that are themselves inactive (``mask[q]``
        # is False).  The caller's convention is that ``keys`` doubles
        # as the per-row type flag; an inactive query is the type-
        # isolation pattern and its attention output is meaningless.
        bad = (min_dists > atol) & has_visible_valid & mask
        if not bad.any():
            return True, ""
        q = int(bad.nonzero(as_tuple=False)[0].item())
        return False, (
            f"query row {q}: result doesn't match any value row "
            f"within atol={atol}; closest valid key #{argmin_k[int(q)].item()} "
            f"at L∞ distance {min_dists[int(q)].item():.4f}"
        )

    composite = attach_assert(
        Concatenate([result, values, keys]),
        predicate,
        message=f"attention picked one (atol={atol})",
    )
    from torchwright.graph import Linear

    proj = torch.zeros(d_r + d_v + d_k, d_r)
    for i in range(d_r):
        proj[i, i] = 1.0
    projected = Linear(composite, proj, name="picked_from_result")
    if result.value_type != NodeValueType.unknown():
        return assert_matches_value_type(projected, result.value_type)
    return projected


def assert_softmax_hardness(
    attn_node: Node,
    threshold: float,
) -> Node:
    """Assert the attention's softmax concentrates on the winning key.

    At least ``threshold`` mass must land on the winning key at every
    query position.  Attaches a check whose predicate re-derives the
    softmax weights from the Attn node's Q/K matrices and input
    values, then checks ``max(weights, dim=keys) >= threshold`` per
    query.

    ``threshold=0.99`` means 99% of the attention mass must land on a
    single key.  This directly catches the blending failure mode where
    the softmax spreads mass across multiple keys and the output is a
    garbage weighted average.

    **Safe placement**: on any hard-selection attention output (argmin,
    argmax, argmin_unmasked, etc.).  Not appropriate for
    ``attend_mean_where`` which intentionally spreads mass.
    """
    from torchwright.graph.attn import CAUSAL_MASK_SENTINEL, Attn

    if not isinstance(attn_node, Attn):
        raise TypeError(
            f"assert_softmax_hardness expects an Attn node, got "
            f"{type(attn_node).__name__}"
        )
    attn = attn_node

    query_in_node = attn.inputs[0]
    key_in_node = attn.inputs[1]
    query_matrix = attn.query_matrix.clone()
    key_matrix = attn.key_matrix.clone()
    d_qi = len(query_in_node)
    d_ki = len(key_in_node)
    d_out = len(attn_node)

    attn_name = attn.name or attn.annotation or "attn"

    def predicate(x: torch.Tensor) -> tuple:
        qi = x[:, :d_qi]
        ki = x[:, d_qi : d_qi + d_ki]
        n_pos = qi.shape[0]

        Q = qi @ query_matrix.to(x.device)
        K = ki @ key_matrix.to(x.device)
        logits = Q @ K.t()

        mask = torch.triu(torch.ones(n_pos, n_pos, device=x.device), diagonal=1)
        logits = torch.where(
            mask == 1,
            torch.full_like(logits, CAUSAL_MASK_SENTINEL),
            logits,
        )
        weights = torch.softmax(logits, dim=1)
        max_weights = weights.max(dim=1).values

        # Skip query positions where the attention has nothing meaningful
        # to concentrate on.  Two cases that both produce a uniform
        # softmax over the causal window — neither indicates a real
        # blending bug because the downstream consumer gates the output
        # away (e.g. ``select(is_render, ...)``):
        #
        #   (a) The query input itself has near-zero norm.  The
        #       documented "type isolation by zero query/key" pattern
        #       (see ``attend_argmax_dot``'s docstring): callers
        #       ``cond_gate`` the query at non-participating positions.
        #       Residual noise from ``cond_gate``'s additive cancellation
        #       is on the order of 1e-6, so ``qi.norm > 1e-3`` cleanly
        #       separates noise from a real query (one-hots, scores, and
        #       ±1-encoded gates are all order-1).
        #
        #   (b) The query input is non-zero but every causally-visible
        #       key row has near-zero K norm (e.g. the SORTED-stage
        #       argmin-above-integer attention at a query that precedes
        #       any WALL token).  Without a non-trivial K, Q · K is
        #       dominated by noise and the softmax is forced uniform.
        residual_noise_floor = 1e-3
        query_active = qi.norm(dim=1) > residual_noise_floor
        if query_active.any():
            k_norms = K.norm(dim=1)  # shape (n_pos,)
            # Cumulative max over causally-visible keys [0..q] inclusive.
            cumulative_max_k = torch.cummax(k_norms, dim=0).values
            keys_meaningful = cumulative_max_k > residual_noise_floor
            query_active = query_active & keys_meaningful

        min_hw = max_weights.min().item()
        argmax_positions = weights.argmax(dim=1).tolist()
        active_count = int(query_active.sum().item())
        print(
            f"  [hardness] {attn_name}: "
            f"min={min_hw:.6f} mean={max_weights.mean().item():.6f} "
            f"(threshold={threshold}, n_pos={n_pos}, "
            f"active_queries={active_count}, "
            f"selected_positions={argmax_positions})"
        )

        bad = (max_weights < threshold) & query_active
        if not bad.any():
            return True, ""
        q = int(bad.nonzero(as_tuple=False)[0].item())
        return False, (
            f"softmax not hard enough at query {q}: "
            f"max_weight={max_weights[q].item():.6f} < {threshold}; "
            f"top-3 weights: {weights[q].topk(min(3, n_pos)).values.tolist()}"
        )

    composite = attach_assert(
        Concatenate([query_in_node, key_in_node, attn_node]),
        predicate,
        message=f"softmax_hardness >= {threshold}",
    )
    from torchwright.graph import Linear

    proj = torch.zeros(d_qi + d_ki + d_out, d_out)
    for i in range(d_out):
        proj[d_qi + d_ki + i, i] = 1.0
    projected = Linear(composite, proj, name="hardness_checked")
    if attn_node.value_type != NodeValueType.unknown():
        return assert_matches_value_type(projected, attn_node.value_type)
    return projected
