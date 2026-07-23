"""One-hot-native lookup table: the compiled weight matrix *is* the table.

``onehot_lookup`` is the one-hot counterpart of
:func:`torchwright.ops.relu.map_select.map_to_table`.  ``map_to_table`` does a soft
nearest-key dot-product match that only resolves because the spherical-code
embeddings are geometrically well separated and a sharpness term concentrates
onto the winner.  ``onehot_lookup`` instead *assumes* its inputs are one-hot
blocks and turns the lookup into exact integer counting, with two shapes:

* **Single one-hot input** (the key is one one-hot vector): the lookup is a
  plain :class:`~torchwright.graph.Linear` whose row ``k`` holds the value for
  token ``k`` (``default`` for absent tokens).  ``y = inp @ W`` copies the
  selected row — a literal selection matrix, no ReLU, exact.

* **Several one-hot inputs concatenated** (e.g. ``digit ⊕ digit ⊕ carry``):
  one hidden ReLU unit per table row, firing only when *every* block matches.
  For an input ``x`` (a concatenation of one-hots) and a key row that is also
  a concatenation of one-hots, ``x · key`` counts the agreeing blocks — an
  integer in ``0..n_blocks``.  It equals ``n_blocks`` only for the exact key
  and is ``≤ n_blocks - 1`` for every other input, so a bias of
  ``-(n_blocks - 0.5)`` puts the ReLU at exactly ``0.5`` for the unique winner
  and ``0`` for everyone else — a uniform margin of ``0.5`` on each side, no
  tuning.  The first ``Linear``'s rows are literally the keys; the second
  ``Linear``'s rows are literally ``2 · (value − default)``.

Because exactly one row fires (or none → ``default``), the output is always
one of the table's value vectors, so the claimed value range is the *tight*
``[min, max]`` over the values and the default — not ``map_to_table``'s
pessimistic ``default ± Σ|value − default|`` widening, which is what blew up
the interval arithmetic through a long chain of lookups.
"""

import torch

from torchwright.graph import Linear, Node
from torchwright.graph.asserts import assert_matches_value_type
from torchwright.graph.value_type import NodeValueType, Range
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear

_ONEHOT_ATOL = 1e-6


def _count_one_hot_blocks(key: torch.Tensor) -> int:
    """Number of ``1`` entries in ``key``, after checking it is all 0/1.

    The count is the number of one-hot blocks concatenated in the key — the
    ``n_blocks`` the AND-of-matches threshold is sized against.
    """
    near_zero = key.abs() <= _ONEHOT_ATOL
    near_one = (key - 1.0).abs() <= _ONEHOT_ATOL
    if not bool((near_zero | near_one).all()):
        raise ValueError(
            "onehot_lookup keys must be concatenations of one-hot blocks "
            "(every entry 0 or 1); got a key with an entry that is neither"
        )
    return int(near_one.sum().item())


def onehot_lookup(
    inp: Node,
    key_to_value: dict[torch.Tensor, torch.Tensor],
    default: torch.Tensor,
) -> Node:
    """Map a one-hot (or concatenation of one-hots) input to a lookup table.

    Args:
        inp: Node whose value is a one-hot block, or several one-hot blocks
            concatenated (e.g. ``concat([digit_a, digit_b, carry])``).
        key_to_value: The table.  Every key must be a 0/1 vector of width
            ``len(inp)`` with the same number of ones (one per block); every
            value must have width ``len(default)``.
        default: Value returned for any input that matches no key.

    Returns:
        Node holding the looked-up value, carrying a tight ``[min, max]``
        value range over the table's values and the default.
    """
    if not key_to_value:
        raise ValueError("onehot_lookup requires a non-empty table")

    d_key = len(inp)
    d_value = len(default)
    for key in key_to_value:
        if len(key) != d_key:
            raise ValueError(
                f"onehot_lookup key width {len(key)} != input width {d_key}"
            )
    for value in key_to_value.values():
        if len(value) != d_value:
            raise ValueError(
                f"onehot_lookup value width {len(value)} != default width {d_value}"
            )

    block_counts = {_count_one_hot_blocks(key) for key in key_to_value}
    if len(block_counts) != 1:
        raise ValueError(
            f"onehot_lookup keys must all have the same number of one-hot "
            f"blocks; got {sorted(block_counts)}"
        )
    n_blocks = block_counts.pop()
    if n_blocks < 1:
        raise ValueError("onehot_lookup keys must have at least one one-hot block")

    # Tight output range: the output is always exactly one value row (or the
    # default), so [min, max] over them all bounds every element.
    stacked = torch.stack(
        [v.to(torch.float32) for v in key_to_value.values()]
        + [default.to(torch.float32)]
    )
    out_range = Range(float(stacked.min().item()), float(stacked.max().item()))

    if n_blocks == 1:
        # A single one-hot selects one row: the lookup *is* a selection
        # matrix.  Start every row at ``default`` (so absent tokens map to
        # the default), then overwrite the present keys' rows with their
        # values.  ``y = inp @ W`` copies the row the one-hot selects.
        weight = default.to(torch.float32).unsqueeze(0).repeat(d_key, 1).clone()
        for key, value in key_to_value.items():
            row = int((key > 0.5).nonzero(as_tuple=False)[0].item())
            weight[row] = value.to(torch.float32)
        result: Node = Linear(inp, weight, name="onehot_lookup_select")
    else:
        # One hidden unit per row.  match(row) = inp · key (agreeing blocks);
        # bias -(n_blocks - 0.5) leaves only the exact-match unit positive, at
        # 0.5.  output_proj scales that 0.5 spike back to (value - default).
        d_hidden = len(key_to_value)
        input_proj = torch.zeros(d_hidden, d_key)
        input_bias = torch.full((d_hidden,), -(n_blocks - 0.5))
        output_proj = torch.zeros(d_hidden, d_value)
        default_f = default.to(torch.float32)
        for i, (key, value) in enumerate(key_to_value.items()):
            input_proj[i] = key.to(torch.float32)
            output_proj[i] = 2.0 * (value.to(torch.float32) - default_f)
        result = linear_relu_linear(
            input_node=inp,
            input_proj=input_proj,
            input_bias=input_bias,
            output_proj=output_proj,
            output_bias=default_f,
            name="onehot_lookup",
        )

    return assert_matches_value_type(result, NodeValueType(value_range=out_range))
