"""Range printer: two-level autoregressive loop derisking example.

Demonstrates the core pattern for wall-column render tokens: a transformer
iterates through items in order, and for each item iterates through a range
of column values [lo, hi).  The outer loop uses attend_argmin_unmasked with
a mask that advances only when the inner loop completes.

Phase 1 (Prefill): N ITEM tokens carrying (item_index, range_lo, range_hi).
Phase 2 (Print):   Autoregressive PRINT tokens.  Each step emits one column
                    value and computes feedback for the next step.

Token sequence:  ITEM_0  ITEM_1  ...  ITEM_{N-1}  PRINT  PRINT  ...

Example: items with ranges [2,5), [7,9), [0,3)
         output sequence:   2, 3, 4, 7, 8, 0, 1, 2
"""

import torch

from torchwright.graph import Concatenate, Linear, Node
from torchwright.graph.asserts import assert_01, assert_integer, assert_onehot
from torchwright.graph.spherical_codes import index_to_vector
from torchwright.ops.arithmetic_ops import (
    add,
    add_const,
    add_scaled_nodes,
    compare,
    subtract,
)
from torchwright.ops.attention_ops import attend_argmin_unmasked
from torchwright.ops.inout_nodes import (
    create_input,
    create_literal_value,
    create_rope_config,
)
from torchwright.ops.logic_ops import bool_all_true, bool_not, equals_vector
from torchwright.ops.map_select import in_range, select

# ---------------------------------------------------------------------------
# Token types
# ---------------------------------------------------------------------------
TOKEN_ITEM = 0
TOKEN_PRINT = 1
E8_ITEM = index_to_vector(TOKEN_ITEM)
E8_PRINT = index_to_vector(TOKEN_PRINT)

D_TOKEN_TYPE = 8
MAX_ITEMS = 8
_SENTINEL_SCORE = 99.0

# RoPE substrate. The item-selection head is a content head whose match width
# is W = 1 + max_items (score col + one mask/position-onehot column per item),
# and RoPE places those W columns on the slowest d_head/2 planes, so it needs
# d_head >= 2*W. At the default MAX_ITEMS=8 that is d_head >= 18; D_HEAD=32
# clears it and matches the d_head the tests compile at.
D_HEAD = 32
MAX_POSITIONS = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_from(
    node: Node,
    d_total: int,
    start: int,
    width: int,
    name: str,
) -> Node:
    """Slice *width* columns starting at *start* from a *d_total*-wide node."""
    m = torch.zeros(d_total, width)
    for i in range(width):
        m[start + i, i] = 1.0
    return Linear(node, m, name=name)


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_range_printer_graph(
    max_items: int = MAX_ITEMS,
) -> Node:
    """Build the range_printer computation graph.

    Returns ``output_node``.
    """
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)

    # --- Inputs --------------------------------------------------------
    token_type = create_input("token_type", D_TOKEN_TYPE, value_range=(-1.0, 1.0))
    item_index = create_input("item_index", 1, value_range=(0.0, float(max_items)))
    range_lo = create_input("range_lo", 1, value_range=(-999.0, 999.0))
    range_hi = create_input("range_hi", 1, value_range=(-999.0, 999.0))
    print_mask = create_input("print_mask", max_items, value_range=(0.0, 1.0))
    col = create_input("col", 1, value_range=(0.0, 255.0))
    is_new_item = create_input("is_new_item", 1, value_range=(-1.0, 1.0))

    # --- Token type detection ------------------------------------------
    is_item = equals_vector(token_type, E8_ITEM)

    # --- Score (ITEM positions get item_index; others get sentinel) -----
    sentinel = create_literal_value(
        torch.tensor([_SENTINEL_SCORE]),
        name="sentinel",
    )
    score = assert_integer(select(is_item, item_index, sentinel))

    # --- Position one-hot ({0,1}, width max_items) ---------------------
    item_index_p1 = add_const(item_index, 1.0)
    onehot_bool = in_range(item_index, item_index_p1, max_items)
    ones_oh = create_literal_value(torch.ones(max_items), name="ones_oh")
    position_onehot = assert_onehot(add_scaled_nodes(0.5, onehot_bool, 0.5, ones_oh))

    # --- Value to retrieve from selected item --------------------------
    item_value = Concatenate([range_lo, range_hi, position_onehot])
    d_val = 2 + max_items

    # --- Attention: select first unmasked item -------------------------
    selected = attend_argmin_unmasked(
        rope,
        score=score,
        mask_vector=assert_01(print_mask),
        position_onehot=position_onehot,
        value=item_value,
    )

    sel_lo = _extract_from(selected, d_val, 0, 1, "sel_lo")
    sel_hi = _extract_from(selected, d_val, 1, 1, "sel_hi")
    sel_onehot = _extract_from(selected, d_val, 2, max_items, "sel_onehot")

    # --- State machine -------------------------------------------------
    # Active column: new item reads lo from attention, otherwise from fb.
    active_col = select(is_new_item, sel_lo, col)

    # Inner-loop test: does active_col + 1 still fall within [lo, hi)?
    next_col_val = add_const(active_col, 1.0)
    remaining = subtract(sel_hi, next_col_val)
    inner_continues = compare(remaining, 0.5)  # +1 if hi - next > 0.5
    inner_done = bool_not(inner_continues)  # +1 when range exhausted

    # Mask update: OR in the selected item when inner loop finishes.
    mask_with_new = add(print_mask, sel_onehot)
    next_mask = select(inner_done, mask_with_new, print_mask)

    # Done flag: all items masked?
    mask_sum = Linear(
        mask_with_new,
        torch.ones(max_items, 1),
        name="mask_sum",
    )
    all_done = compare(mask_sum, max_items - 0.5)
    done_flag = bool_all_true([inner_done, all_done])

    # Next-step feedback
    next_is_new_item = inner_done  # +1 = advancing to new item
    zero_col = create_literal_value(torch.tensor([0.0]), name="zero_col")
    next_col_output = select(inner_continues, next_col_val, zero_col)

    # --- Output --------------------------------------------------------
    output = Concatenate(
        [
            active_col,  # 0:   the column value emitted this step
            done_flag,  # 1:   +1 when all items exhausted
            next_mask,  # 2:   updated mask for next step
            next_col_output,  # 2+N: next col value
            next_is_new_item,  # 3+N: next is_new_item flag
        ]
    )
    return output


# --- Output index helpers (for host-side parsing) ----------------------


def out_active_col() -> int:
    return 0


def out_done_flag() -> int:
    return 1


def out_feedback_slice(max_items: int) -> slice:
    """Slice covering [next_mask, next_col, next_is_new_item]."""
    return slice(2, 2 + max_items + 2)
