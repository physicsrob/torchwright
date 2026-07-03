"""The swish-machine op library, built to ``docs/ops_plain_english.md``.

Importing from this package *is* the machine choice: every op here
returns swish-activation :class:`~torchwright.graph.ffn.FFN` nodes, and
the compiler's uniformity check (all FFNs in a graph share one
activation) is the backstop against accidental mixing with the frozen
ReLU library in ``torchwright/ops/relu/``.  No mode flags, no
``activation`` parameter anywhere in op code.

The hinge-sharpening constant ``scale`` lives in
``torchwright/ops/const.py`` (machine-neutral level — the compiler's
weight writer imports it too).
"""

from torchwright.ops.swiglu.arithmetic_ops import (
    abs,
    ceil_int,
    clamp,
    compare,
    floor_int,
    min,
    mod_const,
    multiply,
    piecewise_linear,
    reciprocal,
    square,
    thermometer_floor_div,
)
from torchwright.ops.swiglu.global_recency import global_position_from_bos
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import (
    broadcast_select,
    dynamic_extract,
    in_range,
    map_to_table,
    select,
    switch,
    table_lookup_2d,
)
from torchwright.ops.swiglu.onehot_table import onehot_lookup
from torchwright.ops.swiglu.scalar_encoding import scalar_to_embedding
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn
