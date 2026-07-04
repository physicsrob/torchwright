# Machine-selection boundary: `torchwright.ops.relu` is today's op
# library (moved, frozen); `torchwright.ops.swiglu` is the swish-machine
# library.  The import path *is* the machine choice — no mode flags.
# Machine-neutral code lives at this level: `const`, `inout_nodes`,
# `linear` (purely linear ops — no MLP sublayer, so no activation
# choice), `attention_ops` (attention hardware), and `_math` (shared
# pure-math helpers).  Both machines build on it.
#
# Compatibility aliases: every pre-split import path
# (``torchwright.ops.<mod>``) must keep working with preserved module
# identity — some relu-module cross-imports use those paths, and so do
# torchwright_doom files and the frozen `tests/ops/*` baseline.
# Registration order is the modules' dependency order, and each alias is
# registered *before* the next module loads, so the moved files resolve
# their cross-imports through `sys.modules`.
import importlib as _importlib
import sys as _sys

_RELU_MODULES = (
    "linear_relu_linear",
    "arithmetic_ops",
    "logic_ops",
    "map_select",
    "onehot_table",
    "scalar_encoding",
    "embedding_arithmetic",
    "marker_count",
    "global_recency",
    "sequence_ops",
)
for _name in _RELU_MODULES:
    _mod = _importlib.import_module(f"torchwright.ops.relu.{_name}")
    _sys.modules[f"torchwright.ops.{_name}"] = _mod
    globals()[_name] = _mod
del _importlib, _sys, _name, _mod

# Linear (machine-neutral)
from torchwright.ops.linear import (
    add,
    add_const,
    add_scaled_nodes,
    bool_to_01,
    concat,
    multiply_const,
    negate,
    subtract,
    sum_nodes,
)

# Arithmetic
from torchwright.ops.relu.arithmetic_ops import (
    abs,
    ceil_int,
    compare,
    min,
    floor_int,
    mod_const,
    multiply_2d,
    multiply_integers,
    piecewise_linear,
    reciprocal,
    relu_add,
    square,
    thermometer_floor_div,
)

# Logic
from torchwright.ops.relu.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    cond_gate,
    equals_vector,
)

# Selection and lookup
from torchwright.ops.relu.map_select import (
    broadcast_select,
    dynamic_extract,
    in_range,
    map_to_table,
    select,
    switch,
    table_lookup_2d,
)

# I/O nodes
from torchwright.ops.inout_nodes import (
    create_embedding,
    create_input,
    create_literal_value,
    create_rope_config,
    create_unembedding,
)

# Scalar encoding
from torchwright.ops.relu.scalar_encoding import (
    digit_to_scaled_scalar,
    digits_to_number,
    number_to_digit_scalars,
    scalar_to_embedding,
)

# Embedding-space arithmetic
from torchwright.ops.relu.embedding_arithmetic import (
    sum_digit_seqs,
    sum_digits,
)

# Sequence
from torchwright.ops.relu.sequence_ops import (
    NumericSequence,
    check_is_digit,
    output_sequence,
    remove_leading_0s,
)

# FFN builder
from torchwright.ops.relu.linear_relu_linear import linear_relu_linear
