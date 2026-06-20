"""Advanced calculator: the same graph as ``calculator_simple`` but wired to the
depth-optimized arithmetic (``onehot_arithmetic_fast``).

Identical token plumbing and parse → compute → emit body — it just passes the
fast arithmetic module (carry-lookahead add/subtract, carry-save/Wallace
multiply) to :func:`examples.calculator_simple.build_calculator`.  The
algorithms grow logarithmically in depth instead of linearly, at a width cost;
see ``scripts/arithmetic_scaling.py`` for the simple-vs-advanced comparison.

This variant exists for measurement and contrast; the blog explains the simple
version line by line and only cites the advanced curves.  ``D_MODEL`` and
``create_network_parts`` mirror ``calculator_simple`` so ``examples.compile``
and the token-example tests treat the two interchangeably.
"""

from typing import Tuple

from torchwright.graph import Node, Embedding, PosEncoding
from torchwright.ops import onehot_arithmetic_fast

import examples.calculator_simple as calculator_simple

D_MODEL = calculator_simple.D_MODEL
CALC_VOCAB = calculator_simple.CALC_VOCAB


def create_network_parts(
    max_digits: int = 3,
) -> Tuple[Node, PosEncoding, Embedding]:
    """The advanced calculator: ``build_calculator`` over ``onehot_arithmetic_fast``."""
    return calculator_simple.build_calculator(onehot_arithmetic_fast, max_digits)
