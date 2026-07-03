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

from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn
