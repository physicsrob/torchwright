"""The ops graph-building layer.

Machine-selection boundary: ``torchwright.ops.relu`` is the frozen
ReLU-machine op library; ``torchwright.ops.swiglu`` is the swish-machine
library.  The import path *is* the machine choice — no mode flags, no
re-exports at this level.  The compiler's uniformity check (all FFNs in
a graph share one activation) is the backstop against accidental mixing.

Machine-neutral code lives at this level, and both libraries build on
it:

- ``const`` — module constants (sharpness, ``scale``, lane budget).
- ``inout_nodes`` — graph I/O construction.
- ``linear`` — purely linear ops (no MLP sublayer, so no activation
  choice).
- ``attention_ops`` — attention hardware.
- ``_math`` — private pure-math helpers shared by the two libraries'
  twins of the same op.
"""
