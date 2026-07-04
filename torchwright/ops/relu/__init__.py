"""The ReLU-machine op library — moved from ``torchwright/ops/``, frozen.

These modules are the machine-specific half of today's op library: every
op here spends MLP hidden lanes through ReLU activations.  The purely
linear ops, the attention hardware, and the shared pure-math helpers
live one level up (``torchwright/ops/linear.py``, ``attention_ops.py``,
``_math.py``) — they carry no machine choice and both libraries build on
them.

Frozen means: no new features land here.  ``tests/ops/*`` are the
regression baseline; the swish counterpart library is
``torchwright/ops/swiglu/``.  Retirement policy lives in
``docs/swiglu_step2_plan.md``.
"""
