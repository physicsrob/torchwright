"""The ReLU-machine op library — moved from ``torchwright/ops/``, frozen.

These modules are today's op library, byte-identical to their pre-split
form; their internal cross-imports still use the old
``torchwright.ops.<mod>`` paths, resolved through the ``sys.modules``
aliases that ``torchwright/ops/__init__.py`` registers in dependency
order before anything here loads.  Keep this ``__init__`` import-free:
an import here would run *before* those aliases exist and break the
bootstrap.

Frozen means: no new features land here.  ``tests/ops/*`` (still on the
old import paths) are the regression baseline; the swish counterpart
library is ``torchwright/ops/swiglu/``.  Retirement policy lives in
``docs/swiglu_step2_plan.md``.
"""
