"""The schematic: the hash-bound record shipped with every compiled artifact.

Format home and consumer surface for ``torchwright_schematic.json``.
Submodules: ``format`` (constants, hashing, the run encoding),
``validate`` (the checks both builder and readers run), ``support``
(the nonzero-coordinate npz sidecar).  Importing this package never
imports torch; numpy loads only when a support archive is opened.
"""
