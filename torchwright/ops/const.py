"""Constants for the ops graph-building layer.

Boolean convention: throughout ops, boolean-valued nodes use
1.0 for true and -1.0 for false, with 0.0 as the decision threshold.
Functions like ``compare``, ``equals_vector``, ``bool_not``, etc.
all follow this convention.
"""

step_sharpness = 10.0
embedding_step_sharpness = 1.0  # For embedding-space ops (map_to_table, equals_vector).
# Lower than step_sharpness because the margin (1/speed) must
# absorb dot-product errors from approximate embeddings.
# Embedding norms are ~40 (self-dot ~1600), so even tiny
# Euclidean errors become large dot-product errors.

# Hinge sharpening for the swish machine (docs/swiglu_step2_plan.md, settled
# decision 5).  Only ever used self-normalizing — folded into gate rows with
# the matching /scale folded into out_proj (``Swish(scale·z)/scale``), so no
# value path carries it; never a per-call knob.  Recorded cost: hidden slots
# saturate at ~1e5-magnitude values, foreclosing fp16 export (the artifact is
# fp32 everywhere today).  The numeric claims tied to this value are pinned
# by tests/docs/test_swish_constants.py.
scale = 100.0

# Swish's global-minimum magnitude: |min_z z·sigmoid(z)| = 0.2784645... at
# z = -1.2784645...  This is the worst gap between the sharpened hinge
# ``Swish(scale·z)/scale`` and ``ReLU(z)`` — ``swish_dip/scale`` in value
# units — so it sizes every dip slack the swish op library adds to its
# value-range asserts and semantic bounds (compare's bend overshoot,
# equals_vector's low-side dip, in_range's per-slot slack).  Pinned against
# the doc's 0.2785 by tests/docs/test_swish_constants.py.
swish_dip = 0.2784645
