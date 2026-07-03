"""Scalar↔embedding conversion on the swish machine.

Only ``scalar_to_embedding`` lands here for now — the rest of the digit
pipeline (``digit_to_scaled_scalar``, ``digits_to_number``,
``number_to_digit_scalars``) are compositions that migrate with their
ingredients (docs/swiglu_step2_plan.md, Phase B item 11).
"""

import torch

from torchwright.graph import Node
from torchwright.graph.embedding import Embedding
from torchwright.ops.const import scale, step_sharpness
from torchwright.ops.swiglu.swiglu_ffn import swiglu_ffn


def scalar_to_embedding(inp: Node, embedding: Embedding) -> Node:
    """Convert a scalar digit (0.0-9.0) back to its embedding vector.

    A :func:`piecewise_linear` special case ported hinge-for-hinge: nine
    unit steps at half-integer thresholds, vector-valued (the embedding
    deltas fold into out_proj, all lanes degenerate)::

        step_k = hinge(z_k) − hinge(z_k − 1),  z_k = S·(x − (k+0.5))
        result = embed(0) + Σ_k step_k · (embed(k+1) − embed(k))

    The single-FFN form stays safe — no floor_int-style two-stage split:
    there are 9 boundaries and the sharpened arguments top out near
    ``scale·S·9 ≈ 1e4``, the small-and-safe end of the construction.  An
    integer digit puts every hinge argument on an exact integer with
    ``|z| ≥ 5`` — every sharpened argument ≥ 500, saturated or
    underflowed exactly — so the indicators are exact 0/1 and the only
    error is out_proj rounding (~ulps of the embedding components).
    Input-noise headroom unchanged: a digit scalar off by up to ±0.4
    reconstructs the same embedding; mid-ramp inputs blend the two
    adjacent embeddings linearly, as today.  The piecewise_linear
    spacing audit closes in closed form here: the pair's hinges sit
    ``1/S`` apart with fillet radius ``17/(scale·S)``, non-overlapping
    exactly when ``scale > 34`` — the module scale clears it 3x.

    Args:
        inp: 1D scalar node with value in [0.0, 9.0].
        embedding: Embedding table (must contain "0"-"9").

    Returns:
        Node of width ``embedding.d_embed`` containing the reconstructed
        embedding vector.
    """
    assert len(inp) == 1, "Input must be a 1D scalar node"
    d_embed = embedding.d_embed
    n_thresholds = 9
    n_lanes = 2 * n_thresholds
    S = step_sharpness

    gate_proj = torch.zeros(n_lanes, 1)
    gate_bias = torch.zeros(n_lanes)
    output_proj = torch.zeros(n_lanes, d_embed)

    for k in range(n_thresholds):
        threshold = k + 0.5
        row = 2 * k

        # Unit-step hinge pair: bends at z_k = 0 and z_k = 1.
        gate_proj[row, 0] = scale * S
        gate_proj[row + 1, 0] = scale * S
        gate_bias[row] = -scale * S * threshold
        gate_bias[row + 1] = -scale * (S * threshold + 1.0)

        # The step contributes the embedding delta; /scale folds in.
        delta = embedding.get_embedding(str(k + 1)) - embedding.get_embedding(str(k))
        output_proj[row, :] = delta / scale
        output_proj[row + 1, :] = -delta / scale

    # Start from embed("0"); deltas telescope up to embed(d).
    output_bias = embedding.get_embedding(str(0)).clone()

    return swiglu_ffn(
        inp,
        gate_proj,
        gate_bias,
        output_proj,
        output_bias,
        name="scalar_to_embedding",
    )
