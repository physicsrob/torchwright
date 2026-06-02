"""Regression: a streaming compile must SKIP the post-loop in-place weight trims.

``compile_to_onnx`` passes an ``on_layer_compiled`` callback that extracts each
layer's weights and NULLs the tensors per layer (to bound peak memory to one
dense layer's worth regardless of depth — see export._make_stream_layer_weights_cb).
The post-loop ``trim_unused_heads`` / ``trim_unused_slots`` then slice those
tensors; if they run after the streaming null they hit ``NoneType`` and crash
(``'NoneType' object is not subscriptable`` / ``'bool' object has no attribute
'any'``).

The onnxruntime-based ``compile_to_onnx`` tests are skipped when onnxruntime is
absent, so this no-onnxruntime test pins the trim/null ordering directly: with a
streaming callback present, ``forward_compile`` must complete (trims skipped).
The streamed sparse export already drops the zero heads/slots, so the in-memory
trim is both impossible and redundant in that mode.
"""

import torch

from torchwright.compiler.forward.compile import forward_compile
from torchwright.graph import PosEncoding
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear_relu_linear import linear_relu_linear


def _null_layer_weights(layer) -> None:
    """Mimic the ONNX streaming callback: free each layer's dense weights."""
    layer.attn.attn.query_matrix = None
    layer.attn.attn.key_matrix = None
    layer.attn.attn.value_matrix = None
    layer.attn.attn.output_matrix = None
    layer.mlp.linear1.output_matrix = None
    layer.mlp.linear2.output_matrix = None


def test_streaming_callback_skips_inplace_trim():
    inp = create_input("x", 16)
    pos = PosEncoding(9)  # trig_width 8 == d_head below
    torch.manual_seed(0)
    out = linear_relu_linear(
        inp,
        torch.randn(24, 16),
        torch.randn(24),
        torch.randn(24, 16),
        torch.randn(16),
        name="chain",
    )

    fired: list[int] = []

    def on_layer_compiled(i, layer):
        _null_layer_weights(layer)
        fired.append(i)

    # trim_heads=True + a streaming callback that nulls weights: the post-loop
    # trims must be skipped. Before the fix this raised inside trim_unused_heads
    # / trim_unused_slots; it must now complete.
    net = forward_compile(
        d=64,
        d_head=8,
        output_node=out,
        pos_encoding=pos,
        on_layer_compiled=on_layer_compiled,
        trim_heads=True,
        verbose=False,
        device="cpu",
    )

    assert fired, "streaming callback never fired"
    assert net is not None
