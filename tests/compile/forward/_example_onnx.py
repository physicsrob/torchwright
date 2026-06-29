"""Shared helper for the token-example ONNX correctness tests.

Each token example used to be compiled in-process (``forward_compile`` →
``HeadlessTransformer.compute``) and decoded by nearest embedding.  These
tests now export through :func:`compile_to_onnx` and decode with
:meth:`OnnxTokenModule.generate` (argmax over logits).  The decoded output
is token-identical: the embedding table is an equal-norm spherical code, so
``argmax(x · eᵢ)`` selects the same row as ``argmin ‖x − eᵢ‖`` — the
constant ``‖eᵢ‖²`` term drops out of the comparison.
"""

import os

import pytest

onnxruntime = pytest.importorskip("onnxruntime")

from torchwright.compiler.export import compile_to_onnx
from torchwright.compiler.onnx_load import load_onnx

D = 1024
D_HEAD = 16


def load_example(build_fn, out_dir, *, d=D, d_head=D_HEAD, name="example"):
    """Compile a token example to ONNX; return ``(model, artifact)``.

    ``build_fn()`` returns ``(output_node, embedding)`` — the RoPE-era
    ``create_network_parts`` contract every token example shares (position is
    a rotation inside attention now, so there is no ``pos_encoding`` to pass).
    ``out_dir`` is a directory the model + sidecars may be written to (a
    ``tmp_path_factory.mktemp(...)`` result works).
    """
    output_node, embedding = build_fn()
    onnx_path = os.path.join(str(out_dir), f"{name}.onnx")
    artifact = compile_to_onnx(
        output_node,
        embedding,
        onnx_path,
        d=d,
        d_head=d_head,
        verbose=False,
    )
    return load_onnx(onnx_path), artifact


def run(model, input_text, *, bos_token="<bos>", ref_token=None, max_new_tokens=24):
    """Argmax-decode one prompt; return the joined output string.

    ``input_text`` is everything after the marker prefix: the prompt is
    ``[bos_token] (+ [ref_token]) + list(input_text)``.  Recency examples (those
    whose graph builds a :func:`~torchwright.ops.recency_heads.
    recency_rank_from_tokens`) pass ``ref_token="<ref>"`` so the bucket-2
    readout has its second always-visible marked token at position 1; others
    leave it ``None``.
    """
    return "".join(
        model.generate(
            input_text,
            bos_token=bos_token,
            ref_token=ref_token,
            max_new_tokens=max_new_tokens,
        )
    )
