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


def load_example(
    build_fn, out_dir, *, d=D, d_head=D_HEAD, name="example", rms_norm=None
):
    """Compile a token example to ONNX; return ``(model, artifact)``.

    ``build_fn()`` returns ``(output_node, embedding)`` — the RoPE-era
    ``create_network_parts`` contract every token example shares (position is
    a rotation inside attention now, so there is no ``pos_encoding`` to pass).
    ``out_dir`` is a directory the model + sidecars may be written to (a
    ``tmp_path_factory.mktemp(...)`` result works).

    ``rms_norm`` threads to :func:`compile_to_onnx` (default ``None`` = on, the
    production default).  Pass ``rms_norm=False`` for an example whose ``d`` is
    outside the norm's supported-width contract (any multiple of 1024 up to
    16384, or any power of two — ``docs/rms_norm_dmodel.md``); such examples
    use an odd width for residual reasons, not to exercise the norm.
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
        rms_norm=rms_norm,
    )
    return load_onnx(onnx_path), artifact


def run(model, input_text, *, bos_token="<bos>", max_new_tokens=24):
    """Argmax-decode one prompt; return the joined output string.

    ``input_text`` is everything after the BOS marker: the prompt is
    ``[bos_token] + list(input_text)``.  (RoPE recency is the intrinsic local
    rotary lobe now — there is no injected ``<ref>`` token; ``docs/
    rope_port_plan.md`` Phase 6.)
    """
    return "".join(
        model.generate(
            input_text,
            bos_token=bos_token,
            max_new_tokens=max_new_tokens,
        )
    )
