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

    ``build_fn()`` returns ``(output_node, pos_encoding, embedding)`` — the
    ``create_network_parts`` contract every token example shares.
    ``out_dir`` is a directory the model + sidecars may be written to (a
    ``tmp_path_factory.mktemp(...)`` result works).
    """
    output_node, pos_encoding, embedding = build_fn()
    onnx_path = os.path.join(str(out_dir), f"{name}.onnx")
    artifact = compile_to_onnx(
        output_node,
        pos_encoding,
        embedding,
        onnx_path,
        d=d,
        d_head=d_head,
        verbose=False,
    )
    return load_onnx(onnx_path), artifact


def run(model, input_text, *, bos_token="<bos", max_new_tokens=24):
    """Argmax-decode one prompt; return the joined output string.

    ``input_text`` is everything after the BOS token: the example's old
    ``["<bos>"] + list(input) [+ ["\\n"]]`` prompt becomes ``bos_token``
    plus this string.  ``bos_token`` is ``"<bos"`` for adder/calculator (a
    vocab quirk — their BOS has no closing ``>``) and ``"<bos>"`` for the
    rest.
    """
    return "".join(
        model.generate(input_text, bos_token=bos_token, max_new_tokens=max_new_tokens)
    )
