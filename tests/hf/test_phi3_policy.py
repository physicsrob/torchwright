"""Repository policy gates for the strong stock-Phi-3 default."""

from __future__ import annotations

import inspect
from pathlib import Path

from torchwright.compiler.hf import compile_hf_bundle, compile_to_hf
from torchwright.compiler.token_model import CompileProfile


def test_public_hf_defaults_are_phi3():
    assert inspect.signature(compile_to_hf).parameters["architecture"].default == "phi3"
    assert (
        inspect.signature(compile_hf_bundle).parameters["architecture"].default
        == "phi3"
    )
    assert CompileProfile.PHI3.machine == "swish"
    assert CompileProfile.PHI3.bias is False
    assert CompileProfile.PHI3.rms_norm is True


def test_examples_do_not_import_relu_dialect():
    examples = Path(__file__).resolve().parents[2] / "examples"
    offenders = []
    for path in examples.glob("*.py"):
        if "torchwright.ops.relu" in path.read_text():
            offenders.append(path.name)
    assert not offenders, (
        "examples must compile under the default Phi-3 profile; ReLU imports "
        f"require a non-example custom fixture instead: {offenders}"
    )


def test_named_example_widths_are_rmsnorm_compatible():
    from torchwright.compiler.forward.compile import rms_norm_width_supported
    from examples import (
        adder,
        binary_increment,
        caesar_cipher,
        calculator_scratchpad,
        calculator_simple,
        fibonacci,
        sort_digits_v1,
    )

    modules = [
        adder,
        binary_increment,
        caesar_cipher,
        calculator_scratchpad,
        calculator_simple,
        fibonacci,
        sort_digits_v1,
    ]
    unsupported = {
        module.__name__: module.D_MODEL
        for module in modules
        if not rms_norm_width_supported(module.D_MODEL)
    }
    assert not unsupported


def test_onnx_accepts_the_same_phi3_profile(tmp_path):
    from torchwright.compiler.export import compile_to_onnx
    from torchwright.graph.rope import rotary_offset_head
    from torchwright.ops.inout_nodes import create_onehot_embedding

    embedding = create_onehot_embedding(["<bos>", "<eos>", "a"])
    output = rotary_offset_head(embedding, delta_pos=-1, d_qk=16)
    artifact = compile_to_onnx(
        output,
        embedding,
        str(tmp_path / "profile.onnx"),
        d=256,
        d_head=16,
        profile=CompileProfile.PHI3,
    )
    assert artifact.activation == "swish"
    assert artifact.bias is False
