"""Interactive REPL that runs inference on a token-I/O ONNX export.

Thin front-end over :func:`torchwright.compiler.onnx_load.load_onnx` /
:class:`torchwright.compiler.onnx_load.OnnxTokenModule` (which carry a
torch dependency).  Requires: onnxruntime, torch, numpy.
"""

import sys

from torchwright.compiler.onnx_load import OnnxTokenModule, load_onnx


def _load_token_model(onnx_path: str) -> OnnxTokenModule:
    model = load_onnx(onnx_path)
    if not isinstance(model, OnnxTokenModule):
        raise ValueError(
            f"{onnx_path}: the repl needs a token-I/O export "
            f"(compile_to_onnx); this is a {type(model).__name__} artifact"
        )
    return model


def run_once(
    onnx_path: str,
    prompt: str,
    max_new_tokens: int = 20,
) -> None:
    """Run a single prompt and print the result.

    Args:
        onnx_path: Path to the .onnx model file.
        prompt: The input string (e.g. "12+34").
        max_new_tokens: Maximum tokens to generate.
    """
    model = _load_token_model(onnx_path)
    for token in model.generate(prompt + "\n", max_new_tokens):
        sys.stdout.write(token)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_repl(
    onnx_path: str,
    max_new_tokens: int = 20,
) -> None:
    """Load an ONNX model and run an interactive REPL.

    Expects a meta file at <onnx_path_without_ext>.meta.json (token format).

    Args:
        onnx_path: Path to the .onnx model file.
        max_new_tokens: Maximum tokens to generate per query.
    """
    model = _load_token_model(onnx_path)

    print(f"Loaded {onnx_path} ({len(model.vocab)} tokens). Type 'q' to quit.")
    while True:
        text = input("> ")
        if text.lower() == "q":
            print("Bye")
            break
        for token in model.generate(text + "\n", max_new_tokens):
            sys.stdout.write(token)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
