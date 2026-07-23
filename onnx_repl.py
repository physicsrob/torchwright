"""REPL for an ONNX-compiled torchwright model.

Usage:
    python compile_interact.py model.onnx
"""

import argparse

from torchwright.compiler.repl import run_repl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive REPL for a compiled ONNX model"
    )
    parser.add_argument("onnx_path", help="Path to the .onnx model file")
    parser.add_argument(
        "--max-tokens", type=int, default=20, help="Max tokens to generate"
    )
    parser.add_argument(
        "-p", "--prompt", type=str, default=None, help="Single prompt (skip REPL)"
    )
    args = parser.parse_args()
    if args.prompt is not None:
        from torchwright.compiler.repl import run_once

        run_once(args.onnx_path, args.prompt, max_new_tokens=args.max_tokens)
    else:
        run_repl(args.onnx_path, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
