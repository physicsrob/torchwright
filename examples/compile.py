"""Compile an example to ONNX.  Usage: uv run python -m examples.compile <name>"""

import importlib
import sys

from torchwright.compiler.export import compile_to_onnx


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <example_name>")
        sys.exit(1)

    name = sys.argv[1]
    module = importlib.import_module(f"examples.{name}")

    if not hasattr(module, "create_network_parts"):
        print(f"Error: examples.{name} has no create_network_parts()")
        sys.exit(1)

    output_node, embedding = module.create_network_parts()
    compile_to_onnx(output_node, embedding, f"{name}.onnx", d=module.D_MODEL)


if __name__ == "__main__":
    main()
