"""End-to-end CI smoke: compile an example, generate, check the output.

Compiles ``examples/binary_increment.py`` to a Hugging Face bundle and
verifies that a ``transformers`` pipeline loaded from it increments a
binary string.  This exercises the full stack — graph construction,
scheduling, weight writing, bundle export, tokenizer, generation —
in about a minute on a CPU runner, standing in for the full suite on
every push (the full suite runs on Modal locally and on the weekly
full-tests workflow).
"""

import sys
import tempfile
from pathlib import Path


def main() -> int:
    from examples.binary_increment import D_HEAD, D_MODEL, create_network_parts
    from torchwright import compile_hf_bundle

    output_node, embedding = create_network_parts()
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "binary_increment_hf_bundle"
        compile_hf_bundle(output_node, embedding, str(bundle), d=D_MODEL, d_head=D_HEAD)

        from transformers import pipeline

        generate = pipeline("text-generation", model=str(bundle))
        out = generate("1011\n", return_full_text=False)[0]["generated_text"]

    print(f"generated: {out!r}")
    if out.strip() != "1100":
        print("FAIL: expected '1100'", file=sys.stderr)
        return 1
    print("smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
