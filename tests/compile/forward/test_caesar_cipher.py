"""Caesar cipher: export to ONNX and verify correctness.

Tests various shift amounts and letter combinations.  Converted from
``forward_compile``; see ``_example_onnx``.
"""

import pytest

from examples.caesar_cipher import D_HEAD, create_network_parts

from ._example_onnx import load_example, run


@pytest.fixture(scope="module")
def caesar(tmp_path_factory):
    return load_example(
        create_network_parts,
        tmp_path_factory.mktemp("caesar"),
        d_head=D_HEAD,
        name="caesar",
    )


def _caesar(text: str, shift: int) -> str:
    """Reference Caesar cipher implementation."""
    return "".join(chr((ord(c) - ord("a") + shift) % 26 + ord("a")) for c in text)


def test_caesar_cipher(caesar):
    model, artifact = caesar
    # RoPE local recency is a single rotary head (intrinsic distance-decay lobe,
    # shallower than the deleted octant ramp; docs/rope_port_plan.md Phase 6) —
    # comfortable upper bound with margin.
    assert artifact.n_layers <= 48, f"Too many layers: {artifact.n_layers}"

    test_cases = [
        # shift=0: identity
        ("0", "hello", "hello"),
        ("0", "abcde", "abcde"),
        # shift=1: each letter +1
        ("1", "abcde", "bcdef"),
        ("1", "hello", "ifmmp"),
        # shift=3: classic Caesar
        ("3", "hello", "khoor"),
        ("3", "abcde", "defgh"),
        # Wraparound
        ("1", "xyzab", "yzabc"),
        ("3", "xyzab", "abcde"),
        # Larger shifts
        ("9", "abcde", "jklmn"),
        ("5", "vwxyz", "abcde"),
    ]
    for shift, plaintext, expected in test_cases:
        # Verify reference matches expected
        assert (
            _caesar(plaintext, int(shift)) == expected
        ), f"Reference mismatch for shift={shift}, text={plaintext}"
        # Prompt: bos, shift token, plaintext chars, newline trigger.
        result = run(model, shift + plaintext + "\n", bos_token="<bos>")
        assert result == expected, (
            f"For shift={shift}, {plaintext!r}: "
            f"expected {expected!r} but got {result!r}"
        )
