"""Reproduce the wide-operand truncation in the calculator parse.

The published n=12 calculators (2026-07-26 verification) multiplied
``A x B`` with each operand truncated to its 8 most significant digits:
every operand digit at region index >= 8 read the parse windows' ``"0"``
default.  Root cause: collapse_pl's chained hinge bands (fixed the same
day — see ``_excusable_bands`` in ``torchwright/compiler/pl_function.py``
and the regression tests in ``tests/compile/test_collapse_pl.py``).
This script walks the evidence chain and stays as the diagnostic for
any recurrence:

1. reference-eval the *shared parse* at ``max_digits=12`` on a
   full-width prompt — is the graph math itself clean?
2. the marker-distance count and member-key one-hots per position
   (rebuilt through the same public ops ``IndexedRegion`` uses);
3. the same probe compiled small, oracle-diffed per node.

Run locally (seconds, 27 positions) or on Modal::

    uv run python -m scripts.investigate_region_truncation
    make modal-run MODULE=scripts.investigate_region_truncation CPU_ONLY=1
"""

import torch

from examples._calculator_common import (
    CALC_VOCAB,
    D_HEAD,
    MAX_POSITIONS,
    parse_expression,
)
from torchwright.debug.probe import reference_eval
from torchwright.graph import Node, RopeConfig
from torchwright.graph.embedding import Embedding, bos_token
from torchwright.ops.attention_ops import get_prev_value
from torchwright.ops.inout_nodes import create_onehot_embedding, create_rope_config
from torchwright.ops.linear import add_const, bool_to_01, concat
from torchwright.ops.swiglu.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    equals_vector,
)
from torchwright.ops.swiglu.map_select import cond_gate, in_range
from torchwright.ops.swiglu.marker_count import count_since_marker
from torchwright.ops.swiglu.sequence_ops import check_is_digit

MAX_DIGITS = 12
A = "999999999999"
B = "123456789012"


def _decode(embedding: Embedding, vec: torch.Tensor) -> str:
    sims = embedding.table @ vec
    token = list(embedding.tokenizer.vocab)[int(sims.argmax())]
    return f"{token!r} (top={sims.max():.3f})"


def _token_ids(embedding: Embedding, tokens: list[str]) -> torch.Tensor:
    return torch.tensor(
        [embedding.tokenizer.get_token_id(t) for t in tokens], dtype=torch.long
    ).reshape(-1, 1)


def _count_probe(rope: RopeConfig, embedding: Embedding) -> Node:
    """The first operand's count / latched length / member keys.

    Rebuilt from the same public ops ``IndexedRegion`` composes (minus
    the marker-row sentinel, irrelevant to the member-key diagnostic),
    so the probe tracks the parse's real construction without reaching
    into the region's private nodes.
    """
    embed = embedding.get_embedding
    is_operator = bool_any_true(
        [equals_vector(embedding, embed(op)) for op in ("+", "-", "*")]
    )
    saw_newline = equals_vector(embedding, embed("\n"))
    seen_newline = get_prev_value(rope, saw_newline, saw_newline)
    is_input_operator = bool_all_true([is_operator, bool_not(seen_newline)])
    saw_bos = equals_vector(embedding, embed(bos_token))
    seen_operator = get_prev_value(rope, is_input_operator, is_input_operator)
    in_first = bool_all_true([check_is_digit(embedding), bool_not(seen_operator)])

    seen_marker = get_prev_value(rope, saw_bos, saw_bos)
    count = count_since_marker(
        rope, seen_marker, bool_to_01(saw_bos), max_gap=MAX_DIGITS + 1
    )
    length = add_const(get_prev_value(rope, count, is_input_operator), -1.0)
    own_index = add_const(count, -1.0)
    own_onehot = bool_to_01(in_range(own_index, add_const(own_index, 1.0), MAX_DIGITS))
    member_keys = cond_gate(in_first, own_onehot)
    return concat([count, length, member_keys])


def _print_key_rows(
    tokens: list[str], values: torch.Tensor, oracle: torch.Tensor | None = None
) -> None:
    for p in range(len(tokens)):
        keys = values[p, 2:]
        cmp = "" if oracle is None else f"/{float(oracle[p, 0]):7.3f}"
        cmp_len = "" if oracle is None else f"/{float(oracle[p, 1]):7.3f}"
        print(
            f"  {p:3d} {tokens[p]!r:6s} {float(values[p, 0]):7.3f}{cmp}  "
            f"{float(values[p, 1]):8.3f}{cmp_len}   "
            f"lane {int(keys.argmax()):2d}: {keys.max():.3f} | sum {keys.sum():.3f}"
        )


def stage1_parse_windows(embedding: Embedding, rope: RopeConfig) -> None:
    first, second, _p, _m, _t, _nl = parse_expression(rope, embedding, MAX_DIGITS)
    out = concat(first + second)
    tokens = [bos_token, *A, "*", *B, "\n"]
    vals = reference_eval(
        out, {"embedding_input": _token_ids(embedding, tokens)}, len(tokens)
    )
    print(f"prompt: {A} * {B}  ({len(tokens)} positions)")
    print("\n== stage 1: parsed windows at the newline (latched values) ==")
    for label, window, expect in (("first", first, A), ("second", second, B)):
        for i, node in enumerate(window):
            got = _decode(embedding, vals[node][-1])
            print(f"  {label}[{i:2d}] -> {got}   expected {expect[i]!r}")


def stage2_reference_internals(embedding: Embedding, rope: RopeConfig) -> None:
    print("\n== stage 2: count / length / member keys (reference eval) ==")
    probe = _count_probe(rope, embedding)
    tokens = [bos_token, *A, "*", *B, "\n"]
    pv = reference_eval(
        probe, {"embedding_input": _token_ids(embedding, tokens)}, len(tokens)
    )[probe]
    print("  pos tok   count   latched_len  key_lanes (argmax:val | sum)")
    _print_key_rows(tokens, pv)


def stage3_compiled_vs_oracle(embedding: Embedding, rope: RopeConfig) -> None:
    print("\n== stage 3: compiled probe vs oracle ==")
    from torchwright.compiler.export import compile_headless
    from torchwright.debug.probe import probe_compiled

    probe = _count_probe(rope, embedding)
    tokens = [bos_token, *A, "*", *B, "\n"]
    ids = _token_ids(embedding, tokens)
    iv = {"embedding_input": ids}
    compiled = compile_headless(probe, d=2048, d_head=D_HEAD)
    print(f"  compiled: {compiled.n_layers} layers")
    report = probe_compiled(compiled, probe, iv, len(tokens), atol=0.25)
    print(report.format_short())
    compiled(compiled.build_prefill(iv, len(tokens)), debug=True)
    cv = compiled.debug_value(probe)
    assert cv is not None, "probe node has no residual assignment"
    pv = reference_eval(probe, iv, len(tokens))[probe]
    print("  pos tok   count(cmp/orc)     len(cmp/orc)    keys argmax:val | sum")
    _print_key_rows(tokens, cv, oracle=pv)


def main() -> None:
    embedding = create_onehot_embedding(CALC_VOCAB)
    rope = create_rope_config(d_head=D_HEAD, max_positions=MAX_POSITIONS)
    stage1_parse_windows(embedding, rope)
    stage2_reference_internals(embedding, rope)
    stage3_compiled_vs_oracle(embedding, rope)


if __name__ == "__main__":
    main()
