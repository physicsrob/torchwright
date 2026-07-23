"""The memorizing calculator: every fact correct, every formula validated.

``calculator_memorize`` stores one swish lane per possible expression, so
its correctness claim is finite and fully checkable: at ``max_digits=1``
all 300 facts are verified end to end (reference evaluation of the whole
graph — parse, fact lookup, emission).  Verification is teacher-forced:
one evaluation per expression over the prompt + expected-answer token
stream, asserting the argmax prediction at every answer position — which
is exactly greedy decoding whenever it passes.

The module's closed-form ``n_facts`` / ``n_params`` expressions (the
extrapolation story for the unbuildable ``n >= 3``) are validated against
the built graph's actual lane and weight counts, and the width-unlimited
layer floor is pinned constant across n — the two regimes the module
docstring states.
"""

from typing import cast

import pytest

from examples import calculator_memorize as cm
from torchwright.compiler.forward.cpsat_scheduler import critical_path_layers
from torchwright.compiler.lower import lower
from torchwright.compiler.utils import get_ancestor_nodes
from torchwright.debug.probe import reference_eval
from torchwright.graph.embedding import bos_token
from torchwright.graph.ffn import FFN


@pytest.fixture(scope="module")
def model_n1():
    return cm.create_network_parts(max_digits=1)


@pytest.fixture(scope="module")
def model_n2():
    return cm.create_network_parts(max_digits=2)


def _check_expr(out_node, embedding, expr: str, expected: str) -> None:
    """Teacher-forced check: the model's argmax predicts the expected next token.

    Checked at the newline and every answer position.
    """
    tokens = [bos_token, *list(expr), "\n", *list(expected), "<eos>"]
    cache = reference_eval(
        out_node, cast("dict", {"embedding_input": tokens}), len(tokens)
    )
    logits = cache[out_node]
    vocab = embedding.tokenizer.vocab
    start = 1 + len(expr)  # the newline's position predicts the first digit
    for k, want in enumerate([*list(expected), "<eos>"]):
        got = vocab[int(logits[start + k].argmax())]
        assert got == want, (expr, k, got, want)


def test_all_300_facts_at_n1(model_n1):
    out, embedding = model_n1
    for a in range(10):
        for b in range(10):
            for op, fn in (("+", a + b), ("-", a - b), ("*", a * b)):
                _check_expr(out, embedding, f"{a}{op}{b}", str(fn))


N2_CASES = [
    ("0+0", "0"),
    ("99+99", "198"),
    ("99*99", "9801"),
    ("0-99", "-99"),
    ("99-0", "99"),
    ("10-99", "-89"),
    ("12*34", "408"),
    ("90-9", "81"),
    ("5*9", "45"),  # narrow operands key through the zero-padded windows
    ("7+8", "15"),
    ("00+07", "7"),  # explicit padding stays valid
    ("40*25", "1000"),
    ("13-31", "-18"),
    ("99-98", "1"),
]


def test_sampled_facts_at_n2(model_n2):
    out, embedding = model_n2
    for expr, expected in N2_CASES:
        _check_expr(out, embedding, expr, expected)


def _fact_table_ffns(out_node, max_digits: int):
    lanes = 10 ** (2 * max_digits - 1)
    d_key = 34 * max_digits + 3  # both operand windows + the operator one-hot
    return [
        n
        for n in get_ancestor_nodes({out_node})
        if isinstance(n, FFN)
        and n.gate_proj.shape[0] == lanes
        and n.gate_proj.shape[1] == d_key
    ]


@pytest.mark.parametrize("n", [1, 2])
def test_fact_and_param_formulas_match_built_graph(n, model_n1, model_n2):
    out, _embedding = {1: model_n1, 2: model_n2}[n]
    ffns = _fact_table_ffns(out, n)
    assert len(ffns) == 30
    assert sum(f.gate_proj.shape[0] for f in ffns) == cm.n_facts(n)
    counted = sum(
        f.gate_proj.numel()
        + f.gate_bias.numel()
        + f.out_proj.numel()
        + f.out_bias.numel()
        for f in ffns
    )
    assert counted == cm.n_params(n)


def test_layer_floor_is_constant_in_n(model_n1, model_n2):
    floors = []
    for out, _embedding in (model_n1, model_n2):
        lowered = lower(
            out, collapse_univariate=True, collapse_pl=True, collapse_lane_cap=2048
        )
        floors.append(critical_path_layers(lowered.output_node))
    assert floors[0] == floors[1], floors


def test_n3_is_stated_as_unbuildable():
    with pytest.raises(ValueError, match="3,000,000 fact lanes"):
        cm.create_network_parts(max_digits=3)


def test_compiled_layers_law_matches_measured_compiles():
    """Pin the capacity law to the three witnessed optimize=0 compiles.

    Measured 2026-07-20 via scripts.measure_calculator_compiled_layers
    on Modal. The default d_hidden is the family's canonical D_HIDDEN.
    """
    assert cm.compiled_layers(1, d_hidden=8192) == 15
    assert cm.compiled_layers(2, d_hidden=8192) == 18
    assert cm.compiled_layers(2, d_hidden=16384) == 16
    assert cm.compiled_layers(2) == cm.compiled_layers(2, d_hidden=cm.D_HIDDEN)
