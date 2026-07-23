"""Tests for numerical robustness when chaining map_to_table, equals_vector, and select.

These tests capture the core blocker for calculator multiplication: when the output
of one operation (map_to_table, select) feeds into another operation that uses
step_sharpness-scaled dot products, small Euclidean errors get amplified into
incorrect results.

The tests are written to FAIL against the current implementation and PASS once the
numerical robustness issues are fixed.
"""

import pytest
import torch

from torchwright.graph import Embedding
from torchwright.ops.inout_nodes import create_literal_value
from torchwright.ops.linear import concat
from torchwright.ops.relu.logic_ops import equals_vector
from torchwright.ops.relu.map_select import map_to_table, select

VOCAB = [str(i) for i in range(10)] + ["+", "-", "*", "=", "<eos>", "<bos>"]


def _make_digit_embedding():
    return Embedding(vocab=VOCAB)


def _digit_identity_table(embedding):
    """Lookup table that maps each digit embedding to itself."""
    table = {}
    for i in range(10):
        table[embedding.get_embedding(str(i))] = embedding.get_embedding(str(i))
    return table


# ---------------------------------------------------------------------------
# map_to_table output fidelity
# ---------------------------------------------------------------------------


def test_map_to_table_output_fidelity():
    """map_to_table output should be close to the true embedding."""
    embedding = _make_digit_embedding()
    table = _digit_identity_table(embedding)

    for digit in range(10):
        inp = create_literal_value(embedding.get_embedding(str(digit)))
        looked_up = map_to_table(inp, table, default=embedding.get_embedding("0"))
        result = looked_up.compute(n_pos=1, input_values={}).squeeze()
        true_emb = embedding.get_embedding(str(digit))
        dist = (result - true_emb).norm().item()
        assert dist < 0.1, (
            f"digit {digit}: map_to_table output dist={dist:.4f} from true embedding"
        )


def test_chained_map_to_table_output_fidelity():
    """Chaining two map_to_table lookups should still produce a recognizable embedding."""
    embedding = _make_digit_embedding()
    table = _digit_identity_table(embedding)

    for digit in range(10):
        inp = create_literal_value(embedding.get_embedding(str(digit)))
        first = map_to_table(inp, table, default=embedding.get_embedding("0"))
        second = map_to_table(first, table, default=embedding.get_embedding("0"))
        result = second.compute(n_pos=1, input_values={}).squeeze()
        true_emb = embedding.get_embedding(str(digit))
        dist = (result - true_emb).norm().item()
        assert dist < 1.0, f"digit {digit}: double map_to_table dist={dist:.4f}"


# ---------------------------------------------------------------------------
# equals_vector after map_to_table
# ---------------------------------------------------------------------------


def test_equals_vector_after_map_to_table():
    """equals_vector should recognize an embedding that went through map_to_table."""
    embedding = _make_digit_embedding()
    table = _digit_identity_table(embedding)

    for digit in range(10):
        inp = create_literal_value(embedding.get_embedding(str(digit)))
        looked_up = map_to_table(inp, table, default=embedding.get_embedding("0"))
        result = equals_vector(looked_up, embedding.get_embedding(str(digit)))
        output = result.compute(n_pos=1, input_values={}).item()
        assert output == pytest.approx(1.0, abs=0.1), (
            f"digit {digit}: equals_vector={output:.4f}, expected ~1.0"
        )


def test_equals_vector_after_map_to_table_rejects_wrong_digit():
    """equals_vector should reject wrong digits even after map_to_table."""
    embedding = _make_digit_embedding()
    table = _digit_identity_table(embedding)

    inp = create_literal_value(embedding.get_embedding("5"))
    looked_up = map_to_table(inp, table, default=embedding.get_embedding("0"))

    for wrong_digit in [0, 1, 2, 3, 4, 6, 7, 8, 9]:
        result = equals_vector(looked_up, embedding.get_embedding(str(wrong_digit)))
        output = result.compute(n_pos=1, input_values={}).item()
        assert output == pytest.approx(-1.0, abs=0.1), (
            f"wrong digit {wrong_digit}: equals_vector={output:.4f}, expected ~-1.0"
        )


def test_equals_vector_after_double_map_to_table():
    """equals_vector should work after two chained map_to_table lookups."""
    embedding = _make_digit_embedding()
    table = _digit_identity_table(embedding)

    for digit in range(10):
        inp = create_literal_value(embedding.get_embedding(str(digit)))
        first = map_to_table(inp, table, default=embedding.get_embedding("0"))
        second = map_to_table(first, table, default=embedding.get_embedding("0"))
        result = equals_vector(second, embedding.get_embedding(str(digit)))
        output = result.compute(n_pos=1, input_values={}).item()
        assert output == pytest.approx(1.0, abs=0.1), (
            f"digit {digit}: equals_vector after double lookup={output:.4f}"
        )


# ---------------------------------------------------------------------------
# equals_vector output bounds
# ---------------------------------------------------------------------------


def test_equals_vector_output_bounds_exact_input():
    """equals_vector output should be in [-1, 1] for exact embeddings."""
    embedding = _make_digit_embedding()

    for i in range(10):
        for j in range(10):
            inp = create_literal_value(embedding.get_embedding(str(i)))
            result = equals_vector(inp, embedding.get_embedding(str(j)))
            output = result.compute(n_pos=1, input_values={}).item()
            assert -1.0 - 1e-3 <= output <= 1.0 + 1e-3, (
                f"equals_vector({i}, {j}) = {output:.4f}, outside [-1, 1]"
            )


# ---------------------------------------------------------------------------
# equals_vector after select
# ---------------------------------------------------------------------------


def test_equals_vector_after_select():
    """equals_vector should work on select output (the remove_leading_0s path).

    select with an exact boolean condition should produce exact embeddings,
    so equals_vector should work. This is the simpler chain that
    remove_leading_0s relies on.
    """
    embedding = _make_digit_embedding()

    e3 = embedding.get_embedding("3")
    e7 = embedding.get_embedding("7")

    true_node = create_literal_value(e3)
    false_node = create_literal_value(e7)

    # Condition = true → should select e3
    cond_true = create_literal_value(torch.tensor([1.0]))
    selected = select(cond=cond_true, true_node=true_node, false_node=false_node)
    result = equals_vector(selected, e3)
    output = result.compute(n_pos=1, input_values={}).item()
    assert output == pytest.approx(1.0, abs=0.1), (
        f"equals_vector(select(true→3), 3) = {output:.4f}, expected ~1.0"
    )

    # Also verify it rejects the wrong digit
    result_wrong = equals_vector(selected, e7)
    output_wrong = result_wrong.compute(n_pos=1, input_values={}).item()
    assert output_wrong == pytest.approx(-1.0, abs=0.1), (
        f"equals_vector(select(true→3), 7) = {output_wrong:.4f}, expected ~-1.0"
    )

    # Condition = false → should select e7
    cond_false = create_literal_value(torch.tensor([-1.0]))
    selected_f = select(cond=cond_false, true_node=true_node, false_node=false_node)
    result_f = equals_vector(selected_f, e7)
    output_f = result_f.compute(n_pos=1, input_values={}).item()
    assert output_f == pytest.approx(1.0, abs=0.1), (
        f"equals_vector(select(false→7), 7) = {output_f:.4f}, expected ~1.0"
    )


def test_equals_vector_after_nested_select():
    """equals_vector should work after two levels of select (remove_leading_0s depth=2)."""
    embedding = _make_digit_embedding()

    e0 = embedding.get_embedding("0")
    e5 = embedding.get_embedding("5")
    e9 = embedding.get_embedding("9")

    # Level 1: select between 0 and 5 (choose 5)
    level1 = select(
        cond=create_literal_value(torch.tensor([1.0])),
        true_node=create_literal_value(e5),
        false_node=create_literal_value(e0),
    )

    # Level 2: select between level1 output and 9 (choose level1 = 5)
    level2 = select(
        cond=create_literal_value(torch.tensor([1.0])),
        true_node=level1,
        false_node=create_literal_value(e9),
    )

    result = equals_vector(level2, e5)
    output = result.compute(n_pos=1, input_values={}).item()
    assert output == pytest.approx(1.0, abs=0.1), (
        f"equals_vector after 2 selects = {output:.4f}, expected ~1.0"
    )


# ---------------------------------------------------------------------------
# map_to_table after map_to_table (the multiplication carry chain)
# ---------------------------------------------------------------------------


def test_map_to_table_chain_produces_correct_lookup():
    """A second map_to_table should produce the correct value when fed output of a first.

    This is the core multiplication blocker: multiply_digit_pair (map_to_table)
    feeds into sum_digits (map_to_table) for carry propagation.
    """
    embedding = _make_digit_embedding()
    table = _digit_identity_table(embedding)

    for digit in range(10):
        inp = create_literal_value(embedding.get_embedding(str(digit)))
        first = map_to_table(inp, table, default=embedding.get_embedding("0"))
        second = map_to_table(first, table, default=embedding.get_embedding("0"))
        result = second.compute(n_pos=1, input_values={}).squeeze()
        true_emb = embedding.get_embedding(str(digit))

        # Find closest digit
        min_dist = float("inf")
        closest = -1
        for d in range(10):
            dist = (result - embedding.get_embedding(str(d))).norm().item()
            if dist < min_dist:
                min_dist = dist
                closest = d

        assert closest == digit, (
            f"digit {digit}: chained map_to_table decoded as {closest} "
            f"(dist to correct={(result - true_emb).norm().item():.1f}, "
            f"dist to decoded={min_dist:.1f})"
        )


def test_sum_digits_on_map_to_table_output():
    """sum_digits should work when its inputs come from map_to_table.

    This directly tests the multiplication carry chain:
    multiply_digit_pair → sum_digits.
    """
    from torchwright.ops.relu.embedding_arithmetic import sum_digits

    embedding = _make_digit_embedding()

    # multiply_digit_pair(3, 4) = 12 → tens=1, ones=2
    # Then sum_digits(ones=2, 0, no_carry) should give (2, no_carry)
    prod_table_ones = {}
    for i in range(10):
        for j in range(10):
            key = torch.cat(
                [
                    embedding.get_embedding(str(i)),
                    embedding.get_embedding(str(j)),
                ]
            )
            prod_table_ones[key] = embedding.get_embedding(str((i * j) % 10))

    inp = concat(
        [
            create_literal_value(embedding.get_embedding("3")),
            create_literal_value(embedding.get_embedding("4")),
        ]
    )
    ones = map_to_table(inp, prod_table_ones, default=embedding.get_embedding("0"))

    zero = create_literal_value(embedding.get_embedding("0"))
    no_carry = create_literal_value(torch.tensor([-1.0]))
    digit_sum, _carry_out = sum_digits(embedding, ones, zero, no_carry)

    result = digit_sum.compute(n_pos=1, input_values={}).squeeze()
    expected = embedding.get_embedding("2")
    dist = (result - expected).norm().item()

    # Decode result
    min_dist = float("inf")
    decoded = -1
    for d in range(10):
        dd = (result - embedding.get_embedding(str(d))).norm().item()
        if dd < min_dist:
            min_dist = dd
            decoded = d

    assert decoded == 2, (
        f"sum_digits(map_to_table(3*4 ones), 0, no_carry) decoded as {decoded}, "
        f"expected 2 (dist={dist:.1f})"
    )


# ---------------------------------------------------------------------------
# remove_leading_0s chaining
# ---------------------------------------------------------------------------


def test_remove_leading_0s_single_level():
    """remove_leading_0s with max_removals=1 should work."""
    from torchwright.ops.relu.sequence_ops import remove_leading_0s

    embedding = _make_digit_embedding()
    e0 = embedding.get_embedding("0")
    e5 = embedding.get_embedding("5")

    seq = [create_literal_value(e0), create_literal_value(e5)]
    result = remove_leading_0s(embedding, seq, max_removals=1)

    # After removing one leading zero: [5, 5]
    out = result[0].compute(n_pos=1, input_values={}).squeeze()
    dist = (out - e5).norm().item()
    assert dist < 0.5, f"remove_leading_0s(1): first digit dist from '5' = {dist:.4f}"


def test_remove_leading_0s_two_levels():
    """remove_leading_0s with max_removals=2 should work.

    This requires equals_vector to work on select output (level 2 checks
    whether the shifted sequence still has a leading zero).
    """
    from torchwright.ops.relu.sequence_ops import remove_leading_0s

    embedding = _make_digit_embedding()
    e0 = embedding.get_embedding("0")
    e3 = embedding.get_embedding("3")

    seq = [create_literal_value(e0), create_literal_value(e0), create_literal_value(e3)]
    result = remove_leading_0s(embedding, seq, max_removals=2)

    # After removing two leading zeros: [3, 3, 3]
    out = result[0].compute(n_pos=1, input_values={}).squeeze()
    dist = (out - e3).norm().item()
    assert dist < 0.5, f"remove_leading_0s(2): first digit dist from '3' = {dist:.4f}"


def test_remove_leading_0s_three_levels():
    """remove_leading_0s with max_removals=3 — needed for multiplication results."""
    from torchwright.ops.relu.sequence_ops import remove_leading_0s

    embedding = _make_digit_embedding()
    e0 = embedding.get_embedding("0")
    e7 = embedding.get_embedding("7")

    seq = [
        create_literal_value(e0),
        create_literal_value(e0),
        create_literal_value(e0),
        create_literal_value(e7),
    ]
    result = remove_leading_0s(embedding, seq, max_removals=3)

    out = result[0].compute(n_pos=1, input_values={}).squeeze()
    dist = (out - e7).norm().item()
    assert dist < 0.5, f"remove_leading_0s(3): first digit dist from '7' = {dist:.4f}"


def test_remove_leading_0s_no_removal_needed():
    """remove_leading_0s should leave a sequence alone when there are no leading zeros."""
    from torchwright.ops.relu.sequence_ops import remove_leading_0s

    embedding = _make_digit_embedding()
    e4 = embedding.get_embedding("4")
    e2 = embedding.get_embedding("2")

    seq = [create_literal_value(e4), create_literal_value(e2)]
    result = remove_leading_0s(embedding, seq, max_removals=2)

    out0 = result[0].compute(n_pos=1, input_values={}).squeeze()
    out1 = result[1].compute(n_pos=1, input_values={}).squeeze()
    assert (out0 - e4).norm().item() < 0.5, "first digit should remain '4'"
    assert (out1 - e2).norm().item() < 0.5, "second digit should remain '2'"


def _assert_trimmed_relu(seq_tokens, max_removals, expected_tokens):
    from torchwright.ops.relu.sequence_ops import remove_leading_0s

    embedding = _make_digit_embedding()
    seq = [create_literal_value(embedding.get_embedding(t)) for t in seq_tokens]
    result = remove_leading_0s(embedding, seq, max_removals)
    assert len(result) == len(expected_tokens)
    for i, (node, tok) in enumerate(zip(result, expected_tokens, strict=False)):
        out = node.compute(n_pos=1, input_values={}).squeeze()
        dist = (out - embedding.get_embedding(tok)).norm().item()
        assert dist < 0.5, f"slot {i}: dist from '{tok}' = {dist:.4f}"


def test_remove_leading_0s_cap_limits_run():
    """Run length 2 with budget 1: exactly one zero is removed."""
    _assert_trimmed_relu(["0", "0", "3"], 1, ["0", "3", "3"])


def test_remove_leading_0s_interior_zero_untouched():
    """A zero after a nonzero digit is not a leading zero."""
    _assert_trimmed_relu(["5", "0", "2"], 2, ["5", "0", "2"])


def test_remove_leading_0s_cap_beyond_length():
    """A budget past n-1 is a no-op beyond pinning to the last element."""
    _assert_trimmed_relu(["0", "6"], 9, ["6", "6"])
