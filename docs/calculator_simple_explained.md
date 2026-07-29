# How the Simple Calculator Works

`calculator_simple.py` builds a calculator for expressions such as `12*34`. It
reads the two numbers and the operator, performs the arithmetic, and writes the
answer one character at a time.

The code starts from pencil-and-paper arithmetic: addition carries from right
to left, subtraction borrows, and multiplication works with place-value
columns. The model is not trained to discover these rules. Torchwright converts
the rules in the Python code into the fixed internal settings of an ordinary
text-generating model.

## Building the Calculator

The file supplies its arithmetic procedures to a shared calculator builder:

```python
return build_calculator(
    max_digits,
    add_digit_seqs=add_digit_seqs,
    subtract_digit_seqs=subtract_digit_seqs,
    multiply_digit_seqs=multiply_digit_seqs,
)
```

This code builds the calculator rather than evaluating a particular
expression. The shared builder handles reading an input such as `12*34\n`,
padding shorter operands with zeros, choosing the requested operation, and
producing the answer. The newline marks the end of the expression and later
acts as the signal to begin output.

## Multiplication

The multiplication code starts from long multiplication, then makes one useful
change: it collects all the digit products before carrying. The work falls into
three stages. It looks up every digit-sized product, places those products into
columns, and makes one carry pass across the column totals.

### 1. Build the Times Table

The code starts by constructing a 100-entry Python mapping for the `0–9` times
table. Each product is stored as separate tens and ones:

```python
for a in range(10):
    for b in range(10):
        key = torch.cat(
            [embedding.get_embedding(str(a)), embedding.get_embedding(str(b))]
        )
        product_table[key] = torch.tensor(
            [float(a * b // 10), float(a * b % 10)]
        )
```

For example, `7 × 8` is stored as `(5, 6)`. This Python loop runs once while
building the graph. Every later call to `onehot_lookup` materializes the
mapping as an independent 100-lane FFN node; the finished transformer does not
contain one shared table queried by every digit pair.

### 2. Place the Products into Columns

The next loop creates one lookup for every pair of digit positions:

```python
product = onehot_lookup(
    concat([seq1[i], seq2[j]]), product_table, default_product
)
tens = slice_columns(product, 0, 1, name="product_tens")
ones = slice_columns(product, 1, 1, name="product_ones")

columns[i + j].append(tens)
columns[i + j + 1].append(ones)
```

Each pair gets its own lookup, so all the digit products can be found together.
For the default three-digit calculator, there are `3 × 3 = 9` independent
lookup circuits, comprising 900 FFN lanes. The operands, columns, and result
are all stored most-significant digit first. For operands at positions `i` and
`j`, the product's tens go into column `i + j` and its ones go into the next
column, `i + j + 1`.

For `12 × 34`, the nonzero columns look like this from most significant to
least significant:

| Column | Contributions | Result |
| --- | --- | --- |
| Hundreds | `1 × 3` contributes `3`, plus the carry | Write `4` |
| Tens | `2 × 3` contributes `6`; `1 × 4` contributes `4` | Write `0`, carry `1` |
| Ones | `2 × 4` contributes `8` | Write `8` |

### 3. Make One Carry Pass

The last multiplication step moves from the least-significant column to the
most-significant one. The essential part of the code is:

```python
for column in reversed(columns):
    contributions = column or [zero_scalar]
    total = add(sum_nodes(contributions), carry)
    total_onehot = bool_to_01(
        in_range(total, add_const(total, 1.0), max_total + 1)
    )
    out_lsb_first.append(
        onehot_lookup(total_onehot, digit_table, embedding.get_embedding("0"))
    )
    carry = piecewise_linear(
        total,
        sorted(carry_knots),
        carry_knots.__getitem__,
        input_scale=step_sharpness,
        name="carry_staircase",
    )
```

Although `columns` is stored most-significant first, carrying must proceed from
right to left, so the loop traverses `reversed(columns)`. For each column,
`total` combines its contributions with the incoming carry. The lookup selects
the final decimal digit, while `carry_staircase` calculates how many tens move
into the next column. This stage is sequential because each column needs the
carry produced by the column to its right. The loop accumulates the output
least-significant digit first, then reverses it before returning the result.

The result is held at a fixed width, so `12 × 34` may initially appear as
`000408`. The leading zeros are removed later.

## Choosing and Formatting the Answer

At this point, the calculator has paths for addition, subtraction, and
multiplication. The operator flags choose the correct answer one slot at a
time:

```python
answer = [
    switch(
        [is_plus, is_minus, is_times],
        [add_seq[i], sub_seq[i], mul_seq[i]],
    )
    for i in range(seq_len)
]
```

The selected answer is then shifted left to remove unnecessary leading zeros.
For a negative subtraction, the formatter inserts `-` at the front. It always
preserves one zero when the answer itself is zero.

## Choosing the Output at Every Token

One slightly odd consequence of putting the calculator inside a text generator
is that computing `408` is not the final step. The same calculator machinery
runs at every token position and must decide which single character should
come next.

It uses the newline as a landmark and computes `steps_since`: the current
position in the output sequence. It compares that position with every answer
slot, allowing only the matching slot through:

```python
for i, value in enumerate(seq):
    at_slot_i = bool_all_true(
        [
            compare(steps_since, thresh=i - 0.5),
            compare(negate(steps_since), thresh=-(i + 0.5)),
        ]
    )
    out_values.append(cond_gate(at_slot_i, value))
```

At the newline, the count selects answer slot `0`, so the model predicts the
first character. That character is appended to the text, the position advances,
and the same calculation selects slot `1`. For `12*34\n`, the successive
predictions are:

```text
4
0
8
<eos>
```

`<eos>` is a special end marker that tells the text generator to stop. On each
pass, the calculator determines the answer again, and the position logic lets
through the one character that belongs at the current token.
