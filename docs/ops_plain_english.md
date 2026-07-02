# Ops in plain English

Each op as a one-line description plus its formula in the core primitives
(`Mul`, `Swish`, `Linear`, `Add`). `scale` is a sharpness constant (~10–15).

## The gate, and how to build a real product

The only multiply the hardware has is the **gate**: `Mul(Swish(g), u)` — one
operand always passes through `Swish`, the other goes through untouched. So in
every formula below, a `Mul`'s first argument is always a `Swish(...)`.

A *clean* product of two live values is therefore not a primitive — you **build**
it from two gates, pairing `+a`/`+b` with `−a`/`−b` so the Swish's sigmoid
factors cancel:

```
multiply(a, b) = Add( Mul(Swish(a), b), Mul(Swish(-a), -b) )   =  a·b   (exact)
```

This is exact because `Swish(z) = z·σ(z)` and `σ(a) + σ(-a) = 1`, so the two
gates sum to `a·b·(σ(a) + σ(-a)) = a·b`. This `±` identity is the foundation of
every arithmetic op below; `select` uses the same gate to apply an *indicator*
to a value instead.

---

### select(cond, a, b)

Choose `a` if `cond` is true, or `b` if `cond` is false.

```
select(cond, a, b) = Add( Mul(Swish(scale·cond)/scale,  a),
                          Mul(Swish(-scale·cond)/scale, b) )
```

Notes:
- `cond` is ±1 (true = +1, false = −1), enforced by an assert.
- `Swish(scale·cond)/scale ≈ ReLU(cond)`: it's 1 at `cond=+1` and 0 at
  `cond=−1`. So the two gates are complementary on/off indicators — one picks
  `a`, the other picks `b`.
- The `/scale` matters: Swish is not a 0/1 gate. It grows like the identity on
  the right, so raw `Swish(scale·cond) ≈ scale` for `cond=+1`. Dividing by
  `scale` turns it back into a clean indicator. The `/scale` folds into the
  projections, so it's free.
- Error is `~e^{-scale}` (from `Swish` not being exactly the identity at
  `scale`). It relies on `cond` sitting safely at ±1, away from Swish's bend at
  0 — the assert guarantees that.
- Fits in one gated sublayer (hidden width 2).

---

### cond_gate(cond, inp)

Output `inp` if `cond` is true, else `0`.

```
cond_gate(cond, inp) = Mul(Swish(scale·cond)/scale, inp)
```

Notes:
- `select` with the false-branch pinned to zero — just the `a` term.
- `Swish(scale·cond)/scale ≈ ReLU(cond)`: 1 when `cond=+1`, 0 when `cond=−1`.
  Multiplying `inp` by it passes the value through or zeroes it.
- `cond` is ±1, enforced by an assert; same `~e^{-scale}` error as `select`.
- Because the multiply is direct, error scales with the *actual* value of
  `inp`, not the range maximum — so small gated values stay accurate.
- Fits in one gated sublayer (hidden width = width of `inp`).

---

### map_to_table(inp, table, default)

Return the value whose key matches `inp`, else `default`.

```
gate(key) = Swish(scale·(inp·key − key·key) + C) / C     # a scalar, ≈1 at the matching key, ≈0 at others
result    = default + gates @ (values − default)
```

Notes:
- `inp·key` is a plain dot product with **one** key. Computing `gate(key)` for
  every entry gives the vector `gates` (length `N` = number of entries).
- `values − default` is the table of value-deltas, one entry per row
  (`N × d_value`); `gates @ (values − default)` scales each entry's value by its
  gate and adds them up. For scalar values this is just a dot product.
- Still `Linear`, not `Mul`: the values are constants, so the scaling folds into
  weights. `map_to_table` is a pure `Linear → Swish → Linear` lookup — no gate
  primitive involved.
- `C` lifts the matching key into Swish's straight region so its value comes
  through un-shrunk; `scale` sets how hard the others are pushed toward 0.
- Bumps can overlap for nearby keys, so it's an approximate match, not exact
  selection. Losers must be pushed *past* Swish's small negative dip, not just
  below zero, or they leak.

---

### multiply(a, b)

Multiply two live values.

```
multiply(a, b) = Add( Mul(Swish(a), b), Mul(Swish(-a), -b) )
```

Notes:
- Exact (`a·b`), for all `a`, `b` — no range limit, no grid. See *The gate*
  above for why: the `±` pair makes the Swish sigmoids sum to 1.
- Both terms share the sign of `a·b`, so they add constructively — no
  catastrophic cancellation.
- One gated sublayer, hidden width 2 (a `+a` lane and a `−a` lane, summed).
- This replaces the ReLU-era workarounds for multiplication (the quarter-square
  construction in `multiply_2d`, the `signed_multiply` chain).

---

### square(inp)

Compute `inp²`.

```
square(inp) = Add( Mul(Swish(inp), inp), Mul(Swish(-inp), -inp) )
```

Notes:
- `multiply(a, b)` with `a = b = inp`. Exact (`x²`) for all `inp`.
- Both terms are `x²·σ(±inp)` — non-negative, so they add cleanly.
- Drops the current `[0, max_value]` restriction, the `step`/grid, and the huge
  near-zero relative error of the piecewise-linear version (which approximates
  `x²` by straight segments and is worst exactly where `x²` is smallest).
- One gated sublayer, hidden width 2.
