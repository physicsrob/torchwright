"""Probe the accumulated one-hot leak behind the Phase C 123*456 flake.

Follow-up to ``docs/onehot_accumulated_leak_postmortem.md``.  Three
questions, each answered by direct measurement on the real ops:

P1  **Mechanism.**  Where does the per-element ~1e-5 leak in a
    machine-built one-hot come from, bit for bit?  Hypothesis: the
    saturated hinge values are exact integers, but ``in_range``'s
    folded out_proj weight ``2/scale`` (0.02 — not representable in
    fp32) rounds each product at magnitude up to ``~20*n_slots``, and
    the near-cancellation of the two hinges in each ramp preserves
    those product roundings as the residue.  Test: predict every slot
    of ``in_range(t, t+1, 61)`` from four scalar fp32 products.

P2  **Guard soundness.**  The carry lookup's input is fully described
    by (integer total t, upstream noise delta) — 61 totals, delta
    bounded by upstream guards.  Sweep that whole space and bound the
    worst accumulated deviation against the shipped slack
    ``_lookup_numeric_slack(6, 1, 61)``.  Also sweep the two-block
    times-table shape, and measure sensitivity to the CPU thread count
    (the reference eval runs on CPU; run-to-run flake variation must
    come from environment-dependent reduction order, not CUDA).

P3  **Scale comparison.**  With ``scale`` a power of two
    (``2/scale`` exactly representable), every product in the
    saturated lanes is exact and integer-fed chains are bit-exact:
    leak identically zero.  The script runs the shipped scale (128
    since 2026-07-04) against the legacy 100 to pin that delta — the
    legacy leg reproduces the flake-era leak for reference.

Run locally:         ../.venv/bin/python -m scripts.investigate_onehot_leak
Cross-host (Modal):  make modal-run MODULE=scripts.investigate_onehot_leak CPU_ONLY=1
"""

from typing import Dict, List, Optional, Tuple

import torch

import torchwright.ops.swiglu.map_select as _map_select
import torchwright.ops.swiglu.onehot_table as _onehot_table
from torchwright.graph.node import suppress_checks
from torchwright.ops._math import _lookup_numeric_slack
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.linear import add_const, bool_to_01, concat
from torchwright.ops.swiglu.map_select import in_range
from torchwright.ops.swiglu.onehot_table import onehot_lookup

N_SLOTS = 61  # max_total + 1 for the 3-digit multiply (20*n, n=3)

# Upstream noise on a column total: sums of product-lookup outputs plus the
# previous carry, each inside its own guard — |delta| stays well under 1e-2.
COARSE_DELTAS = [0.0, 1e-5, -1e-5, 1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]
DENSE_DELTAS = torch.linspace(-3e-3, 3e-3, 201).tolist()


def _f32(x: float) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Graph builders (scale patched per build; weights are baked at build time)
# ---------------------------------------------------------------------------


def _patched_scale(value: Optional[float]):
    class _Ctx:
        def __enter__(self):
            self.saved = (_map_select.scale, _onehot_table.scale)
            if value is not None:
                _map_select.scale = value
                _onehot_table.scale = value

        def __exit__(self, *exc):
            _map_select.scale, _onehot_table.scale = self.saved

    return _Ctx()


def build_carry_chain(scale_override: Optional[float]) -> Dict[str, object]:
    """total -> in_range(total, total+1, 61) -> bool_to_01 -> lookups.

    The exact shape of calculator_simple.multiply_digit_seqs' carry sweep,
    minus the embedding-valued digit table (replaced by a 0/1-valued table
    of the same width class).
    """
    with _patched_scale(scale_override):
        total = create_input("total", 1, value_range=(-1.0, float(N_SLOTS)))
        ind = in_range(total, add_const(total, 1.0), N_SLOTS)
        onehot = bool_to_01(ind)

        keys = [torch.zeros(N_SLOTS) for _ in range(N_SLOTS)]
        for t, k in enumerate(keys):
            k[t] = 1.0
        carry_table = {keys[t]: torch.tensor([float(t // 10)]) for t in range(N_SLOTS)}
        d6_table = {  # the D6 repro's shape: half the rows at max magnitude
            keys[t]: torch.tensor([6.0 if t % 2 == 0 else 0.0]) for t in range(N_SLOTS)
        }
        b01_table = {  # 0/1-valued rows, the digit table's magnitude class
            keys[t]: torch.tensor([1.0 if t % 2 == 0 else 0.0]) for t in range(N_SLOTS)
        }
        carry = onehot_lookup(onehot, carry_table, torch.tensor([0.0]))
        d6 = onehot_lookup(onehot, d6_table, torch.tensor([0.0]))
        b01 = onehot_lookup(onehot, b01_table, torch.tensor([0.0]))
    return {
        "ind": ind,
        "onehot": onehot,
        "carry": carry,
        "d6": d6,
        "b01": b01,
    }


def build_times_table(scale_override: Optional[float]) -> Dict[str, object]:
    """Two machine-built one-hot digits -> two-block times-table lookup.

    The multi-block (swiglu-lane) path of onehot_lookup, fed by the same
    in_range/bool_to_01 construction — calculator_simple's step-1 shape
    with machine-built keys instead of literal embeddings.
    """
    with _patched_scale(scale_override):
        a = create_input("a", 1, value_range=(-1.0, 10.0))
        b = create_input("b", 1, value_range=(-1.0, 10.0))
        a1h = bool_to_01(in_range(a, add_const(a, 1.0), 10))
        b1h = bool_to_01(in_range(b, add_const(b, 1.0), 10))
        key_node = concat([a1h, b1h])

        table: Dict[torch.Tensor, torch.Tensor] = {}
        for da in range(10):
            for db in range(10):
                k = torch.zeros(20)
                k[da] = 1.0
                k[10 + db] = 1.0
                table[k] = torch.tensor([float(da * db // 10), float(da * db % 10)])
        product = onehot_lookup(key_node, table, torch.tensor([0.0, 0.0]))
    return {"product": product}


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def sweep_carry(
    nodes: Dict[str, object], deltas: List[float], device: torch.device
) -> Dict[str, torch.Tensor]:
    ts, dvals = [], []
    for t in range(N_SLOTS):
        for d in deltas:
            ts.append(t)
            dvals.append(d)
    x = (
        torch.tensor(ts, dtype=torch.float32) + torch.tensor(dvals, dtype=torch.float32)
    ).unsqueeze(1)
    inp = {"total": x.to(device)}
    n = len(ts)
    # The sweep deliberately feeds out-of-tolerance deltas, so the ops'
    # attached asserts would fire — suppress them; this script MEASURES
    # the drift those asserts guard against.
    with suppress_checks():
        out = {
            name: nodes[name].compute(n, inp).cpu()
            for name in ("ind", "onehot", "carry", "d6", "b01")
        }
    t_idx = torch.tensor(ts)
    ideal_ind = -torch.ones(n, N_SLOTS)
    ideal_ind[torch.arange(n), t_idx] = 1.0
    out["leak"] = out["ind"] - ideal_ind  # per-element indicator leak
    out["t"] = t_idx
    out["delta"] = torch.tensor(dvals)
    out["carry_dev"] = (out["carry"].squeeze(1) - (t_idx // 10).float()).abs()
    ideal_d6 = torch.tensor([6.0 if t % 2 == 0 else 0.0 for t in ts])
    out["d6_dev"] = (out["d6"].squeeze(1) - ideal_d6).abs()
    ideal_b01 = torch.tensor([1.0 if t % 2 == 0 else 0.0 for t in ts])
    out["b01_dev"] = (out["b01"].squeeze(1) - ideal_b01).abs()
    # What the closing assert actually checks: distance outside [lo, hi].
    for name, hi in (("carry", 6.0), ("d6", 6.0), ("b01", 1.0)):
        v = out[name].squeeze(1)
        out[f"{name}_viol"] = torch.maximum(-v, v - hi).clamp(min=0.0)
    return out


def sweep_times_table(
    nodes: Dict[str, object], deltas: List[float], device: torch.device
) -> Dict[str, torch.Tensor]:
    rows: List[Tuple[int, int, float]] = []
    for da in range(10):
        for db in range(10):
            for d in deltas:
                rows.append((da, db, d))
    a = torch.tensor([r[0] + r[2] for r in rows], dtype=torch.float32).unsqueeze(1)
    b = torch.tensor([float(r[1]) for r in rows], dtype=torch.float32).unsqueeze(1)
    n = len(rows)
    with suppress_checks():
        product = (
            nodes["product"].compute(n, {"a": a.to(device), "b": b.to(device)}).cpu()
        )
    ideal = torch.tensor(
        [[float(r[0] * r[1] // 10), float(r[0] * r[1] % 10)] for r in rows]
    )
    return {
        "dev": (product - ideal).abs().max(dim=1).values,
        "rows": torch.tensor([(r[0], r[1]) for r in rows]),
        "delta": torch.tensor([r[2] for r in rows]),
    }


# ---------------------------------------------------------------------------
# P1: bit-level prediction of every in_range slot from four scalar products
# ---------------------------------------------------------------------------


def predict_in_range_slots(scale: float, sharpness: float) -> torch.Tensor:
    """Predict in_range(t, t+1, N_SLOTS) leak per (t, slot) in scalar fp32.

    Saturated lanes: hidden value is the exact gate argument (sigma == 1
    or == 0 bit-exactly); the only inexact steps are the two or four
    out_proj products lane * (+-2/scale) and their summation.  Predicts
    with the natural adjacent-lane association ((A - B) - (C - D)).
    """
    w = _f32(2.0) / _f32(scale)  # fl(2/scale)
    coeff = float(scale * sharpness)
    leak = torch.zeros(N_SLOTS, N_SLOTS)
    for t in range(N_SLOTS):
        for i in range(N_SLOTS):
            c = i + 0.5
            # gate args, exact integer arithmetic at these magnitudes
            x_pl = _f32(coeff * (c - t))
            x_pu = _f32(coeff * (c - t - 1.0))
            terms = []
            for arg0 in (x_pl, x_pu):
                pair = []
                for shift in (0.0, float(scale)):
                    arg = _f32(float(arg0) - shift)
                    if float(arg) >= 17.0 * 1.0:
                        pair.append(arg * w)  # one fp32 product rounding
                    else:
                        pair.append(_f32(0.0))  # sigma underflows to 0
                terms.append(pair[0] - pair[1])  # Sterbenz-exact
            # matmul reduction (adjacent-lane pairwise association), then
            # out_bias, then the same f32 subtraction the sweep uses.
            out = (terms[0] - terms[1]) + _f32(-1.0)
            ideal = _f32(1.0 if i == t else -1.0)
            leak[t, i] = float(out - ideal)
    return leak


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.3e}"


def run(device: torch.device) -> None:
    torch.set_default_dtype(torch.float32)
    fl_002 = float((_f32(2.0) / _f32(100.0)).item())
    print(f"device={device}  torch={torch.__version__}")
    print(f"fl(2/100) = {fl_002!r}  (off 0.02 by {_fmt(fl_002 - 0.02)})")
    print(f"fl(2/128) = {float((_f32(2.0) / _f32(128.0)).item())!r}  (exact)")
    slack_carry = _lookup_numeric_slack(6.0, 1.0, N_SLOTS)
    slack_b01 = _lookup_numeric_slack(1.0, 1.0, N_SLOTS)
    slack_tt = _lookup_numeric_slack(9.0, 1.0, 20)
    print(
        f"shipped slacks: carry/d6={_fmt(slack_carry)}  b01={_fmt(slack_b01)}  "
        f"times-table={_fmt(slack_tt)}  (old fixed: 1.000e-03)\n"
    )

    shipped = float(_map_select.scale)
    for scale_override, label in (
        (None, f"scale={shipped:g} (shipped)"),
        (100.0, "scale=100 (legacy)"),
    ):
        print(f"=== {label} ===")
        chain = build_carry_chain(scale_override)
        tt = build_times_table(scale_override)

        # ---- P1: mechanism, delta = 0 ----
        s = sweep_carry(chain, [0.0], device)
        leak = s["leak"]  # (61, 61): row = total t, col = slot i
        below = torch.tril(torch.ones(N_SLOTS, N_SLOTS), diagonal=-1).bool()
        above = torch.triu(torch.ones(N_SLOTS, N_SLOTS), diagonal=1).bool()
        diag = torch.eye(N_SLOTS).bool()
        print("P1 per-element indicator leak at integer totals (delta=0):")
        print(
            f"  slots below t: max|leak| = {_fmt(leak[below].abs().max().item())}"
            f"   (exact-zero fraction {float((leak[below] == 0).float().mean()):.4f})"
        )
        print(f"  slot i == t:   max|leak| = {_fmt(leak[diag].abs().max().item())}")
        print(
            f"  slots above t: max|leak| = {_fmt(leak[above].abs().max().item())}"
            f"  mean|leak| = {_fmt(leak[above].abs().mean().item())}"
        )
        row_sum = leak.abs().sum(dim=1)
        worst_t = int(row_sum.argmax())
        print(
            f"  sum_k |leak_k| per total: max = {_fmt(row_sum.max().item())} "
            f"at t={worst_t}, mean = {_fmt(row_sum.mean().item())}"
        )
        eff_scale = scale_override if scale_override is not None else shipped
        pred = predict_in_range_slots(eff_scale, _map_select.step_sharpness)
        mismatch = (pred - leak).abs()
        exact_frac = float((mismatch == 0).float().mean())
        print(
            f"  four-product prediction: bit-exact on {exact_frac:.4f} of slots, "
            f"max mismatch = {_fmt(mismatch.max().item())}"
        )

        # ---- P2: guard-soundness sweep over (t, delta) ----
        deltas = sorted(set(COARSE_DELTAS + DENSE_DELTAS))
        s2 = sweep_carry(chain, deltas, device)
        for name, slack in (
            ("carry", slack_carry),
            ("d6", slack_carry),
            ("b01", slack_b01),
        ):
            dev = s2[f"{name}_dev"]
            viol = s2[f"{name}_viol"]
            k = int(viol.argmax())
            at_zero = viol[s2["delta"] == 0.0]
            print(
                f"P2 {name:5s} lookup: worst range violation = "
                f"{_fmt(viol.max().item())}"
                f"  (t={int(s2['t'][k])}, delta={s2['delta'][k]:+.1e})"
                f"  vs slack {_fmt(slack)}"
                f"  -> headroom x{slack / max(viol.max().item(), 1e-30):.1f}"
                f"   [delta=0 worst: {_fmt(at_zero.max().item())};"
                f" worst |out-ideal|: {_fmt(dev.max().item())}]"
            )
        s3 = sweep_times_table(tt, [0.0, 1e-4, -1e-4, 1e-3, -1e-3], device)
        k = int(s3["dev"].argmax())
        print(
            f"P2 times-table (2-block swiglu path): worst dev = "
            f"{_fmt(s3['dev'].max().item())} (a={int(s3['rows'][k][0])}, "
            f"b={int(s3['rows'][k][1])}, delta={s3['delta'][k]:+.1e}) "
            f"vs slack {_fmt(slack_tt)}"
        )

        # ---- P2b: environment sensitivity (CPU reduction order) ----
        if device.type == "cpu":
            base = None
            spreads = []
            saved_threads = torch.get_num_threads()
            for nthreads in (1, 2, 4, saved_threads):
                torch.set_num_threads(nthreads)
                si = sweep_carry(chain, [0.0], device)
                if base is None:
                    base = si["carry"]
                spreads.append((nthreads, float((si["carry"] - base).abs().max())))
            torch.set_num_threads(saved_threads)
            print(
                "P2b carry-value spread across torch thread counts: "
                + ", ".join(f"{n}t:{_fmt(sp)}" for n, sp in spreads)
            )
        print()


def main() -> None:
    # Reference eval computes on whatever device the graph tensors were
    # built on — plain CPU, same as the flaked test (the conftest device
    # fixture only steers the compiler).  There is no CUDA leg to probe.
    run(torch.device("cpu"))


if __name__ == "__main__":
    main()
