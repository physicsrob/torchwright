"""Op-level repro: compiled ``reciprocal`` error at count_since_marker's grid.

The calculator wide-operand truncation traced to the compiled count
diverging from its reference by up to 0.445 in the last gaps before
``max_gap`` (scripts/investigate_region_truncation.py).  This isolates
the op: build ``reciprocal`` with the *exact* grid ``count_since_marker``
derives for a given ``max_gap``, feed the exact attention means
``1/(gap+1)``, and print three-way values per gap — exact ``1/x``, the
node's own ``compute`` (the oracle the probe compared against), and the
compiled forward.  Splits the fault between the op's emitted lanes
(compiled != compute) and the grid design (compute != exact).

    uv run python -m scripts.investigate_reciprocal_compiled
"""

import math

import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import reference_eval
from torchwright.ops._math import _RECIP_REL_SAFETY
from torchwright.ops.inout_nodes import create_input
from torchwright.ops.swiglu.arithmetic_ops import reciprocal

MAX_GAPS = [7, 10, 13]


def grid(max_gap: int) -> tuple[float, float, float, int]:
    """Replicate count_since_marker's reciprocal grid derivation verbatim."""
    recip_lo = 1.0 / (max_gap + 1.5)
    recip_hi = 1.5
    target_rel = 0.5 / (max_gap + 1.0) / _RECIP_REL_SAFETY
    r_max = 1.0 + math.sqrt(8.0 * target_rel)
    n_breakpoints = max(32, int(math.log(recip_hi / recip_lo) / math.log(r_max)) + 2)
    step = (recip_hi - recip_lo) / (n_breakpoints - 1)
    return recip_lo, recip_hi, step, n_breakpoints


def main() -> None:
    for max_gap in MAX_GAPS:
        lo, hi, step, n_bp = grid(max_gap)
        print(
            f"\n== max_gap={max_gap}: grid [{lo:.4f}, {hi}] step {step:.4f} "
            f"({n_bp} breakpoints; steep-end segment spans counts "
            f"{1 / lo:.1f} -> {1 / (lo + step):.1f}) =="
        )
        x = create_input("x", 1)
        out = reciprocal(x, min_value=lo, max_value=hi, step=step)
        means = torch.tensor([[1.0 / (g + 1.0)] for g in range(max_gap + 1)])
        n_pos = means.shape[0]
        oracle = reference_eval(out, {"x": means}, n_pos)[out]
        compiled = compile_headless(out, d=256, d_head=16)
        compiled(
            compiled.build_prefill({"x": means}, n_pos),
            debug=True,
        )
        cv = compiled.debug_value(out)
        assert cv is not None, "output node has no residual assignment"
        print("  gap   mean     exact  oracle(err)      compiled(err)")
        for g in range(max_gap + 1):
            exact = g + 1.0
            o, c = float(oracle[g, 0]), float(cv[g, 0])
            print(
                f"  {g:3d}  {float(means[g, 0]):.5f}  {exact:6.2f} "
                f"{o:8.3f} ({o - exact:+.3f})  {c:8.3f} ({c - o:+.3f} vs oracle)"
            )


if __name__ == "__main__":
    main()
