"""Static-routing measurement for MLP Add routing (plan Step 5 gate).

docs/plan_additional_mlp_routing.md, *Policy and compatibility decision*:
the 2026-07-14 sweep of these configurations found a static MLP-Add
default costs one layer per Add wedged between MLP-sublayer ops
(calculator +5, fibonacci +1 at ``optimize=0``) while ``optimize>=1``
is unaffected (flex routing decides per node) — so the shipping default
keeps Adds on attention (``SchedulingPolicy.add_in_attention="always"``)
and the ``opt0-addmlp`` row documents the known static cost.

Configurations per graph (natural width):

- ``opt0-legacy`` — ``optimize=0``, ``LEGACY_POLICY`` (everything on
  attention, the historical heuristic).
- ``opt0-default`` — ``optimize=0``, default policy (Linears to MLP,
  Adds to attention).
- ``opt0-addmlp`` — ``optimize=0``, ``add_in_attention="never"``
  (the rejected static MLP-Add default).
- ``opt1-legacy`` — ``optimize=1``, ``LEGACY_POLICY`` +
  ``cpsat_flex_routing=False`` (the documented legacy CP-SAT pair).
- ``opt1-default`` — ``optimize=1``, default policy + flex routing (the
  production configuration: Add routes are solver decisions).

Each run reports ``n_layers``, the schedule origin (heuristic / solver /
fallback), wall time, and the realized Add-op mix read off the replay plan
(attention ``add_into``/``compute_add`` vs MLP ``add_into_bypass``/
``compute_add_bypass``), plus the plan's emitted head and bypass-slot
totals.  The schedule cache is redirected to a fresh temp dir so every
solve is cold.

Run on Modal (CP-SAT solves want cores; compiles want the sanctioned box):

    make modal-run MODULE=scripts.measure_add_routing_flip

Options via ARGS, e.g.:

    make modal-run MODULE=scripts.measure_add_routing_flip ARGS="--graphs calculator caesar"

The DOOM flagship is deliberately not swept here — it lives in
torchwright_doom and its compile is a separate budgeted Modal run; run it
against this branch before flipping the default for the flagship.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Any

# Cold solves: isolate the schedule cache before torchwright imports read it.
os.environ["TW_SCHEDULE_CACHE_DIR"] = tempfile.mkdtemp(prefix="tw-add-flip-")

from torchwright.compiler.forward.compile import forward_compile
from torchwright.compiler.forward.scheduling_policy import (
    LEGACY_POLICY,
    SchedulingPolicy,
)

_ADD_ATTN_OPS = ("add_into", "compute_add")
_ADD_MLP_OPS = ("add_into_bypass", "compute_add_bypass")


def _example_specs():
    from examples import (
        binary_increment,
        caesar_cipher,
        calculator_simple,
        fibonacci,
        sort_digits_v1,
    )

    return {
        # calculator: d pinned at the geometry the committed measurements ran
        # at (the canonical publish D_MODEL=8192 would change the measurement
        # context); d_head follows the module — baked into the graph.
        "calculator": (
            lambda: calculator_simple.create_network_parts()[0],
            1024,
            calculator_simple.D_HEAD,
        ),
        "caesar": (
            lambda: caesar_cipher.create_network_parts()[0],
            caesar_cipher.D_MODEL,
            caesar_cipher.D_HEAD,
        ),
        "sort_digits": (
            lambda: sort_digits_v1.create_network_parts()[0],
            sort_digits_v1.D_MODEL,
            sort_digits_v1.D_HEAD,
        ),
        "fibonacci": (
            lambda: fibonacci.create_network_parts()[0],
            fibonacci.D_MODEL,
            fibonacci.D_HEAD,
        ),
        "binary_increment": (
            lambda: binary_increment.create_network_parts()[0],
            binary_increment.D_MODEL,
            binary_increment.D_HEAD,
        ),
    }


def _run_config(out, d, d_head, *, optimize, policy, flex):
    captured = {}

    def on_layer(_i, _layer) -> None:
        pass

    on_layer.on_replay_plan = lambda plan: captured.__setitem__("plan", plan)
    t0 = time.perf_counter()
    try:
        net = forward_compile(
            d=d,
            d_head=d_head,
            output_node=out,
            device="cpu",
            verbose=False,
            optimize=optimize,
            policy=policy,
            cpsat_flex_routing=flex,
            on_layer_compiled=on_layer,
        )
    except Exception as exc:  # noqa: BLE001 - record the failure, keep sweeping
        return {
            "status": f"FAIL:{type(exc).__name__}",
            "wall_s": time.perf_counter() - t0,
        }
    wall = time.perf_counter() - t0
    if optimize == 0:
        origin = "heuristic"
    else:
        stats = net.cpsat_solve_stats
        origin = (
            "solver"
            if stats is not None
            and stats.status_name in ("OPTIMAL", "FEASIBLE", "CACHED")
            else "fallback"
        )
    plan = captured.get("plan")
    add_attn = add_mlp = heads = slots = None
    if plan is not None:
        types = [
            op.op_type
            for layer in plan.layers
            for op in (*layer.attention_ops, *layer.mlp_ops)
        ]
        add_attn = sum(types.count(t) for t in _ADD_ATTN_OPS)
        add_mlp = sum(types.count(t) for t in _ADD_MLP_OPS)
        heads = plan.total_attention_heads
        slots = plan.total_mlp_bypass_slots
    return {
        "status": origin,
        "wall_s": wall,
        "n_layers": len(net.layers),
        "add_attn_ops": add_attn,
        "add_mlp_ops": add_mlp,
        "emitted_heads": heads,
        "bypass_slots": slots,
    }


_CONFIGS: list[tuple[str, dict[str, Any]]] = [
    ("opt0-legacy", {"optimize": 0, "policy": LEGACY_POLICY, "flex": True}),
    ("opt0-default", {"optimize": 0, "policy": None, "flex": True}),
    (
        "opt0-addmlp",
        {
            "optimize": 0,
            "policy": SchedulingPolicy(add_in_attention="never"),
            "flex": True,
        },
    ),
    ("opt1-legacy", {"optimize": 1, "policy": LEGACY_POLICY, "flex": False}),
    ("opt1-default", {"optimize": 1, "policy": None, "flex": True}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="*", default=None)
    args = parser.parse_args()

    specs = _example_specs()
    names = args.graphs or list(specs)
    print(f"schedule cache isolated at {os.environ['TW_SCHEDULE_CACHE_DIR']}")
    for name in names:
        build, d, d_head = specs[name]
        out = build()
        print(f"\n=== {name} (d={d}, d_head={d_head}) ===")
        print(
            f"  {'config':<14} {'status':<10} {'layers':>6} {'addA':>5} "
            f"{'addM':>5} {'heads':>6} {'slots':>6} {'wall_s':>8}"
        )
        for label, cfg in _CONFIGS:
            policy = cfg["policy"] or SchedulingPolicy()
            r = _run_config(
                out,
                d,
                d_head,
                optimize=cfg["optimize"],
                policy=policy,
                flex=cfg["flex"],
            )
            print(
                f"  {label:<14} {r['status']:<10} "
                f"{r.get('n_layers', '-'):>6} {r.get('add_attn_ops', '-'):>5} "
                f"{r.get('add_mlp_ops', '-'):>5} {r.get('emitted_heads', '-'):>6} "
                f"{r.get('bypass_slots', '-'):>6} {r['wall_s']:>8.1f}"
            )


if __name__ == "__main__":
    main()
