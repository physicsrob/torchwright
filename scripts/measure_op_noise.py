"""Measure per-op noise for every op in ``_TARGET_OPS``.

Canonical source of truth for noise numbers is ``docs/op_noise_data.json``;
``docs/numerical_noise.md`` and the per-op docstring footers are *generated*
from that JSON. Run::

    make measure-noise

to refresh all three artefacts. Run with ``--check`` to fail instead of
rewriting.

Drift between the committed JSON and what the current code actually
measures is caught on every CI run by
``tests/docs/test_numerical_noise_drift.py``, which re-invokes
``_measure_all`` and asserts the regenerated JSON matches the committed
copy. Format consistency between JSON, markdown, and docstring footers is
caught separately by ``tests/docs/test_numerical_noise_consistency.py``.

To add a new op, append a :class:`TargetOp` to ``_target_ops()`` and run
``make measure-noise``. See ``docs/numerical_noise.md`` for methodology.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TypedDict

import torch

from torchwright.debug.noise import (
    InputDistribution,
    NoiseMeasurement,
    measure_op_isolated,
    update_docstring_footer,
)
from torchwright.graph import Node
from torchwright.ops.const import step_sharpness
from torchwright.ops.relu.arithmetic_ops import (
    abs as abs_op,
    ceil_int,
    clamp,
    compare,
    floor_int,
    min as min_op,
    mod_const,
    multiply_2d,
    multiply_integers,
    piecewise_linear,
    reciprocal,
    square,
    thermometer_floor_div,
)
from torchwright.ops.relu.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    cond_gate,
    equals_vector,
)
from torchwright.ops import swiglu as swiglu_ops

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_JSON = REPO_ROOT / "docs" / "op_noise_data.json"
DOCS_MD = REPO_ROOT / "docs" / "numerical_noise.md"


def footer_source_files() -> list[Path]:
    """Every op-library module carrying a generated noise footer.

    Derived from ``_target_ops()`` so the set tracks the library layout
    (``ops/relu/``, ``ops/swiglu/``) — ``modal_measure_noise.py`` syncs
    exactly these files back after a remote measurement run.
    """
    return sorted({REPO_ROOT / t.source_file for t in _target_ops()})


# ---------------------------------------------------------------------------
# Target-op records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetOp:
    """Declarative record for one op we measure.

    ``build_graph`` receives ``{name: InputNode}`` and returns the output
    ``Node``. ``reference_fn`` receives the same input-tensor dict (as passed
    to ``Node.compute``) and returns the oracle output tensor.

    ``build_graphs_per_distribution`` overrides ``build_graph`` for specific
    distribution names. Use this when different production callsites of the
    same op use different parameters (e.g., `reciprocal` with different
    `min_value` in `wall.py:431` vs `wall.py:813`), and a single combined
    graph would have weaker precision than either production configuration.
    ``reference_fns_per_distribution`` is its sibling for modes whose exact
    math differs (e.g. `floor_int` with an `output_map`): the mode stays a
    distribution of the one op — one (machine, name), one docstring footer
    aggregating across modes — instead of masquerading as a separate op.

    ``machine`` is the op-library axis: ``"relu"`` (`torchwright/ops/relu/`,
    frozen) or ``"swiglu"`` (`torchwright/ops/swiglu/`). The two libraries
    share op names (both have a ``square``), so ops are identified by
    ``(machine, name)`` everywhere downstream — the JSON, the markdown, and
    the drift/consistency tests.
    """

    name: str
    module: str
    build_graph: Callable[[Dict[str, Node]], Node]
    input_specs: Dict[str, int]
    reference_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    distribution_names: Sequence[str]
    source_file: str
    notes: str = ""
    machine: str = "relu"
    build_graphs_per_distribution: Dict[str, Callable[[Dict[str, Node]], Node]] = field(
        default_factory=dict
    )
    reference_fns_per_distribution: Dict[
        str, Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    ] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input distributions
# ---------------------------------------------------------------------------


def _uniform_1d(
    name: str,
    description: str,
    lo: float,
    hi: float,
    input_name: str = "x",
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Uniform samples in [lo, hi] for a single 1D input, plus a dense grid."""
    gen = torch.Generator().manual_seed(seed)
    random_part = torch.rand((n_samples - 256, 1), generator=gen) * (hi - lo) + lo
    grid_part = torch.linspace(lo, hi, 256).unsqueeze(1)
    data = torch.cat([random_part, grid_part], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_name: data},
        n_samples=n_samples,
    )


def _signed_log_uniform_with_v_bottom(
    name: str,
    description: str,
    min_abs: float,
    max_abs: float,
    input_name: str = "x",
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Signed log-uniform |x|∈[min_abs, max_abs] plus V-bottom grid.

    Used by ``log_abs``: |x| is log-uniform on [min_abs, max_abs] with
    sign uniform ±; structured grid concentrates samples in
    ``[-2·min_abs, 2·min_abs]`` to stress the V-bottom transition zone.
    """
    import math as _math

    gen = torch.Generator().manual_seed(seed)
    n_v_grid = 64  # structured grid around V-bottom
    n_random = n_samples - n_v_grid

    log_lo = _math.log(min_abs)
    log_hi = _math.log(max_abs)
    u = torch.rand(n_random, generator=gen)
    abs_x = torch.exp(log_lo + u * (log_hi - log_lo))
    signs = (
        torch.randint(0, 2, (n_random,), generator=gen).to(torch.float32) * 2.0 - 1.0
    )
    random_part = (signs * abs_x).unsqueeze(1)

    # V-bottom grid: linear from -2·min_abs to +2·min_abs.
    v_grid = torch.linspace(-2.0 * min_abs, 2.0 * min_abs, n_v_grid).unsqueeze(1)

    data = torch.cat([random_part, v_grid], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_name: data},
        n_samples=n_samples,
    )


def _uniform_2d(
    name: str,
    description: str,
    lo1: float,
    hi1: float,
    lo2: float,
    hi2: float,
    input_names: tuple = ("a", "b"),
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Uniform samples over the 2D box, plus a 64×64 grid."""
    gen = torch.Generator().manual_seed(seed)
    n_grid = 64 * 64
    n_rand = n_samples - n_grid
    rand1 = torch.rand((n_rand, 1), generator=gen) * (hi1 - lo1) + lo1
    rand2 = torch.rand((n_rand, 1), generator=gen) * (hi2 - lo2) + lo2
    g1 = torch.linspace(lo1, hi1, 64)
    g2 = torch.linspace(lo2, hi2, 64)
    gx, gy = torch.meshgrid(g1, g2, indexing="ij")
    grid1 = gx.reshape(-1, 1)
    grid2 = gy.reshape(-1, 1)
    data1 = torch.cat([rand1, grid1], dim=0)
    data2 = torch.cat([rand2, grid2], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_names[0]: data1, input_names[1]: data2},
        n_samples=n_samples,
    )


def _integer_1d(
    name: str,
    description: str,
    lo: int,
    hi: int,
    input_name: str = "x",
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Integer samples uniformly drawn from [lo, hi], padded with each value."""
    gen = torch.Generator().manual_seed(seed)
    n_values = hi - lo + 1
    random_part = torch.randint(
        lo, hi + 1, (n_samples - n_values, 1), generator=gen
    ).to(torch.float32)
    grid_part = torch.arange(lo, hi + 1, dtype=torch.float32).unsqueeze(1)
    data = torch.cat([random_part, grid_part], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_name: data},
        n_samples=n_samples,
    )


def _integer_with_plateau_jitter_1d(
    name: str,
    description: str,
    lo: int,
    hi: int,
    input_name: str = "x",
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Integer samples in [lo, hi] with uniform jitter inside the plateau
    contract band ±1/(2·step_sharpness) — the in-contract inputs of a
    collapse-synthesized staircase (integer ± tolerated noise), plus the
    exact-integer grid so every plateau center is covered."""
    gen = torch.Generator().manual_seed(seed)
    n_grid = min(hi - lo + 1, 1024)
    n_random = n_samples - n_grid
    ints = torch.randint(lo, hi + 1, (n_random, 1), generator=gen).to(torch.float32)
    w = 1.0 / (2.0 * step_sharpness)
    jitter = (torch.rand((n_random, 1), generator=gen) * 2.0 - 1.0) * w
    grid_part = torch.linspace(float(lo), float(hi), n_grid).round().unsqueeze(1)
    data = torch.cat([ints + jitter, grid_part], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_name: data},
        n_samples=n_samples,
    )


# --- Collapse-synthesized staircase configuration -------------------------
# The exact shape the univariate collapse pass emits
# (torchwright/compiler/collapse.py, docs/univariate_collapse_plan.md):
# step pairs at half-integers, 1/step_sharpness apart, input_scale =
# step_sharpness, one plateau per integer, tabulated values.  Plateau
# values are a deterministic pseudo-random integer table in [-50, 50]
# whose every adjacent pair differs (37 and 101 are coprime), so the
# staircase emits the full 2·(n_plateaus − 1) lanes — the pass's
# worst-case lane shape at a given plateau count.


def _staircase_table(n_plateaus: int) -> torch.Tensor:
    ks = torch.arange(n_plateaus, dtype=torch.int64)
    return ((ks * 37) % 101 - 50).to(torch.float32)


def _staircase_build(pl_fn, n_lanes: int):
    n_plateaus = n_lanes // 2 + 1
    values = _staircase_table(n_plateaus).tolist()
    h = 1.0 / (2.0 * step_sharpness)
    breakpoints: list = []
    for k in range(n_plateaus - 1):
        breakpoints.extend((k + 0.5 - h, k + 0.5 + h))

    def fn(x: float) -> float:
        return values[min(max(round(x), 0), n_plateaus - 1)]

    def build(nodes):
        return pl_fn(
            nodes["x"],
            breakpoints=breakpoints,
            fn=fn,
            d_max=n_lanes,
            input_scale=step_sharpness,
        )

    return build


def _staircase_reference(n_lanes: int):
    n_plateaus = n_lanes // 2 + 1
    table = _staircase_table(n_plateaus)

    def ref(inputs):
        idx = inputs["x"].round().long().clamp(0, n_plateaus - 1)
        return table[idx]

    return ref


def _integer_pair(
    name: str,
    description: str,
    lo: int,
    hi: int,
    input_names: tuple = ("a", "b"),
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Two independent integer streams in [lo, hi]."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(lo, hi + 1, (n_samples, 1), generator=gen).to(torch.float32)
    b = torch.randint(lo, hi + 1, (n_samples, 1), generator=gen).to(torch.float32)
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_names[0]: a, input_names[1]: b},
        n_samples=n_samples,
    )


def _near_threshold_1d(
    name: str,
    description: str,
    thresh: float,
    half_width: float,
    input_name: str = "x",
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Samples concentrated in ``[thresh - half_width, thresh + half_width]``."""
    gen = torch.Generator().manual_seed(seed)
    lo = thresh - half_width
    hi = thresh + half_width
    data = torch.rand((n_samples, 1), generator=gen) * (hi - lo) + lo
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_name: data},
        n_samples=n_samples,
    )


def _below_multiples_1d(
    name: str,
    description: str,
    multiple: float,
    m_lo: int,
    m_hi: int,
    off_lo: float,
    off_hi: float,
    input_name: str = "x",
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Samples at ``m·multiple − offset`` with the offset log-uniform in
    ``[off_lo, off_hi]`` — stresses the sliver just below staircase
    boundaries (``radix_floor_int``'s divisor boundaries, ramp zones)."""
    gen = torch.Generator().manual_seed(seed)
    m = torch.randint(m_lo, m_hi + 1, (n_samples, 1), generator=gen).to(torch.float32)
    u = torch.rand((n_samples, 1), generator=gen)
    off = off_lo * (off_hi / off_lo) ** u
    return InputDistribution(
        name=name,
        description=description,
        inputs={input_name: m * multiple - off},
        n_samples=n_samples,
    )


def _bool_single_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    gen = torch.Generator().manual_seed(seed)
    signs = (
        torch.randint(0, 2, (n_samples, 1), generator=gen).to(torch.float32) * 2.0 - 1.0
    )
    return InputDistribution(
        name=name,
        description=description,
        inputs={"x": signs},
        n_samples=n_samples,
    )


def _bool_triple_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    gen = torch.Generator().manual_seed(seed)
    signs = (
        torch.randint(0, 2, (n_samples, 3), generator=gen).to(torch.float32) * 2.0 - 1.0
    )
    return InputDistribution(
        name=name,
        description=description,
        inputs={
            "a": signs[:, 0:1],
            "b": signs[:, 1:2],
            "c": signs[:, 2:3],
        },
        n_samples=n_samples,
    )


def _equals_vector_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    gen = torch.Generator().manual_seed(seed)
    target = torch.tensor([1.0, 2.0, 3.0])
    n_match = n_samples // 2
    matches = target.unsqueeze(0).expand(n_match, -1)
    # Non-match samples stay in [-1, 1]^3 so the inner product `target @ x`
    # cannot exceed `target @ target = 14`. `equals_vector`'s ReLU formulation
    # is only defined for inputs where the dot product sits *at or below* the
    # target's self-dot; over-shooting causes the approximate sign value-type
    # assert to fire (outputs exceed +1). Tight-to-target inputs are still the
    # interesting test case because they exercise the transition ball.
    rand = torch.rand((n_samples - n_match, 3), generator=gen) * 2.0 - 1.0
    data = torch.cat([matches, rand], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={"x": data},
        n_samples=n_samples,
    )


def _map_to_table_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    """Half exact key matches (cycling three keys), half far-off vectors in
    [-1, 1]^3 (no-match → default; dot products stay below every key's
    self-dot, per the op contract — see _equals_vector_distribution)."""
    gen = torch.Generator().manual_seed(seed)
    keys = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [-1.0, -1.0, 2.0]])
    n_match = n_samples // 2
    matches = keys[torch.arange(n_match) % 3]
    rand = torch.rand((n_samples - n_match, 3), generator=gen) * 2.0 - 1.0
    data = torch.cat([matches, rand], dim=0)
    return InputDistribution(
        name=name,
        description=description,
        inputs={"x": data},
        n_samples=n_samples,
    )


def _onehot_lookup_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    """Two-block one-hot inputs (digit 3 ⊕ carry 2), uniformly over all six
    combinations — half hit the three-row table, half miss → default."""
    gen = torch.Generator().manual_seed(seed)
    d = torch.randint(0, 3, (n_samples,), generator=gen)
    c = torch.randint(0, 2, (n_samples,), generator=gen)
    data = torch.zeros(n_samples, 5)
    data[torch.arange(n_samples), d] = 1.0
    data[torch.arange(n_samples), 3 + c] = 1.0
    return InputDistribution(
        name=name,
        description=description,
        inputs={"x": data},
        n_samples=n_samples,
    )


def _cond_gate_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    gen = torch.Generator().manual_seed(seed)
    cond = (
        torch.randint(0, 2, (n_samples, 1), generator=gen).to(torch.float32) * 2.0 - 1.0
    )
    value = torch.rand((n_samples, 1), generator=gen) * 10.0 - 5.0
    return InputDistribution(
        name=name,
        description=description,
        inputs={"cond": cond, "inp": value},
        n_samples=n_samples,
    )


def _select_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    """±1 cond plus two independent uniform branch values in [-5, 5]."""
    gen = torch.Generator().manual_seed(seed)
    cond = (
        torch.randint(0, 2, (n_samples, 1), generator=gen).to(torch.float32) * 2.0 - 1.0
    )
    a = torch.rand((n_samples, 1), generator=gen) * 10.0 - 5.0
    b = torch.rand((n_samples, 1), generator=gen) * 10.0 - 5.0
    return InputDistribution(
        name=name,
        description=description,
        inputs={"cond": cond, "a": a, "b": b},
        n_samples=n_samples,
    )


def _in_range_distribution(
    name: str, description: str, n_slots: int = 8, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    """Integer (lower, upper) bound pairs over [0, n_slots], lower ≤ upper
    (the op contract — inverted bounds are out of contract and read -3)."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randint(0, n_slots + 1, (n_samples, 1), generator=gen).to(torch.float32)
    b = torch.randint(0, n_slots + 1, (n_samples, 1), generator=gen).to(torch.float32)
    lower = torch.minimum(a, b)
    upper = torch.maximum(a, b)
    return InputDistribution(
        name=name,
        description=description,
        inputs={"lower": lower, "upper": upper},
        n_samples=n_samples,
    )


def _broadcast_select_distribution(
    name: str, description: str, n_slots: int = 4, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    """±1 masks over n_slots plus per-slot true/false values in [-5, 5]."""
    gen = torch.Generator().manual_seed(seed)
    masks = (
        torch.randint(0, 2, (n_samples, n_slots), generator=gen).to(torch.float32) * 2.0
        - 1.0
    )
    t = torch.rand((n_samples, n_slots), generator=gen) * 10.0 - 5.0
    f = torch.rand((n_samples, n_slots), generator=gen) * 10.0 - 5.0
    return InputDistribution(
        name=name,
        description=description,
        inputs={"masks": masks, "t": t, "f": f},
        n_samples=n_samples,
    )


def _dynamic_extract_distribution(
    name: str,
    description: str,
    n_entries: int = 4,
    d_fill: int = 2,
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Runtime tables in [-5, 5] plus integer indices in [0, n_entries-1]."""
    gen = torch.Generator().manual_seed(seed)
    table = torch.rand((n_samples, n_entries * d_fill), generator=gen) * 10.0 - 5.0
    idx = torch.randint(0, n_entries, (n_samples, 1), generator=gen).to(torch.float32)
    return InputDistribution(
        name=name,
        description=description,
        inputs={"table": table, "idx": idx},
        n_samples=n_samples,
    )


def _table_lookup_2d_distribution(
    name: str, description: str, n_samples: int = 4096, seed: int = 0
) -> InputDistribution:
    """Integer (i, j) index pairs over an 8×6 table, plus 25% out-of-range
    indices (the clamp path)."""
    gen = torch.Generator().manual_seed(seed)
    i = torch.randint(-2, 10, (n_samples, 1), generator=gen).to(torch.float32)
    j = torch.randint(-2, 8, (n_samples, 1), generator=gen).to(torch.float32)
    return InputDistribution(
        name=name,
        description=description,
        inputs={"i": i, "j": j},
        n_samples=n_samples,
    )


def _bin_index_distribution(
    name: str,
    description: str,
    x_min_val: float,
    x_max_val: float,
    probe_margin: float = 2.0,
    n_samples: int = 4096,
    seed: int = 0,
) -> InputDistribution:
    """Samples for ``linear_bin_index``: fixed ``x_min``/``x_max``, varying ``x``."""
    gen = torch.Generator().manual_seed(seed)
    lo = x_min_val - probe_margin
    hi = x_max_val + probe_margin
    x = torch.rand((n_samples, 1), generator=gen) * (hi - lo) + lo
    x_min = torch.full((n_samples, 1), x_min_val)
    x_max = torch.full((n_samples, 1), x_max_val)
    return InputDistribution(
        name=name,
        description=description,
        inputs={"x": x, "x_min": x_min, "x_max": x_max},
        n_samples=n_samples,
    )


# Non-uniform breakpoint tables for representative 2D op measurements. Kept
# small so the script stays fast; production configurations may use dozens
# more breakpoints but the measurement is representative.
_DIFF_BP = [-40.0, -20.0, -10.0, -5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0]
_TRIG_BP = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
_VEL_BP = [-0.7, -0.35, -0.1, 0.0, 0.1, 0.35, 0.7]
_ATAN_BP_CROSS = [-2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0]
_ATAN_BP_DOT = [0.5, 1.0, 2.0, 5.0, 10.0]


def _distributions() -> Dict[str, InputDistribution]:
    """All named input distributions, keyed by name."""
    return {
        "reciprocal_03_200": _uniform_1d(
            "reciprocal_03_200",
            "1/x over [0.3, 200.0], step=1.0.",
            0.3,
            200.0,
        ),
        # NOTE: `reciprocal_01_50_step01` is defined here for future use but
        # is currently not referenced by any TargetOp — see the NOTE on the
        # `reciprocal` TargetOp for the op-math precision mismatch that
        # blocks it, and the "Known gap: reciprocal at step=0.1" entry
        # in `docs/numerical_noise_findings.md`.
        "reciprocal_01_50_step01": _uniform_1d(
            "reciprocal_01_50_step01",
            "1/x over [0.1, 50.0], step=0.1.",
            0.1,
            50.0,
        ),
        "log_4decades_001_100": _uniform_1d(
            "log_4decades_001_100",
            "log(x) over [0.01, 100] — 4 decades. With per-decade "
            "sectioning (default), each section's pre-cancellation "
            "magnitude is bounded by `section_factor=10`, so float32 "
            "ULP noise is ~1.2e-6 per section. The dominant residual "
            "error comes from the multiply_2d blending grid in routing.",
            0.01,
            100.0,
        ),
        "log_6decades_wide": _uniform_1d(
            "log_6decades_wide",
            "log(x) over [0.01, 30000] — 6 decades. Stresses the "
            "sectioning path: a single-section piecewise log over this "
            "range fails outright (cancellation floor `(x_max/x_min) · "
            "2⁻²³ ≈ 0.36 absolute, observed worst ~1.0 abs). Sectioning "
            "drops the floor to ~5e-3 absolute, dominated by "
            "compare-cancellation noise at large `x` propagating "
            "through multiply_2d blending.",
            0.01,
            30000.0,
        ),
        "log_abs_3decades_pm100": _signed_log_uniform_with_v_bottom(
            "log_abs_3decades_pm100",
            "log_abs(x) for x ∈ [-100, 100] with min_abs=0.1, max_abs=100 "
            "(3 decades on |x|). Signed log-uniform |x| samples plus a "
            "V-bottom grid concentrated in [-0.2, 0.2] to stress the "
            "flat-zone transition at ±min_abs. Single-piecewise path "
            "(ratio 1000 < single-piecewise threshold).",
            0.1,
            100.0,
        ),
        "exp_pm5": _uniform_1d(
            "exp_pm5",
            "exp(x) over [-5, 5] — output spans [exp(-5), exp(5)] ≈ "
            "[0.0067, 148.4], the natural pairing for log over [0.01, 100]. "
            "Uniform 256-BP grid (`Δx ≈ 0.0392`); per-cell relative-error "
            "bound is `(Δx)²/8 ≈ 1.9e-4`.",
            -5.0,
            5.0,
        ),
        "parabola_0_10_step1": _uniform_1d(
            "parabola_0_10_step1",
            "f(x)=x² on [0, 10] with integer breakpoints — probes generic "
            "`piecewise_linear` precision between grid points.",
            0.0,
            10.0,
        ),
        "staircase_64_lanes": _integer_with_plateau_jitter_1d(
            "staircase_64_lanes",
            "Integers in [0, 32] ± the plateau contract band "
            "(±1/(2·step_sharpness)) — in-contract inputs of a 64-lane "
            "collapse-synthesized staircase (33 plateaus, every step "
            "value-changing).",
            0,
            32,
        ),
        "staircase_512_lanes": _integer_with_plateau_jitter_1d(
            "staircase_512_lanes",
            "Integers in [0, 256] ± the plateau contract band — a "
            "512-lane staircase (257 plateaus), the thermometer-Attn "
            "collapse scale.",
            0,
            256,
        ),
        "staircase_2048_lanes": _integer_with_plateau_jitter_1d(
            "staircase_2048_lanes",
            "Integers in [0, 1024] ± the plateau contract band — a "
            "2048-lane staircase (1025 plateaus), past the largest "
            "single FFN in production (1,024 lanes) and half the "
            "collapse pass's d_hidden/4 decline cap.",
            0,
            1024,
        ),
        "square_unsigned_0_10": _uniform_1d(
            "square_unsigned_0_10",
            "x² over [0, 10] with step=1.0 — exact at integers, worst error "
            "at half-integers.",
            0.0,
            10.0,
        ),
        "square_signed_pm10": _uniform_1d(
            "square_signed_pm10",
            "x² over [-10, 10] with step=1.0 — covers the negative half the "
            "unsigned variant cannot reach.",
            -10.0,
            10.0,
        ),
        "multiply_uniform_pm10": _uniform_2d(
            "multiply_uniform_pm10",
            "a·b over [-10, 10]² with step=1 — generic uniform product grid; "
            "analytical bound `step1*step2/4 = 0.25`.",
            -10.0,
            10.0,
            -10.0,
            10.0,
        ),
        "multiply_uniform_pm1000": _uniform_2d(
            "multiply_uniform_pm1000",
            "a·b over [-1000, 1000]² — magnitudes far beyond any ReLU-era "
            "grid; probes the range-free claim of the exact gated product.",
            -1000.0,
            1000.0,
            -1000.0,
            1000.0,
        ),
        "square_uniform_pm100": _uniform_1d(
            "square_uniform_pm100",
            "x² over [-100, 100] — signed and wide; probes the range-free "
            "claim of the exact gated square (no [0, max_value] contract).",
            -100.0,
            100.0,
        ),
        "diff_trig_nonuniform": _uniform_2d(
            "diff_trig_nonuniform",
            "Two-input product over DIFF_BP × TRIG_BP — DIFF in [-40, 40] "
            "(non-uniform), TRIG in [-1, 1] (non-uniform).",
            -40.0,
            40.0,
            -1.0,
            1.0,
        ),
        "diff_vel_nonuniform": _uniform_2d(
            "diff_vel_nonuniform",
            "Two-input product over DIFF_BP × VEL_BP — DIFF in [-40, 40] "
            "(non-uniform), VEL in [-0.7, 0.7] (non-uniform).",
            -40.0,
            40.0,
            -0.7,
            0.7,
        ),
        "signed_multiply_pm10": _uniform_2d(
            "signed_multiply_pm10",
            "a·b over [-10, 10]² via polarization identity, step=1.0 — "
            "analytical bound `step × (max_abs1 + max_abs2) / 4 = 5.0` "
            "absolute, much smaller relative.",
            -10.0,
            10.0,
            -10.0,
            10.0,
        ),
        "atan_cross_dot_nonuniform": _uniform_2d(
            "atan_cross_dot_nonuniform",
            "atan2(cross/dot) over [-2, 2] × [0.5, 10] on a non-uniform "
            "grid — rank-3 SVD via `low_rank_2d`.",
            -2.0,
            2.0,
            0.5,
            10.0,
        ),
        "compare_uniform_pm80": _uniform_1d(
            "compare_uniform_pm80",
            "Uniform samples on [-80, 80] — measures compare's ramp error "
            "well away from the threshold (expected ≈ 0).",
            -80.0,
            80.0,
        ),
        "compare_near_thresh_0": _near_threshold_1d(
            "compare_near_thresh_0",
            "Samples within ±0.3 of the threshold at 0.0 — stresses the 2-neuron "
            "ramp zone (width 1/step_sharpness).",
            0.0,
            0.3,
        ),
        "compare_near_thresh_05": _near_threshold_1d(
            "compare_near_thresh_05",
            "Samples within ±0.3 of threshold 0.5 — measures compare's "
            "ramp accuracy at a non-zero threshold.",
            0.5,
            0.3,
        ),
        "floor_uniform_neg5_10": _uniform_1d(
            "floor_uniform_neg5_10",
            "Uniform continuous samples on [-5, 10] — most land in flat zones; "
            "measures baseline `floor_int` accuracy.",
            -5.0,
            10.0,
        ),
        "floor_near_boundary_10": _near_threshold_1d(
            "floor_near_boundary_10",
            "Samples within ±1/step_sharpness of integer k=5 — stresses the "
            "ramp zone where `floor_int` interpolates.",
            5.0,
            0.2,
        ),
        "floor_sawtooth_uniform_0_1023": _uniform_1d(
            "floor_sawtooth_uniform_0_1023",
            "Uniform continuous samples on [0, 1023] for `floor_int` with "
            "`output_map = k % 256` — the sawtooth that folds a following "
            "piecewise-constant op into the floor. Most land in flat zones; "
            "those inside a ramp just below a multiple of 256 carry the "
            "-(H-1)=-255-amplified off-by-one, the worst-case delta.",
            0.0,
            1023.0,
        ),
        "radix_floor_native_uniform": _uniform_1d(
            "radix_floor_native_uniform",
            "Uniform continuous samples on [-1023, 1023] — the native "
            "texture-coordinate range used by the d8192 DOOM graph "
            "(sharpness=10^4).",
            -1023.0,
            1023.0,
        ),
        "radix_floor_divisor_sliver": _below_multiples_1d(
            "radix_floor_divisor_sliver",
            "Samples just below multiples of the divisor D=64 (offsets "
            "log-uniform in [1.2e-4, 6.3e-3]): inside the HI floor's ramp "
            "but outside the LO floor's — the D-amplification hazard the "
            "integer snap neutralizes. Expected ≈ exact.",
            64.0,
            -15,
            15,
            1.2e-4,
            6.3e-3,
        ),
        "radix_floor_lo_ramp": _below_multiples_1d(
            "radix_floor_lo_ramp",
            "Samples within 1/sharpness below random integers in "
            "[-1023, 1023] — inside the LO ramp, the same tolerated "
            "interpolation window as flat `floor_int` (error up to 1 "
            "step, never D-amplified).",
            1.0,
            -1022,
            1023,
            1e-6,
            9.9e-5,
        ),
        "thermometer_integers_0_100_by10": _integer_1d(
            "thermometer_integers_0_100_by10",
            "Integer inputs on [0, 100] with divisor=10 — `thermometer_floor_div` "
            "is defined only for integer inputs.",
            0,
            100,
        ),
        "multiply_integers_0_10": _integer_pair(
            "multiply_integers_0_10",
            "Integer pairs in [0, 10] — `multiply_integers` is exact on integer "
            "inputs by construction; serves as a precision-parity check.",
            0,
            10,
        ),
        "linear_bin_index_tex_col_16": _bin_index_distribution(
            "linear_bin_index_tex_col_16",
            "16-bin quantisation of x ∈ [0, 64] with x_min=0, x_max=64.",
            x_min_val=0.0,
            x_max_val=64.0,
        ),
        "abs_uniform_pm50": _uniform_1d(
            "abs_uniform_pm50",
            "Uniform signed inputs on [-50, 50]. `abs` is exact (ReLU(x) + "
            "ReLU(-x)) so any residual is float32 round-off only.",
            -50.0,
            50.0,
        ),
        "abs_near_zero_pm02": _uniform_1d(
            "abs_near_zero_pm02",
            "|x| ≤ 0.2 — the swish abs fillet zone, where the one-sided "
            "underestimate peaks (2·swish_dip/scale at |x| = 1.278/scale); "
            "the relu abs is exact here.",
            -0.2,
            0.2,
        ),
        "minmax_uniform_pm50": _uniform_2d(
            "minmax_uniform_pm50",
            "Two independent uniform streams on [-50, 50]². `min`/`max` are "
            "exact via the abs-trick; residual is float32 round-off only.",
            -50.0,
            50.0,
            -50.0,
            50.0,
        ),
        "ceil_integers_neg5_10": _integer_1d(
            "ceil_integers_neg5_10",
            "Integer inputs on [-5, 10]. `ceil_int(k) = k` for integer k "
            "(flat zones in the floor staircase reused under -floor(-x)).",
            -5,
            10,
        ),
        "mod_integers_0_100_by7": _integer_1d(
            "mod_integers_0_100_by7",
            "Integer inputs on [0, 100] with divisor=7. `mod_const` is "
            "x − divisor × `thermometer_floor_div(x)`, exact on integers.",
            0,
            100,
        ),
        "clamp_flat_zone_pm20": _uniform_1d(
            "clamp_flat_zone_pm20",
            "Uniform samples on [-8, 8] with clamp bounds [-10, 10] — every "
            "sample is inside the flat identity zone. Residual measures "
            "float32 round-off only.",
            -8.0,
            8.0,
        ),
        "bool_triple_signed": _bool_triple_distribution(
            "bool_triple_signed",
            "Three independent ±1 boolean streams — exercises all 8 truth-"
            "table rows of `bool_any_true` and `bool_all_true`.",
        ),
        "bool_single_signed": _bool_single_distribution(
            "bool_single_signed",
            "Random ±1 inputs (discrete, no ramp-zone samples) — `bool_not` "
            "is only defined on clean boolean inputs.",
        ),
        "equals_vector_match_and_off": _equals_vector_distribution(
            "equals_vector_match_and_off",
            "Half exact-match against target vector [1, 2, 3], half random "
            "vectors in [-3, 3]³ — measures the ReLU-based equality test's "
            "margin of separation between matches and non-matches.",
        ),
        "cond_gate_signed_bool_times_value": _cond_gate_distribution(
            "cond_gate_signed_bool_times_value",
            "±1 conditions paired with scalar values in [-5, 5]. Expected "
            "output: `value` when cond=+1, `0` when cond=-1.",
        ),
        "in_range_integer_bounds_8": _in_range_distribution(
            "in_range_integer_bounds_8",
            "Integer (lower, upper) pairs over [0, 8], 8 slots — integer "
            "bounds keep every hinge ≥ 4 units from its bend (bit-exact "
            "modulo the folded-/scale ulp class).",
        ),
        "broadcast_select_pm1_masks_4x1": _broadcast_select_distribution(
            "broadcast_select_pm1_masks_4x1",
            "±1 masks over 4 slots with per-slot true/false values in "
            "[-5, 5] (d_fill=1). Losing branches exactly zero at clean "
            "masks; winners ~1 ulp relative.",
        ),
        "dynamic_extract_int_idx_4x2": _dynamic_extract_distribution(
            "dynamic_extract_int_idx_4x2",
            "Runtime 4×2 tables in [-5, 5] with integer indices — the "
            "in_range → broadcast_select → sum composition on clean "
            "integer inputs.",
        ),
        "map_to_table_match_and_off": _map_to_table_distribution(
            "map_to_table_match_and_off",
            "Exact key matches (three 3-wide keys) and far-off vectors in "
            "[-1, 1]^3 — matches read their value to ~1 ulp, misses return "
            "the default bit-exactly (indicators underflow).",
        ),
        "onehot_lookup_two_block": _onehot_lookup_distribution(
            "onehot_lookup_two_block",
            "digit(3) ⊕ carry(2) one-hot pairs over all six combinations; "
            "three are table rows, three miss → default.",
        ),
        "table_lookup_2d_int_8x6": _table_lookup_2d_distribution(
            "table_lookup_2d_int_8x6",
            "Integer (i, j) over an 8×6 table with 25% out-of-range "
            "(clamped) indices — every indicator saturated, the value "
            "path is the delta telescoping's fp32 accumulation.",
        ),
        "select_signed_bool_two_values": _select_distribution(
            "select_signed_bool_two_values",
            "±1 conditions paired with two independent branch values in "
            "[-5, 5]. Expected output: `a` when cond=+1, `b` when cond=-1.",
        ),
    }


# ---------------------------------------------------------------------------
# Target ops
# ---------------------------------------------------------------------------


def _floor_ref(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(torch.floor(x), lo, hi)


def _floor_sawtooth_ref(x: torch.Tensor, lo: float, hi: float, H: int) -> torch.Tensor:
    """floor_int(x, output_map=k % H): a piecewise-constant op folded in."""
    return torch.remainder(torch.clamp(torch.floor(x), lo, hi), float(H))


def _bin_index_ref(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, n_bins: int
) -> torch.Tensor:
    r = x_max - x_min
    v = (x - x_min) * n_bins / r
    return torch.clamp(torch.floor(v), 0, n_bins - 1)


def _oh_key(d: int, c: int) -> torch.Tensor:
    key = torch.zeros(5)
    key[d] = 1.0
    key[3 + c] = 1.0
    return key


def _map_to_table_ref(x: torch.Tensor) -> torch.Tensor:
    keys = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [-1.0, -1.0, 2.0]])
    values = torch.tensor([[10.0, -1.0], [20.0, -2.0], [30.0, -3.0]])
    default = torch.tensor([5.0, 0.5])
    out = default.unsqueeze(0).repeat(x.shape[0], 1)
    for k, v in zip(keys, values):
        hit = (x == k).all(dim=-1)
        out[hit] = v
    return out


def _onehot_lookup_ref(x: torch.Tensor) -> torch.Tensor:
    table = {(0, 0): 100.0, (1, 0): 200.0, (2, 1): 300.0}
    d = x[:, :3].argmax(dim=-1)
    c = x[:, 3:].argmax(dim=-1)
    out = torch.full((x.shape[0], 1), -7.0)
    for (dd, cc), v in table.items():
        out[(d == dd) & (c == cc)] = v
    return out


_TL2D_TABLE = (torch.arange(48, dtype=torch.float32).reshape(8, 6) * 7.0) % 23.0 - 11.0


def _table_lookup_2d_ref(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    i = inputs["i"].flatten().long().clamp(0, 7)
    j = inputs["j"].flatten().long().clamp(0, 5)
    return _TL2D_TABLE[i, j].unsqueeze(1)


def _target_ops() -> List[TargetOp]:
    _ARITH = "torchwright.ops.relu.arithmetic_ops"
    _ARITH_FILE = "torchwright/ops/relu/arithmetic_ops.py"
    _SWIGLU_ARITH = "torchwright.ops.swiglu.arithmetic_ops"
    _SWIGLU_ARITH_FILE = "torchwright/ops/swiglu/arithmetic_ops.py"
    _SWIGLU_LOGIC = "torchwright.ops.swiglu.logic_ops"
    _SWIGLU_LOGIC_FILE = "torchwright/ops/swiglu/logic_ops.py"
    _SWIGLU_SELECT = "torchwright.ops.swiglu.map_select"
    _SWIGLU_SELECT_FILE = "torchwright/ops/swiglu/map_select.py"

    return [
        TargetOp(
            name="reciprocal",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: reciprocal(
                nodes["x"], min_value=0.3, max_value=200.0, step=1.0
            ),
            reference_fn=lambda inputs: 1.0 / inputs["x"],
            # NOTE: `reciprocal_01_50_step01` is temporarily omitted. With
            # `step=0.1`, it produces ~500 breakpoints; float32 accumulation
            # across that many ReLU terms exceeds `piecewise_linear`'s declared
            # `atol=1e-3` at the tail. This is an op-math precision-claim
            # mismatch, not a measurement issue — see the "Known gap: reciprocal
            # at step=0.1" entry in `docs/numerical_noise_findings.md`. Once
            # the op-math fix lands, restore `reciprocal_01_50_step01` here with
            # a `build_graphs_per_distribution` entry using
            # `reciprocal(nodes["x"], min_value=0.1, max_value=50.0, step=0.1)`.
            distribution_names=("reciprocal_03_200",),
            notes=(
                "Geometric breakpoint spacing keeps relative interpolation error "
                "roughly constant across the range. Measured against a "
                "reciprocal graph with "
                "`(min_value, max_value, step) = (0.3, 200.0, 1.0)`."
            ),
        ),
        TargetOp(
            name="piecewise_linear",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: piecewise_linear(
                nodes["x"],
                breakpoints=[float(i) for i in range(11)],
                fn=lambda v: v * v,
            ),
            reference_fn=lambda inputs: inputs["x"] ** 2,
            distribution_names=(
                "parabola_0_10_step1",
                "staircase_64_lanes",
                "staircase_512_lanes",
                "staircase_2048_lanes",
            ),
            build_graphs_per_distribution={
                "staircase_64_lanes": _staircase_build(piecewise_linear, 64),
                "staircase_512_lanes": _staircase_build(piecewise_linear, 512),
                "staircase_2048_lanes": _staircase_build(piecewise_linear, 2048),
            },
            reference_fns_per_distribution={
                "staircase_64_lanes": _staircase_reference(64),
                "staircase_512_lanes": _staircase_reference(512),
                "staircase_2048_lanes": _staircase_reference(2048),
            },
            notes=(
                "Foundation 1D piecewise-linear primitive. The parabola "
                "distribution measures the interpolation configuration (11 "
                "integer breakpoints, x²): error is zero at breakpoints and "
                "bounded by per-segment interpolation error. The staircase "
                "distributions measure the univariate-collapse "
                "configuration: step "
                "pairs at half-integers, input_scale = step_sharpness, "
                "in-contract inputs — the axis that grows with lane count "
                "is fp32 accumulation across the lane sum."
            ),
        ),
        TargetOp(
            name="square",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: square(nodes["x"], max_value=10.0, step=1.0),
            reference_fn=lambda inputs: inputs["x"] ** 2,
            distribution_names=("square_unsigned_0_10",),
            notes=(
                "Always an underestimate of x² between grid points. Max "
                "segment error is `(step/2)²` at half-integers."
            ),
        ),
        TargetOp(
            name="multiply_2d",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: multiply_2d(
                nodes["a"],
                nodes["b"],
                max_abs1=10.0,
                max_abs2=10.0,
                step1=1.0,
                step2=1.0,
            ),
            reference_fn=lambda inputs: inputs["a"] * inputs["b"],
            distribution_names=("multiply_uniform_pm10",),
            notes=(
                "Single-sublayer product built analytically via the "
                "quarter-square identity `a*b = ((a+b)^2 - (a-b)^2)/4` "
                "(O(n) neurons, no least-squares solve). Worst-cell absolute "
                "bound is `step1*step2/4 = 0.25` for this configuration."
            ),
        ),
        TargetOp(
            name="compare",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: compare(
                nodes["x"], thresh=0.0, true_level=1.0, false_level=-1.0
            ),
            reference_fn=lambda inputs: torch.where(
                inputs["x"] > 0.0,
                torch.ones_like(inputs["x"]),
                -torch.ones_like(inputs["x"]),
            ),
            distribution_names=("compare_uniform_pm80", "compare_near_thresh_0"),
            notes=(
                "ReLU-pair ramp. Inputs within `1/step_sharpness` of the "
                "threshold land inside the ramp zone and yield an "
                "interpolated value; inputs further away match the step "
                "function exactly. The `compare_near_thresh_*` distributions "
                "deliberately stress the ramp — the reference is a discrete "
                "step function, so samples inside the ramp legitimately "
                "report large error."
            ),
        ),
        TargetOp(
            name="floor_int",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: floor_int(nodes["x"], min_value=-5, max_value=10),
            reference_fn=lambda inputs: _floor_ref(inputs["x"], -5.0, 10.0),
            distribution_names=("floor_uniform_neg5_10", "floor_near_boundary_10"),
            notes=(
                "Staircase with ramp width `1/step_sharpness` directly "
                "below each integer. Inputs near integer boundaries land in "
                "the ramp zone and produce interpolated values; callers with "
                "integer-ish inputs should use `thermometer_floor_div` "
                "instead, whose ramps sit at half-integer thresholds."
            ),
        ),
        TargetOp(
            name="thermometer_floor_div",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: thermometer_floor_div(
                nodes["x"], divisor=10, max_value=100
            ),
            reference_fn=lambda inputs: torch.floor(inputs["x"] / 10.0),
            distribution_names=("thermometer_integers_0_100_by10",),
            notes=(
                "Integer-input-only staircase. Ramps sit at `k*divisor - 0.5`, "
                "giving clean separation for integer inputs. Feeding continuous "
                "floats produces ramp-zone interpolation artefacts — those "
                "belong to `floor_int`, not this op."
            ),
        ),
        TargetOp(
            name="multiply_integers",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: multiply_integers(
                nodes["a"], nodes["b"], max_value=10
            ),
            reference_fn=lambda inputs: inputs["a"] * inputs["b"],
            distribution_names=("multiply_integers_0_10",),
            notes=(
                "Polarization + `square` on non-negative integer inputs. "
                "Exact on integer inputs by construction (the polarization "
                "identity is exact; `square` is exact at integer grid points)."
            ),
        ),
        # -------------------------------------------------------------------
        # Priority B — exact-op negative controls. Expected error ≲ 1e-6.
        # -------------------------------------------------------------------
        TargetOp(
            name="abs",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: abs_op(nodes["x"]),
            reference_fn=lambda inputs: inputs["x"].abs(),
            distribution_names=("abs_uniform_pm50",),
            notes=(
                "Exact decomposition: `ReLU(x) + ReLU(-x)`. Any non-zero "
                "measurement here would indicate a compiler or float32 bug."
            ),
        ),
        TargetOp(
            name="min",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: min_op(nodes["a"], nodes["b"]),
            reference_fn=lambda inputs: torch.minimum(inputs["a"], inputs["b"]),
            distribution_names=("minmax_uniform_pm50",),
            notes="Exact via `(a + b - |a - b|) / 2`.",
        ),
        TargetOp(
            name="ceil_int",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: ceil_int(nodes["x"], min_value=-5, max_value=10),
            reference_fn=lambda inputs: torch.clamp(
                torch.ceil(inputs["x"]), -5.0, 10.0
            ),
            distribution_names=("ceil_integers_neg5_10",),
            notes=(
                "Reduction to `-floor_int(-x)`. On integer inputs every value "
                "lands in a flat zone of the floor staircase, so the op is "
                "exact here even though it inherits approximate semantics for "
                "continuous inputs."
            ),
        ),
        TargetOp(
            name="mod_const",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: mod_const(nodes["x"], divisor=7, max_value=100),
            reference_fn=lambda inputs: inputs["x"]
            - 7.0 * torch.floor(inputs["x"] / 7.0),
            distribution_names=("mod_integers_0_100_by7",),
            notes=(
                "Identity `x % d = x - d · thermometer_floor_div(x, d)`. "
                "Exact on integer inputs (the input class this op is declared "
                "for)."
            ),
        ),
        TargetOp(
            name="clamp",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: clamp(nodes["x"], lo=-10.0, hi=10.0),
            reference_fn=lambda inputs: torch.clamp(inputs["x"], -10.0, 10.0),
            distribution_names=("clamp_flat_zone_pm20",),
            notes=(
                "Piecewise-linear identity with 4 breakpoints at `[lo, "
                "lo + eps, hi - eps, hi]`. In the interior flat zone (this "
                "distribution) the op is a pure identity; boundary samples "
                "would fall inside the `eps`-wide ramp and lose precision "
                "but are out of scope for this negative-control distribution."
            ),
        ),
        # -------------------------------------------------------------------
        # Priority C — logic ops (most derived from compare).
        # -------------------------------------------------------------------
        TargetOp(
            name="bool_not",
            module="torchwright.ops.relu.logic_ops",
            source_file="torchwright/ops/relu/logic_ops.py",
            input_specs={"x": 1},
            build_graph=lambda nodes: bool_not(nodes["x"]),
            reference_fn=lambda inputs: -inputs["x"],
            distribution_names=("bool_single_signed",),
            notes=(
                "Single `compare` with inverted levels. Exact on clean ±1 "
                "inputs; inherits `compare`'s ramp-zone error for inputs near 0."
            ),
        ),
        TargetOp(
            name="bool_any_true",
            module="torchwright.ops.relu.logic_ops",
            source_file="torchwright/ops/relu/logic_ops.py",
            input_specs={"a": 1, "b": 1, "c": 1},
            build_graph=lambda nodes: bool_any_true(
                [nodes["a"], nodes["b"], nodes["c"]]
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["a"] > 0) | (inputs["b"] > 0) | (inputs["c"] > 0),
                torch.ones_like(inputs["a"]),
                -torch.ones_like(inputs["a"]),
            ),
            distribution_names=("bool_triple_signed",),
            notes=(
                "Two stacked `compare` layers. On clean ±1 inputs the "
                "intermediate sum lands well outside every ramp zone, so "
                "the composition is exact."
            ),
        ),
        TargetOp(
            name="bool_all_true",
            module="torchwright.ops.relu.logic_ops",
            source_file="torchwright/ops/relu/logic_ops.py",
            input_specs={"a": 1, "b": 1, "c": 1},
            build_graph=lambda nodes: bool_all_true(
                [nodes["a"], nodes["b"], nodes["c"]]
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["a"] > 0) & (inputs["b"] > 0) & (inputs["c"] > 0),
                torch.ones_like(inputs["a"]),
                -torch.ones_like(inputs["a"]),
            ),
            distribution_names=("bool_triple_signed",),
            notes=(
                "Single `compare` at threshold `N-1` over the sum of N ±1 "
                "inputs. Sum is N only when all are +1; otherwise ≤ N-2 — "
                "threshold cleanly separates the two, so exact on ±1 inputs."
            ),
        ),
        TargetOp(
            name="equals_vector",
            module="torchwright.ops.relu.logic_ops",
            source_file="torchwright/ops/relu/logic_ops.py",
            input_specs={"x": 3},
            build_graph=lambda nodes: equals_vector(
                nodes["x"], torch.tensor([1.0, 2.0, 3.0])
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["x"] == torch.tensor([1.0, 2.0, 3.0])).all(
                    dim=-1, keepdim=True
                ),
                torch.ones_like(inputs["x"][:, :1]),
                -torch.ones_like(inputs["x"][:, :1]),
            ),
            distribution_names=("equals_vector_match_and_off",),
            notes=(
                "Single-ReLU approximate equality test. Inside the "
                "`1/embedding_step_sharpness`-wide transition ball around "
                "the target the output interpolates between -1 and +1."
            ),
        ),
        TargetOp(
            name="cond_gate",
            module="torchwright.ops.relu.logic_ops",
            source_file="torchwright/ops/relu/logic_ops.py",
            input_specs={"cond": 1, "inp": 1},
            build_graph=lambda nodes: cond_gate(nodes["cond"], nodes["inp"]),
            reference_fn=lambda inputs: torch.where(
                inputs["cond"] > 0,
                inputs["inp"],
                torch.zeros_like(inputs["inp"]),
            ),
            distribution_names=("cond_gate_signed_bool_times_value",),
            notes=(
                "`big_offset` ReLU gate. On clean ±1 conditions the output "
                "is exactly `inp` (cond=+1) or `0` (cond=-1) up to float32 "
                "round-off; noisy conditions blur the gate."
            ),
        ),
        # -------------------------------------------------------------------
        # swiglu machine (torchwright/ops/swiglu/) — entries land op by op
        # per docs/swiglu_step2_plan.md Phase B. Expected error class per
        # docs/ops_plain_english.md.
        # -------------------------------------------------------------------
        TargetOp(
            name="multiply",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: swiglu_ops.multiply(nodes["a"], nodes["b"]),
            reference_fn=lambda inputs: inputs["a"] * inputs["b"],
            distribution_names=("multiply_uniform_pm10", "multiply_uniform_pm1000"),
            notes=(
                "Exact ± gated-lane pair: `Swish(a)·b + Swish(-a)·(-b) = a·b` "
                "(σ(a) + σ(-a) = 1). Exact in real math at any magnitude — "
                "no grid, no range limit; measured error is fp32 rounding, "
                "relative to the actual product. Replaces the quarter-square "
                "`multiply_2d` and the `multiply_integers` chain."
            ),
        ),
        TargetOp(
            name="square",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.square(nodes["x"]),
            reference_fn=lambda inputs: inputs["x"] ** 2,
            distribution_names=("square_unsigned_0_10", "square_uniform_pm100"),
            notes=(
                "`multiply` with both operands the same input: "
                "`Swish(x)·x + Swish(-x)·(-x) = x²`, both terms `x²·σ(±x)` "
                "≥ 0. Exact in real math for all x — no [0, max_value] "
                "contract, no grid, and none of the piecewise version's "
                "near-zero relative-error blowup."
            ),
        ),
        TargetOp(
            name="compare",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.compare(
                nodes["x"], thresh=0.0, true_level=1.0, false_level=-1.0
            ),
            reference_fn=lambda inputs: torch.where(
                inputs["x"] > 0.0,
                torch.ones_like(inputs["x"]),
                -torch.ones_like(inputs["x"]),
            ),
            distribution_names=("compare_uniform_pm80", "compare_near_thresh_0"),
            notes=(
                "Sharpened-hinge ramp (`hinge(z) − hinge(z−1)`, "
                "`hinge(z) = Swish(scale·z)/scale`). Same ramp-zone contract "
                "as the ReLU form (in-ramp inputs interpolate; the "
                "near-thresh distribution deliberately stresses it). What's "
                "new vs relu: fillet dips within ~17/(scale·sharpness) of "
                "the two bends can overshoot either level by up to "
                "swish_dip/scale·|T−F| (0.0044 total span at scale=128); "
                "true-side far-field outputs carry fp32 product rounding at "
                "the lane-contribution magnitude — the same class as relu's "
                "far-field noise; the false side is exactly out_bias."
            ),
        ),
        TargetOp(
            name="bool_not",
            machine="swiglu",
            module=_SWIGLU_LOGIC,
            source_file=_SWIGLU_LOGIC_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.bool_not(nodes["x"]),
            reference_fn=lambda inputs: -inputs["x"],
            distribution_names=("bool_single_signed",),
            notes=(
                "Single swiglu `compare` with inverted levels — inherits "
                "compare's entry. On ±1 inputs the false side (input +1) is "
                "exact and the true side is ulp-class."
            ),
        ),
        TargetOp(
            name="bool_any_true",
            machine="swiglu",
            module=_SWIGLU_LOGIC,
            source_file=_SWIGLU_LOGIC_FILE,
            input_specs={"a": 1, "b": 1, "c": 1},
            build_graph=lambda nodes: swiglu_ops.bool_any_true(
                [nodes["a"], nodes["b"], nodes["c"]]
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["a"] > 0) | (inputs["b"] > 0) | (inputs["c"] > 0),
                torch.ones_like(inputs["a"]),
                -torch.ones_like(inputs["a"]),
            ),
            distribution_names=("bool_triple_signed",),
            notes=(
                "Two stacked swiglu `compare` layers. On clean ±1 inputs "
                "the intermediate sum lands well outside every ramp zone "
                "and fillet; the composition is exact to the stacked "
                "far-field ulp class (~1e-6)."
            ),
        ),
        TargetOp(
            name="bool_all_true",
            machine="swiglu",
            module=_SWIGLU_LOGIC,
            source_file=_SWIGLU_LOGIC_FILE,
            input_specs={"a": 1, "b": 1, "c": 1},
            build_graph=lambda nodes: swiglu_ops.bool_all_true(
                [nodes["a"], nodes["b"], nodes["c"]]
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["a"] > 0) & (inputs["b"] > 0) & (inputs["c"] > 0),
                torch.ones_like(inputs["a"]),
                -torch.ones_like(inputs["a"]),
            ),
            distribution_names=("bool_triple_signed",),
            notes=(
                "Single swiglu `compare` at threshold `N-1` over the sum of "
                "N ±1 inputs — sum is N only when all are +1, otherwise "
                "≤ N-2. Exact to the far-field ulp class on ±1 inputs."
            ),
        ),
        TargetOp(
            name="equals_vector",
            machine="swiglu",
            module=_SWIGLU_LOGIC,
            source_file=_SWIGLU_LOGIC_FILE,
            input_specs={"x": 3},
            build_graph=lambda nodes: swiglu_ops.equals_vector(
                nodes["x"], torch.tensor([1.0, 2.0, 3.0])
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["x"] == torch.tensor([1.0, 2.0, 3.0])).all(
                    dim=-1, keepdim=True
                ),
                torch.ones_like(inputs["x"][:, :1]),
                -torch.ones_like(inputs["x"][:, :1]),
            ),
            distribution_names=("equals_vector_match_and_off",),
            notes=(
                "One sharpened hinge on the dot-product margin. Matches are "
                "bit-exact +1; non-matches within ~17/scale past the "
                "1/speed margin land in the hinge dip and read as low as "
                "-1 - 2·swish_dip·speed/scale (-1.0044 at scale=128) — the "
                "value-range assert carries that low-side slack. Inside the "
                "margin ball the output interpolates, as today."
            ),
        ),
        TargetOp(
            name="cond_gate",
            machine="swiglu",
            module=_SWIGLU_LOGIC,
            source_file=_SWIGLU_LOGIC_FILE,
            input_specs={"cond": 1, "inp": 1},
            build_graph=lambda nodes: swiglu_ops.cond_gate(nodes["cond"], nodes["inp"]),
            reference_fn=lambda inputs: torch.where(
                inputs["cond"] > 0,
                inputs["inp"],
                torch.zeros_like(inputs["inp"]),
            ),
            distribution_names=("cond_gate_signed_bool_times_value",),
            notes=(
                "One gated lane per component: `Swish(scale·cond)·inp/scale`. "
                "At clean ±1 conds the off-path is exactly zero in fp32 "
                "(σ(−scale) computes as 0.0) and the on-path passes with ~1 "
                "ulp relative rounding. No offset M, no finite-range "
                "requirement — cond noise lands as δ·|actual value|."
            ),
        ),
        TargetOp(
            name="select",
            machine="swiglu",
            module=_SWIGLU_SELECT,
            source_file=_SWIGLU_SELECT_FILE,
            input_specs={"cond": 1, "a": 1, "b": 1},
            build_graph=lambda nodes: swiglu_ops.select(
                nodes["cond"], nodes["a"], nodes["b"]
            ),
            reference_fn=lambda inputs: torch.where(
                inputs["cond"] > 0, inputs["a"], inputs["b"]
            ),
            distribution_names=("select_signed_bool_two_values",),
            notes=(
                "Complementary gated pair: `Swish(scale·cond)·a/scale + "
                "Swish(−scale·cond)·b/scale`. The losing branch contributes "
                "exactly zero at clean conds; the winner passes with ~1 ulp "
                "relative rounding. (The relu-machine select was never "
                "measured.)"
            ),
        ),
        TargetOp(
            name="abs",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.abs(nodes["x"]),
            reference_fn=lambda inputs: inputs["x"].abs(),
            distribution_names=("abs_uniform_pm50", "abs_near_zero_pm02"),
            notes=(
                "`hinge(x) + hinge(-x)` = x·tanh(scale·x/2) — the rare op "
                "that regresses under swish (the relu identity is exact; "
                "|x| has a corner no smooth lane sum can express). Error is "
                "one-sided and bounded: output in [0, |x|], worst "
                "underestimate 2·swish_dip/scale (0.0044) at "
                "|x| = 1.278/scale; bit-exact for |x| ≳ 0.2 (the whole "
                "integer grid)."
            ),
        ),
        TargetOp(
            name="min",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: swiglu_ops.min(nodes["a"], nodes["b"]),
            reference_fn=lambda inputs: torch.minimum(inputs["a"], inputs["b"]),
            distribution_names=("minmax_uniform_pm50",),
            notes=(
                "`a − hinge(a−b)` plus a's sharpened bypass pair. One-sided "
                "over-estimate ≤ swish_dip/scale (0.0022), only when "
                "|a−b| ≲ 0.2; ties exact; far-apart operands carry the "
                "folded-/scale product-rounding ulp class."
            ),
        ),
        TargetOp(
            name="in_range",
            machine="swiglu",
            module=_SWIGLU_SELECT,
            source_file=_SWIGLU_SELECT_FILE,
            input_specs={"lower": 1, "upper": 1},
            build_graph=lambda nodes: swiglu_ops.in_range(
                nodes["lower"], nodes["upper"], 8
            ),
            reference_fn=lambda inputs: torch.where(
                (inputs["lower"] <= torch.arange(8, dtype=torch.float32) + 0.5)
                & (torch.arange(8, dtype=torch.float32) + 0.5 < inputs["upper"]),
                torch.ones(inputs["lower"].shape[0], 8),
                -torch.ones(inputs["lower"].shape[0], 8),
            ),
            distribution_names=("in_range_integer_bounds_8",),
            notes=(
                "Per slot, two compare-shaped sharpened ramps combined in "
                "out_proj. Integer bounds are exact to the folded ulp "
                "class; continuous bounds inherit compare's ramp contract "
                "per boundary plus up to 4·swish_dip/scale ≈ 0.011 of "
                "fillet dip (the value-range assert and broadcast_select's "
                "mask tolerance both carry it). (The relu in_range was "
                "never measured.)"
            ),
        ),
        TargetOp(
            name="broadcast_select",
            machine="swiglu",
            module=_SWIGLU_SELECT,
            source_file=_SWIGLU_SELECT_FILE,
            input_specs={"masks": 4, "t": 4, "f": 4},
            build_graph=lambda nodes: swiglu_ops.broadcast_select(
                nodes["masks"], nodes["t"], nodes["f"], n_slots=4, d_fill=1
            ),
            reference_fn=lambda inputs: torch.where(
                inputs["masks"] > 0, inputs["t"], inputs["f"]
            ),
            distribution_names=("broadcast_select_pm1_masks_4x1",),
            notes=(
                "Per slot, select's complementary gated pair with mask_i "
                "as the cond — one form for every caller. Junk-mask safe "
                "with no ±1 assert; a mask off ±1 by δ mis-scales the "
                "winner by δ·|value|. (The relu broadcast_select was "
                "never measured.)"
            ),
        ),
        TargetOp(
            name="dynamic_extract",
            machine="swiglu",
            module=_SWIGLU_SELECT,
            source_file=_SWIGLU_SELECT_FILE,
            input_specs={"table": 8, "idx": 1},
            build_graph=lambda nodes: swiglu_ops.dynamic_extract(
                nodes["table"], nodes["idx"], n_entries=4, d_fill=2
            ),
            reference_fn=lambda inputs: inputs["table"]
            .view(-1, 4, 2)[
                torch.arange(inputs["table"].shape[0]),
                inputs["idx"].flatten().long(),
            ]
            .view(-1, 2),
            distribution_names=("dynamic_extract_int_idx_4x2",),
            notes=(
                "in_range one-hot → broadcast_select (zero-literal false "
                "branch, lanes dropped) → free Linear sum. Losing slots "
                "are exactly zero at clean masks, so the sum degenerates "
                "to a copy of the selected slot."
            ),
        ),
        TargetOp(
            name="map_to_table",
            machine="swiglu",
            module=_SWIGLU_SELECT,
            source_file=_SWIGLU_SELECT_FILE,
            input_specs={"x": 3},
            build_graph=lambda nodes: swiglu_ops.map_to_table(
                nodes["x"],
                {
                    torch.tensor([2.0, 0.0, 1.0]): torch.tensor([10.0, -1.0]),
                    torch.tensor([0.0, 3.0, -1.0]): torch.tensor([20.0, -2.0]),
                    torch.tensor([-1.0, -1.0, 2.0]): torch.tensor([30.0, -3.0]),
                },
                torch.tensor([5.0, 0.5]),
            ),
            reference_fn=lambda inputs: _map_to_table_ref(inputs["x"]),
            distribution_names=("map_to_table_match_and_off",),
            notes=(
                "Bank of equals_vector hinges rescaled to 0/1, one "
                "degenerate lane per entry, ported hinge-for-hinge. "
                "Matches read their value to ~1 ulp (the ×scale/÷scale "
                "round trip); no-match indicators underflow, so misses "
                "return the default bit-exactly. Between-keys inputs can "
                "partially fire several indicators, as today. (The relu "
                "map_to_table was never measured.)"
            ),
        ),
        TargetOp(
            name="onehot_lookup",
            machine="swiglu",
            module="torchwright.ops.swiglu.onehot_table",
            source_file="torchwright/ops/swiglu/onehot_table.py",
            input_specs={"x": 5},
            build_graph=lambda nodes: swiglu_ops.onehot_lookup(
                nodes["x"],
                {
                    _oh_key(0, 0): torch.tensor([100.0]),
                    _oh_key(1, 0): torch.tensor([200.0]),
                    _oh_key(2, 1): torch.tensor([300.0]),
                },
                torch.tensor([-7.0]),
            ),
            reference_fn=lambda inputs: _onehot_lookup_ref(inputs["x"]),
            distribution_names=("onehot_lookup_two_block",),
            notes=(
                "Exact integer block counting: the -(n_blocks - 0.5) bias "
                "parks the winner's sharpened hinge argument at +scale/2 "
                "and everyone else's at ≤ -scale/2. Winner indicator is "
                "exactly 0.5 in fp32; losing lanes leak hinge(-0.5) ≈ "
                "-1e-28 — invisible. The tight [min, max] range claim is "
                "the reason this op exists. (The relu onehot_lookup was "
                "never measured.)"
            ),
        ),
        TargetOp(
            name="piecewise_linear",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.piecewise_linear(
                nodes["x"],
                breakpoints=[float(i) for i in range(11)],
                fn=lambda v: v * v,
            ),
            reference_fn=lambda inputs: inputs["x"] ** 2,
            distribution_names=(
                "parabola_0_10_step1",
                "staircase_64_lanes",
                "staircase_512_lanes",
                "staircase_2048_lanes",
            ),
            build_graphs_per_distribution={
                "staircase_64_lanes": _staircase_build(swiglu_ops.piecewise_linear, 64),
                "staircase_512_lanes": _staircase_build(
                    swiglu_ops.piecewise_linear, 512
                ),
                "staircase_2048_lanes": _staircase_build(
                    swiglu_ops.piecewise_linear, 2048
                ),
            },
            reference_fns_per_distribution={
                "staircase_64_lanes": _staircase_reference(64),
                "staircase_512_lanes": _staircase_reference(512),
                "staircase_2048_lanes": _staircase_reference(2048),
            },
            notes=(
                "The relu construction with sharpened hinges "
                "(K = scale·input_scale in the gate rows). Chord error "
                "between breakpoints is untouched and dominates, exactly "
                "as on relu; each corner adds a radius-17/K fillet whose "
                "dip (≤ swish_dip·|Δm|/K) bends toward the curve — grids "
                "with spacing under 34/K must raise input_scale (the "
                "spacing audit; reciprocal and the BOS inversion table do). "
                "The staircase distributions measure the "
                "univariate-collapse configuration (step pairs "
                "1/step_sharpness apart at input_scale = step_sharpness, "
                "which clears the spacing audit; in-contract inputs sit "
                "≥ 0.4 from every hinge, so fillet tails underflow and "
                "the measured axis is fp32 lane-sum accumulation, as on "
                "relu)."
            ),
        ),
        TargetOp(
            name="clamp",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.clamp(nodes["x"], lo=-10.0, hi=10.0),
            reference_fn=lambda inputs: torch.clamp(inputs["x"], -10.0, 10.0),
            distribution_names=("clamp_flat_zone_pm20",),
            notes=(
                "piecewise_linear with 4 breakpoints, input_scale = "
                "step_sharpness; corners 0.1 apart clear the 34/K spacing "
                "audit 3x, so the only additions over relu are two "
                "isolated fillets (≤ swish_dip·|Δm|/K each)."
            ),
        ),
        TargetOp(
            name="reciprocal",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.reciprocal(
                nodes["x"], min_value=0.3, max_value=200.0, step=1.0
            ),
            reference_fn=lambda inputs: 1.0 / inputs["x"],
            distribution_names=("reciprocal_03_200",),
            notes=(
                "Geometric grid; the smallest gap (at min_value) fails the "
                "34/K spacing audit at input_scale=1 (stacked fillets "
                "reached ~2e-2, a 20x regression), so the op derives "
                "input_scale from the smallest gap — with the fillets "
                "separated, the measured error is the chord class, as on "
                "relu."
            ),
        ),
        TargetOp(
            name="thermometer_floor_div",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.thermometer_floor_div(
                nodes["x"], divisor=10, max_value=100
            ),
            reference_fn=lambda inputs: torch.floor(inputs["x"] / 10.0),
            distribution_names=("thermometer_integers_0_100_by10",),
            notes=(
                "Integer-input-only staircase via piecewise_linear at "
                "input_scale = step_sharpness. Integer inputs sit in flat "
                "zones ≥ 4 units from every ramp — saturated, so exact to "
                "the folded ulp class."
            ),
        ),
        TargetOp(
            name="mod_const",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.mod_const(
                nodes["x"], divisor=7, max_value=100
            ),
            reference_fn=lambda inputs: inputs["x"]
            - 7.0 * torch.floor(inputs["x"] / 7.0),
            distribution_names=("mod_integers_0_100_by7",),
            notes=(
                "x − d·thermometer_floor_div(x, d) — linear hardware plus "
                "the staircase; exact on integer inputs to the folded ulp "
                "class, as on relu."
            ),
        ),
        TargetOp(
            name="floor_int",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.floor_int(
                nodes["x"], min_value=-5, max_value=10
            ),
            reference_fn=lambda inputs: _floor_ref(inputs["x"], -5.0, 10.0),
            distribution_names=(
                "floor_uniform_neg5_10",
                "floor_near_boundary_10",
                "floor_sawtooth_uniform_0_1023",
            ),
            build_graphs_per_distribution={
                "floor_sawtooth_uniform_0_1023": lambda nodes: swiglu_ops.floor_int(
                    nodes["x"],
                    min_value=0,
                    max_value=1023,
                    output_map=lambda k: float(k % 256),
                ),
            },
            reference_fns_per_distribution={
                "floor_sawtooth_uniform_0_1023": lambda inputs: _floor_sawtooth_ref(
                    inputs["x"], 0.0, 1023.0, 256
                ),
            },
            notes=(
                "Two chained FFNs (bounded step, then not-yet-ON counting) "
                "— the depth is load-bearing exactly as on relu (fp32 2^24 "
                "accumulation), and the W-slack absorbs stage-1 fillet "
                "noise on ON steps the same way it absorbs fp ulps. Ramp "
                "and fillet zones near boundaries interpolate, as today; "
                "flat zones and integers are exact to the folded ulp class. "
                "The sawtooth distribution measures the `output_map` mode "
                "(a following piecewise-constant op folded in by reweighting "
                "the stage-2 indicators with per-boundary deltas): reweighting "
                "exact 0/1 indicators is exact, but a ramp-zone off-by-one is "
                "amplified by the local |delta| — up to -(H-1) = -255 at the "
                "wrap boundaries, which dominates this op's footer max."
            ),
        ),
        TargetOp(
            name="ceil_int",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.ceil_int(
                nodes["x"], min_value=-5, max_value=10
            ),
            reference_fn=lambda inputs: torch.clamp(
                torch.ceil(inputs["x"]), -5.0, 10.0
            ),
            distribution_names=("ceil_integers_neg5_10",),
            notes="−floor_int(−x); inherits floor_int's entry.",
        ),
        TargetOp(
            name="radix_floor_int",
            machine="swiglu",
            module=_SWIGLU_ARITH,
            source_file=_SWIGLU_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: swiglu_ops.radix_floor_int(
                nodes["x"], min_value=-1023, max_value=1023, sharpness=10_000.0
            ),
            reference_fn=lambda inputs: _floor_ref(inputs["x"], -1023.0, 1023.0),
            distribution_names=(
                "radix_floor_native_uniform",
                "radix_floor_divisor_sliver",
                "radix_floor_lo_ramp",
            ),
            notes=(
                "floor(x) as hi-floor -> integer snap -> lo-floor over "
                "[-1, D] (D = power of two nearest sqrt(2N)); ~8.5*sqrt(N) "
                "lanes vs 3N flat, intermediates ~sqrt(N) wide. The snapped "
                "hi and the extended lo range make the divisor-boundary "
                "sliver reconstruct exactly (no D-amplification); the LO "
                "ramp keeps flat floor_int's tolerated ±1-step window. "
                "Measured at the DOOM native-coordinate configuration "
                "(N=2046, s=10^4, D=64)."
            ),
        ),
        TargetOp(
            name="table_lookup_2d",
            machine="swiglu",
            module=_SWIGLU_SELECT,
            source_file=_SWIGLU_SELECT_FILE,
            input_specs={"i": 1, "j": 1},
            build_graph=lambda nodes: swiglu_ops.table_lookup_2d(
                nodes["i"], nodes["j"], _TL2D_TABLE
            ),
            reference_fn=_table_lookup_2d_ref,
            distribution_names=("table_lookup_2d_int_8x6",),
            notes=(
                "Both axes floor_int's two-stage staircase; the column "
                "stage is one gated lane per boundary (up row = the live "
                "row vector's adjacent difference) — the ReLU machine's "
                "mask staircase and offset-cancellation gate collapse into "
                "it, deleting the offset·0.005 guard term (the family's "
                "last range-coupled offset). On the integer grid every "
                "indicator is exactly 0/1, so the error is the delta "
                "telescoping's fp32 accumulation. (The relu table_lookup_2d "
                "was never measured.)"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Measurement driver
# ---------------------------------------------------------------------------


def _measure_all() -> List[NoiseMeasurement]:
    torch.set_default_device("cpu")
    dists = _distributions()
    out: List[NoiseMeasurement] = []
    for target in _target_ops():
        for dname in target.distribution_names:
            if dname not in dists:
                raise KeyError(f"unknown distribution {dname!r} for op {target.name!r}")
            build_graph = target.build_graphs_per_distribution.get(
                dname, target.build_graph
            )
            reference_fn = target.reference_fns_per_distribution.get(
                dname, target.reference_fn
            )
            m = measure_op_isolated(
                op_name=target.name,
                module=target.module,
                build_graph=build_graph,
                input_specs=target.input_specs,
                reference_fn=reference_fn,
                distribution=dists[dname],
                notes=target.notes,
                machine=target.machine,
            )
            out.append(m)
    return out


# ---------------------------------------------------------------------------
# JSON rendering
# ---------------------------------------------------------------------------


def _round_float(x: float, sig: int = 6) -> Optional[float]:
    """Round ``x`` to ``sig`` sig figs. Returns ``None`` for NaN/inf so the
    JSON encoder emits ``null`` instead of a non-portable ``NaN`` literal."""
    if x != x or x in (float("inf"), float("-inf")):
        return None
    if x == 0:
        return 0.0
    import math

    digits = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, digits)


def render_json(
    measurements: Sequence[NoiseMeasurement], commit: str, measured_at: str
) -> str:
    """Render the canonical JSON document.

    Ops are keyed by ``(machine, name)`` — the relu and swiglu libraries
    share op names — and sorted machine-first, so the frozen relu block
    stays contiguous and diff-stable as swiglu entries land.
    """
    by_op: Dict[tuple, Dict] = {}
    for m in measurements:
        key = (m.machine, m.op_name)
        by_op.setdefault(
            key,
            {
                "name": m.op_name,
                "machine": m.machine,
                "module": m.module,
                "notes": m.notes,
                "distributions": [],
            },
        )
        by_op[key]["distributions"].append(
            {
                "name": m.distribution_name,
                "description": m.distribution_description,
                "n_samples": m.n_samples,
                "max_abs_error": _round_float(m.max_abs_error),
                "mean_abs_error": _round_float(m.mean_abs_error),
                "p99_abs_error": _round_float(m.p99_abs_error),
                "max_rel_error": _round_float(m.max_rel_error),
                "mean_rel_error": _round_float(m.mean_rel_error),
                "p99_rel_error": _round_float(m.p99_rel_error),
                "rel_valid_samples": m.rel_valid_samples,
                "worst_input": {k: _round_float(v) for k, v in m.worst_input.items()},
            }
        )
    ops_sorted = sorted(by_op.values(), key=lambda d: (d["machine"], d["name"]))
    for op in ops_sorted:
        op["distributions"].sort(key=lambda d: d["name"])
    doc = {
        "schema_version": 2,
        "commit": commit,
        "measured_at": measured_at,
        "ops": ops_sorted,
    }
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _fmt_opt(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4g}"


def render_markdown(data: Dict) -> str:
    """Render the human-facing doc from parsed JSON."""
    commit = data["commit"]
    measured_at = data["measured_at"]
    lines: List[str] = []
    lines.append("# Numerical noise reference")
    lines.append("")
    lines.append("*Generated by `scripts/measure_op_noise.py`; do not edit by hand.*")
    lines.append(f"*Measured at commit `{commit}` on `{measured_at}`.*")
    lines.append("")
    lines.append(
        "Every piecewise-linear op in `torchwright/ops/` is measured on one or more"
    )
    lines.append("named input distributions characterising its expected input ranges.")
    lines.append("The canonical data lives in `docs/op_noise_data.json`; this file is")
    lines.append("regenerated from that JSON by `make measure-noise`.")
    lines.append(
        "*Machine* is the activation family the op library targets: "
        "`relu` (`torchwright/ops/relu/`) or `swiglu` "
        "(`torchwright/ops/swiglu/`)."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Op | Machine | Module | Max abs error | Max rel error "
        "| Worst distribution |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for op in data["ops"]:
        worst = max(op["distributions"], key=lambda d: d["max_abs_error"])
        worst_rel = max(
            (
                d["max_rel_error"]
                for d in op["distributions"]
                if d["max_rel_error"] is not None
            ),
            default=None,
        )
        lines.append(
            f"| `{op['name']}` | {op['machine']} | `{op['module']}` "
            f"| {_fmt(worst['max_abs_error'])} "
            f"| {_fmt_opt(worst_rel)} "
            f"| `{worst['name']}` |"
        )
    lines.append("")
    lines.append("## How to read these numbers")
    lines.append("")
    lines.append("Each row reports output noise *at the op's design operating point* —")
    lines.append("i.e., with clean inputs drawn from the listed distribution. It does")
    lines.append(
        "**not** bound output noise when upstream inputs are themselves noisy."
    )
    lines.append("")
    lines.append("Per-op bounds are **not additive** through a chain when any op has")
    lines.append("internal gain (a constant that multiplies an input). Gate ops")
    lines.append(
        "(`cond_gate`, `select`, `attend_mean_where`) and threshold-output ops"
    )
    lines.append(
        "(`compare` in its ramp zone, `floor_int` near integer boundaries) all"
    )
    lines.append(
        "have structural gains > 1. An upstream deviation of magnitude `1/gain`"
    )
    lines.append("can push the gated op outside its published bound, and the bias")
    lines.append("propagates multiplicatively downstream.")
    lines.append("")
    lines.append(
        "To reason about a specific chain, don't try to compose per-op bounds."
    )
    lines.append("Probe the suspect node directly with")
    lines.append(
        "`torchwright.debug.probe.probe_compiled`; the residual at the node is"
    )
    lines.append("observable, the per-op bounds are an aid for spotting the op *class*")
    lines.append("most likely responsible.")
    lines.append("")
    for op in data["ops"]:
        lines.append(f"## `{op['name']}` ({op['machine']})")
        lines.append("")
        if op["notes"]:
            lines.append(op["notes"])
            lines.append("")
        for dist in op["distributions"]:
            lines.append(f"### `{dist['name']}`")
            lines.append("")
            lines.append(dist["description"])
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            lines.append(f"| Samples | {dist['n_samples']} |")
            lines.append(f"| Max abs error | {_fmt(dist['max_abs_error'])} |")
            lines.append(f"| Mean abs error | {_fmt(dist['mean_abs_error'])} |")
            lines.append(f"| p99 abs error | {_fmt(dist['p99_abs_error'])} |")
            lines.append(
                f"| Max rel error | {_fmt_opt(dist['max_rel_error'])} "
                f"(over {dist['rel_valid_samples']} samples with "
                f"&#124;ref&#124; ≥ 1e-6) |"
            )
            lines.append(f"| Mean rel error | {_fmt_opt(dist['mean_rel_error'])} |")
            lines.append(f"| p99 rel error | {_fmt_opt(dist['p99_rel_error'])} |")
            worst_fmt = ", ".join(
                f"`{k}={_fmt(v)}`" for k, v in dist["worst_input"].items()
            )
            lines.append(f"| Worst input | {worst_fmt} |")
            lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "Measurements run through `torchwright.debug.noise.measure_op_isolated`,"
    )
    lines.append("which builds a minimal graph around the op and evaluates it via")
    lines.append("`Node.compute` — the same oracle path the compiler already trusts as")
    lines.append(
        "the semantic definition of each node. The piecewise-linear approximation"
    )
    lines.append(
        "error is the difference between that oracle evaluation and the exact-math"
    )
    lines.append("reference function declared in `_target_ops` at")
    lines.append("`scripts/measure_op_noise.py`.")
    lines.append("")
    lines.append(
        "All runs are CPU-only and seeded so measurements are bit-identical across"
    )
    lines.append(
        "machines at a given commit; the consistency test in `tests/docs/` relies"
    )
    lines.append(
        "on that. Floats in the JSON are rounded to 6 significant figures; the"
    )
    lines.append("markdown and docstring footers display `.4g` formatting.")
    lines.append("")
    lines.append(
        "**Relative error** is reported alongside absolute error because absolute"
    )
    lines.append(
        "numbers are misleading for ops whose output magnitude varies over orders of"
    )
    lines.append(
        "magnitude (reciprocal, multiplication, low-rank lookups). Per-sample relative"
    )
    lines.append("error is `|compiled - reference| / |reference|`, computed only where")
    lines.append(
        "`|reference| >= 1e-6` — samples with a near-zero reference are excluded"
    )
    lines.append(
        "(rel error is ill-defined there) and the count of contributing samples is"
    )
    lines.append(
        "reported as `rel_valid_samples`. Ops whose reference is always near zero"
    )
    lines.append("show `n/a` for relative error.")
    lines.append("")
    lines.append("## Adding a new op")
    lines.append("")
    lines.append(
        "1. In `scripts/measure_op_noise.py`, add any new `InputDistribution` to"
    )
    lines.append("   `_distributions()` with a production-grounded name.")
    lines.append("2. Append a `TargetOp(...)` to `_target_ops()` wiring `build_graph`,")
    lines.append("   `reference_fn`, `input_specs`, and `distribution_names`.")
    lines.append("3. Run `make measure-noise`. Commit the diff in `docs/`,")
    lines.append("   `torchwright/ops/`, and nothing else.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Docstring patching
# ---------------------------------------------------------------------------


class FooterSummary(TypedDict):
    max_abs_error: float
    max_rel_error: float
    total_samples: int


def _footer_summary(op: Dict) -> FooterSummary:
    """Aggregate across every distribution of ``op``.

    The docstring footer reports the worst-case abs and rel error and the
    total sample count across all distributions — no single distribution name
    leaks into the op's docstring (project-specific distribution names belong
    in the markdown, not in a generic math-op docstring).
    """
    dists = op["distributions"]
    worst_abs = max(float(d["max_abs_error"]) for d in dists)
    rel_values = [d["max_rel_error"] for d in dists if d["max_rel_error"] is not None]
    worst_rel = max(rel_values) if rel_values else float("nan")
    total_samples = sum(int(d["n_samples"]) for d in dists)
    return {
        "max_abs_error": worst_abs,
        "max_rel_error": worst_rel,
        "total_samples": total_samples,
    }


def _apply_docstring_footers(data: Dict, *, check: bool) -> List[str]:
    """Update (or check) docstring footers for every op in the JSON.

    Returns the list of file paths that were (or would be) changed.
    """
    target_map = {(t.machine, t.name): t for t in _target_ops()}
    changed: List[str] = []
    for op in data["ops"]:
        target = target_map.get((op["machine"], op["name"]))
        if target is None:
            raise KeyError(
                f"JSON contains op ({op['machine']!r}, {op['name']!r}) "
                f"with no TargetOp record"
            )
        path = REPO_ROOT / target.source_file
        source = path.read_text()
        summary = _footer_summary(op)
        new_source = update_docstring_footer(
            source,
            op["name"],
            max_abs_error=summary["max_abs_error"],
            max_rel_error=summary["max_rel_error"],
            total_samples=summary["total_samples"],
            commit=data["commit"],
        )
        if new_source != source:
            changed.append(str(target.source_file))
            if not check:
                path.write_text(new_source)
    return changed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=7", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.decode().strip()


def _today_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).date().isoformat()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Measure per-op numerical noise.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write artefacts; exit non-zero if they would change.",
    )
    args = parser.parse_args(argv)

    measurements = _measure_all()
    commit = _git_short_sha()
    measured_at = _today_utc()
    json_text = render_json(measurements, commit=commit, measured_at=measured_at)
    data = json.loads(json_text)
    md_text = render_markdown(data)

    changed: List[str] = []
    if not DOCS_JSON.exists() or DOCS_JSON.read_text() != json_text:
        changed.append(str(DOCS_JSON.relative_to(REPO_ROOT)))
        if not args.check:
            DOCS_JSON.parent.mkdir(parents=True, exist_ok=True)
            DOCS_JSON.write_text(json_text)
    if not DOCS_MD.exists() or DOCS_MD.read_text() != md_text:
        changed.append(str(DOCS_MD.relative_to(REPO_ROOT)))
        if not args.check:
            DOCS_MD.parent.mkdir(parents=True, exist_ok=True)
            DOCS_MD.write_text(md_text)

    changed.extend(_apply_docstring_footers(data, check=args.check))

    if args.check and changed:
        print(
            "Noise artefacts out of sync. Run `make measure-noise` to fix.",
            file=sys.stderr,
        )
        for c in changed:
            print(f"  - {c}", file=sys.stderr)
        return 1

    for c in changed:
        print(f"updated {c}")
    if not changed:
        print("no changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
