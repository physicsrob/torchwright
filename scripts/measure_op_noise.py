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
from torchwright.ops.arithmetic_ops import (
    abs as abs_op,
    ceil_int,
    clamp,
    compare,
    exp,
    floor_int,
    linear_bin_index,
    log,
    log_abs,
    low_rank_2d,
    max as max_op,
    min as min_op,
    mod_const,
    multiply_2d,
    multiply_integers,
    piecewise_linear,
    piecewise_linear_2d,
    reciprocal,
    signed_multiply,
    square,
    square_signed,
    thermometer_floor_div,
)
from torchwright.ops.logic_ops import (
    bool_all_true,
    bool_any_true,
    bool_not,
    cond_gate,
    equals_vector,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_JSON = REPO_ROOT / "docs" / "op_noise_data.json"
DOCS_MD = REPO_ROOT / "docs" / "numerical_noise.md"


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
    """

    name: str
    module: str
    build_graph: Callable[[Dict[str, Node]], Node]
    input_specs: Dict[str, int]
    reference_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor]
    distribution_names: Sequence[str]
    source_file: str
    notes: str = ""
    build_graphs_per_distribution: Dict[str, Callable[[Dict[str, Node]], Node]] = field(
        default_factory=dict
    )


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
        torch.randint(0, 2, (n_random,), generator=gen).to(torch.float32) * 2.0
        - 1.0
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
    }


# ---------------------------------------------------------------------------
# Target ops
# ---------------------------------------------------------------------------


def _floor_ref(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return torch.clamp(torch.floor(x), lo, hi)


def _bin_index_ref(
    x: torch.Tensor, x_min: torch.Tensor, x_max: torch.Tensor, n_bins: int
) -> torch.Tensor:
    r = x_max - x_min
    v = (x - x_min) * n_bins / r
    return torch.clamp(torch.floor(v), 0, n_bins - 1)


def _target_ops() -> List[TargetOp]:
    _ARITH = "torchwright.ops.arithmetic_ops"
    _ARITH_FILE = "torchwright/ops/arithmetic_ops.py"

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
            name="log",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: log(
                nodes["x"], min_value=0.01, max_value=100.0, n_breakpoints=256
            ),
            reference_fn=lambda inputs: torch.log(inputs["x"]),
            distribution_names=(
                "log_4decades_001_100",
                "log_6decades_wide",
            ),
            build_graphs_per_distribution={
                "log_6decades_wide": lambda nodes: log(
                    nodes["x"],
                    min_value=0.01,
                    max_value=30000.0,
                    n_breakpoints=256,
                ),
            },
            notes=(
                "Natural log via per-section piecewise-linear "
                "interpolation. The op auto-sections the input range "
                "geometrically by `section_factor=10` (default decades) "
                "and routes via thermometer compare + multiply_2d "
                "blending, so float32 cancellation is bounded by "
                "section width regardless of overall input range. Pairs "
                "with `exp` for log-space arithmetic chains."
            ),
        ),
        TargetOp(
            name="log_abs",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: log_abs(
                nodes["x"], min_abs=0.1, max_abs=100.0, n_breakpoints=256
            ),
            reference_fn=lambda inputs: torch.log(
                torch.clamp(inputs["x"].abs(), min=0.1, max=100.0)
            ),
            distribution_names=("log_abs_3decades_pm100",),
            notes=(
                "Fused `log(clamp(|x|, min_abs, max_abs))` for signed "
                "input. Single-piecewise V-shape over signed `x` for "
                "`max_abs/min_abs ≤ 10⁴` (1 sublayer); falls back to "
                "`abs + sectioned log` for wider ratios (3 sublayers). "
                "Pairs with `exp` and Linear addition for log-domain "
                "multiplication of a signed by a positive value."
            ),
        ),
        TargetOp(
            name="exp",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: exp(
                nodes["x"], min_value=-5.0, max_value=5.0, n_breakpoints=256
            ),
            reference_fn=lambda inputs: torch.exp(inputs["x"]),
            distribution_names=("exp_pm5",),
            notes=(
                "Natural exponential via piecewise-linear interpolation "
                "with uniform breakpoint spacing. Constant relative output "
                "error per cell `(Δx)²/8` because `d²exp/dx² = exp`. "
                "Pairs with `log` to implement `A·B = exp(log A + log B)` "
                "and `A/B = exp(log A − log B)` in log-space chains."
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
            distribution_names=("parabola_0_10_step1",),
            notes=(
                "Foundation 1D piecewise-linear primitive; measured on the "
                "canonical x² test function with 11 integer breakpoints. Error "
                "is zero at breakpoints and bounded by the interpolation "
                "error of the reference function over each segment."
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
            name="square_signed",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1},
            build_graph=lambda nodes: square_signed(nodes["x"], max_abs=10.0, step=1.0),
            reference_fn=lambda inputs: inputs["x"] ** 2,
            distribution_names=("square_signed_pm10",),
            notes=(
                "Same interpolation profile as `square` but spans [-max_abs, "
                "max_abs]; saves one MLP sublayer vs. abs+square in "
                "`multiply_integers` / `signed_multiply` deep strategies."
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
            name="signed_multiply",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: signed_multiply(
                nodes["a"],
                nodes["b"],
                max_abs1=10.0,
                max_abs2=10.0,
                step=1.0,
            ),
            reference_fn=lambda inputs: inputs["a"] * inputs["b"],
            distribution_names=("signed_multiply_pm10",),
            notes=(
                "Polarization identity `a·b = (|a+b|² - |a-b|²)/4`. Absolute "
                "error scales with `step × (max_abs1 + max_abs2)`, not with "
                "`|a·b|`; pathological near-zero × large-magnitude inputs "
                "exhibit high relative error."
            ),
        ),
        TargetOp(
            name="low_rank_2d",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: low_rank_2d(
                nodes["a"],
                nodes["b"],
                breakpoints1=_ATAN_BP_CROSS,
                breakpoints2=_ATAN_BP_DOT,
                fn=lambda x, y: float(torch.atan(torch.tensor(x / y)).item()),
                rank=3,
            ),
            reference_fn=lambda inputs: torch.atan(inputs["a"] / inputs["b"]),
            distribution_names=("atan_cross_dot_nonuniform",),
            notes=(
                "Rank-3 SVD of `atan(cross/dot)` on a non-uniform grid. "
                "Worst-cell error is bounded by σ_{K+1}, the first truncated "
                "singular value — verify against the SVD computed in "
                "`tests/ops/test_low_rank_2d.py:100`."
            ),
        ),
        TargetOp(
            name="piecewise_linear_2d",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: piecewise_linear_2d(
                nodes["a"],
                nodes["b"],
                breakpoints1=_DIFF_BP,
                breakpoints2=_TRIG_BP,
                fn=lambda x, y: x * y,
            ),
            reference_fn=lambda inputs: inputs["a"] * inputs["b"],
            distribution_names=("diff_trig_nonuniform", "diff_vel_nonuniform"),
            notes=(
                "Triangulated-grid lookup with non-uniform breakpoints. On "
                "non-uniform grids `piecewise_linear_2d` uses a constrained "
                "least-squares fit that can oscillate in cell interiors — "
                "`low_rank_2d` is preferred there."
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
        TargetOp(
            name="linear_bin_index",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"x": 1, "x_min": 1, "x_max": 1},
            build_graph=lambda nodes: linear_bin_index(
                nodes["x"],
                nodes["x_min"],
                nodes["x_max"],
                n_bins=16,
                min_range=0.5,
                max_range=200.0,
            ),
            reference_fn=lambda inputs: _bin_index_ref(
                inputs["x"], inputs["x_min"], inputs["x_max"], n_bins=16
            ),
            distribution_names=("linear_bin_index_tex_col_16",),
            notes=(
                "End-to-end composite (~6 MLP sublayers). Error compounds "
                "across `reciprocal`, `signed_multiply`, `clamp`, and the "
                "terminal `thermometer_floor_div`. Inputs close to integer "
                "bin boundaries land in the staircase's ramp zone; `_bin_index_ref` "
                "uses `torch.floor` on the exact linear mapping, so tight "
                "bin edges reveal ramp-zone slippage."
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
            name="max",
            module=_ARITH,
            source_file=_ARITH_FILE,
            input_specs={"a": 1, "b": 1},
            build_graph=lambda nodes: max_op(nodes["a"], nodes["b"]),
            reference_fn=lambda inputs: torch.maximum(inputs["a"], inputs["b"]),
            distribution_names=("minmax_uniform_pm50",),
            notes="Exact via `(a + b + |a - b|) / 2`.",
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
            module="torchwright.ops.logic_ops",
            source_file="torchwright/ops/logic_ops.py",
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
            module="torchwright.ops.logic_ops",
            source_file="torchwright/ops/logic_ops.py",
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
            module="torchwright.ops.logic_ops",
            source_file="torchwright/ops/logic_ops.py",
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
            module="torchwright.ops.logic_ops",
            source_file="torchwright/ops/logic_ops.py",
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
            module="torchwright.ops.logic_ops",
            source_file="torchwright/ops/logic_ops.py",
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
            m = measure_op_isolated(
                op_name=target.name,
                module=target.module,
                build_graph=build_graph,
                input_specs=target.input_specs,
                reference_fn=target.reference_fn,
                distribution=dists[dname],
                notes=target.notes,
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
    """Render the canonical JSON document."""
    by_op: Dict[str, Dict] = {}
    for m in measurements:
        by_op.setdefault(
            m.op_name,
            {
                "name": m.op_name,
                "module": m.module,
                "notes": m.notes,
                "distributions": [],
            },
        )
        by_op[m.op_name]["distributions"].append(
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
    ops_sorted = sorted(by_op.values(), key=lambda d: d["name"])
    for op in ops_sorted:
        op["distributions"].sort(key=lambda d: d["name"])
    doc = {
        "schema_version": 1,
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
    lines.append(
        "The canonical data lives in `docs/op_noise_data.json`; this file is"
    )
    lines.append("regenerated from that JSON by `make measure-noise`.")
    lines.append("")
    lines.append(
        "For commentary on the numbers — which are expected, which warrant "
        "investigation — see the hand-written "
        "`docs/numerical_noise_findings.md`."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Op | Module | Max abs error | Max rel error | Worst distribution |")
    lines.append("| --- | --- | --- | --- | --- |")
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
            f"| `{op['name']}` | `{op['module']}` "
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
    lines.append("propagates multiplicatively downstream. See")
    lines.append(
        "`docs/numerical_noise_findings.md` for the worked Phase E example and"
    )
    lines.append("the list of known amplification hazards.")
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
        lines.append(f"## `{op['name']}`")
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
    target_map = {t.name: t for t in _target_ops()}
    changed: List[str] = []
    for op in data["ops"]:
        target = target_map.get(op["name"])
        if target is None:
            raise KeyError(f"JSON contains op {op['name']!r} with no TargetOp record")
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
