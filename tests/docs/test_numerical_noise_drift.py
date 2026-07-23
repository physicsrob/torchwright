"""Drift test for the per-op noise measurement pipeline.

Re-runs `_measure_all()` on CPU and compares the freshly-regenerated JSON
to the committed `docs/op_noise_data.json`. Fails if any op's measured
numbers have drifted from what's been committed — the canonical signal
that an op's implementation, breakpoint grid, or measurement distribution
has changed without `make measure-noise` being re-run.

The comparison uses relative tolerance (not exact equality) for error
magnitudes, because float32 arithmetic varies across CPU architectures
(e.g. local dev machine vs Modal's AMD EPYC). Fields like ``worst_input``
are ignored entirely since they're the most hardware-sensitive and carry
no safety information. Structural fields (op names, distribution names,
sample counts) are still exact-matched.

Pairs with `test_numerical_noise_consistency.py`:
  - `test_numerical_noise_consistency.py` (~30ms): verifies JSON, markdown,
    and docstring footers agree with each other. Format/schema drift.
  - `test_numerical_noise_drift.py` (this file, ~15s): verifies the JSON
    matches what the current code actually measures. Number drift.

Together they close the gap CLAUDE.md § "Numerical noise" used to
document as a manual obligation.
"""

from __future__ import annotations

import json
from typing import List

from scripts.measure_op_noise import (
    DOCS_JSON,
    _measure_all,
    render_json,
)

_ERROR_METRIC_KEYS = [
    "max_abs_error",
    "mean_abs_error",
    "p99_abs_error",
    "max_rel_error",
    "mean_rel_error",
    "p99_rel_error",
]

# 0.40 (was 0.30) so the check absorbs measurements that sit on a precision
# boundary and flip between cross-test GPU states / CPU configs run-to-run. The
# motivating case is `reciprocal_03_200`'s p99 error, which is bistable on Modal
# (~4.0e-4 vs ~5.8e-4, a ~31% swing) depending on concurrent-shard GPU state —
# no single committed value passes at 0.30 in both states. This guard catches
# gross drift (a number moving by >40%); it is not a tight regression bound.
# See docs/numerical_noise_findings.md.
_RTOL = 0.40
# 5e-4 (was 1e-6): the swiglu ops' measured floor is fp32 product rounding at
# the lane-contribution magnitude, and that rounding is kernel-dependent (FMA
# vs per-product) — an entry measuring exactly 0.0 on one CPU reads a few
# contribution-ulps on another (observed: compare_uniform_pm80 p99 0.0 locally
# vs 5.8e-5 on Modal's EPYC; worst contribution in the table is ~1600, so up
# to ~2e-4). The relu exact ops never exposed this — their identities are
# exact on every kernel. Absolute moves above 5e-4 are genuine drift; tight
# per-op budgets live in the ops' unit tests, not here.
_ATOL = 5e-4
# The staircase rows measure fp32 GEMM reduction order, not the op: the
# decomposition sums lanes whose contributions reach ~6.4e5 and cancel to
# ~4.0, so the surviving error is a small multiple of the fp32 ulp at the
# partial-sum magnitude — quantized in powers of two, and one to two binades
# apart between sequential (local CPU BLAS) and blocked (A100 cuBLAS)
# reduction orders (largest observed swing: p99_abs_error 4x, 0.03125 ->
# 0.125). A factor-of-8 ratio guard covers both orders with one binade of
# headroom while still catching a genuine decomposition regression, which
# moves these numbers by orders of magnitude. See
# docs/numerical_noise_findings.md § staircase.
_STAIRCASE_RATIO = 8.0


def _is_staircase(op_key: tuple, dist_name: str) -> bool:
    return op_key[1] == "piecewise_linear" and dist_name.startswith("staircase_")


def _close_enough(a: float, b: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    return abs(a - b) <= _ATOL + _RTOL * max(abs(a), abs(b))


def _ratio_close(a: float, b: float, factor: float) -> bool:
    """True when the two magnitudes are within ``factor`` of each other —
    the machine-dependent-row guard (see _STAIRCASE_RATIO)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    lo, hi = sorted((abs(a), abs(b)))
    if lo == 0.0:
        return hi <= _ATOL
    return hi / lo <= factor


def _compare_ops(committed: dict, regenerated: dict) -> List[str]:
    """Compare two stripped JSON dicts and return a list of failure messages.

    Ops are keyed by ``(machine, name)`` — the relu and swiglu libraries
    share op names (both have a ``square``).
    """
    c_ops = {(op["machine"], op["name"]): op for op in committed["ops"]}
    r_ops = {(op["machine"], op["name"]): op for op in regenerated["ops"]}

    failures: List[str] = []

    missing = sorted(c_ops.keys() - r_ops.keys())
    if missing:
        failures.append(
            f"Ops in committed JSON but not in fresh measurement: {missing}"
        )
    extra = sorted(r_ops.keys() - c_ops.keys())
    if extra:
        failures.append(f"Ops in fresh measurement but not in committed JSON: {extra}")

    for op_key in sorted(c_ops.keys() & r_ops.keys()):
        op_name = "/".join(op_key)
        c_dists = {d["name"]: d for d in c_ops[op_key]["distributions"]}
        r_dists = {d["name"]: d for d in r_ops[op_key]["distributions"]}

        d_missing = sorted(c_dists.keys() - r_dists.keys())
        if d_missing:
            failures.append(f"{op_name}: distributions removed: {d_missing}")
        d_extra = sorted(r_dists.keys() - c_dists.keys())
        if d_extra:
            failures.append(f"{op_name}: distributions added: {d_extra}")

        for dist_name in sorted(c_dists.keys() & r_dists.keys()):
            cd = c_dists[dist_name]
            rd = r_dists[dist_name]

            if cd["n_samples"] != rd["n_samples"]:
                failures.append(
                    f"{op_name}/{dist_name}: n_samples "
                    f"{cd['n_samples']} -> {rd['n_samples']}"
                )

            for key in _ERROR_METRIC_KEYS:
                cv, rv = cd[key], rd[key]
                if _is_staircase(op_key, dist_name):
                    ok = _ratio_close(cv, rv, _STAIRCASE_RATIO)
                else:
                    ok = _close_enough(cv, rv)
                if not ok:
                    failures.append(f"{op_name}/{dist_name}: {key} {cv} -> {rv}")

    return failures


def _strip_metadata(text: str) -> dict:
    data = json.loads(text)
    data.pop("commit", None)
    data.pop("measured_at", None)
    return data


def test_committed_measurements_match_current_code() -> None:
    """Re-measure every op and fail if the numbers no longer match the
    committed ``docs/op_noise_data.json``.

    To fix a failure of this test:
        1. Run ``make measure-noise`` to regenerate the JSON, markdown,
           and per-op docstring footers from a fresh measurement.
        2. Commit the regenerated files:
            - ``docs/op_noise_data.json``
            - ``docs/numerical_noise.md``
            - the updated ``.. noise-footer::`` blocks in
              ``torchwright/ops/*.py``.
        3. Per CLAUDE.md § "Numerical noise", diff the new JSON against
           the prior commit and update ``docs/numerical_noise_findings.md``
           for any findings-worthy changes.

    See CLAUDE.md § "Numerical noise" for the full workflow.
    """
    measurements = _measure_all()
    regenerated = render_json(
        measurements,
        commit="<ignored>",
        measured_at="<ignored>",
    )

    regenerated_data = _strip_metadata(regenerated)
    committed_data = _strip_metadata(DOCS_JSON.read_text())

    failures = _compare_ops(committed_data, regenerated_data)
    assert not failures, (
        "Per-op noise measurements have drifted from the committed values in "
        f"{DOCS_JSON.name}.\n"
        + "\n".join(f"  - {f}" for f in failures)
        + "\nRun `make measure-noise` to regenerate, then commit the diff. "
        "See CLAUDE.md § 'Numerical noise' for the full workflow."
    )
