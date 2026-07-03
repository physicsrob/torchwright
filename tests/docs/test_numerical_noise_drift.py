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
_ATOL = 1e-6


def _close_enough(a: float, b: float) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if a == b:
        return True
    return abs(a - b) <= _ATOL + _RTOL * max(abs(a), abs(b))


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
                if not _close_enough(cv, rv):
                    failures.append(f"{op_name}/{dist_name}: {key} " f"{cv} -> {rv}")

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
