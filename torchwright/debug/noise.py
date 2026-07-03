"""Per-op numerical-noise measurement harness.

Builds a minimal graph around an op, runs it on a named input distribution via
``Node.compute`` (the oracle path the compiler already trusts), and compares
the output to a reference math function to produce a ``NoiseMeasurement``.

This is the measurement primitive for ``scripts/measure_op_noise.py``. It is
CPU-only and seeded so that measurements are bit-identical across machines at
a given commit; the consistency test in ``tests/docs/`` relies on that.

See ``docs/numerical_noise.md`` for the methodology this module implements.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from torchwright.graph import Node
from torchwright.ops.inout_nodes import create_input

NOISE_FOOTER_MARKER = ".. noise-footer::"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputDistribution:
    """A named, pre-materialised batch of inputs for a single op.

    ``inputs`` maps each input-node name to a tensor of shape ``(N, d)`` where
    ``N`` is ``n_samples`` and ``d`` is the input width. ``Node.compute`` is
    called with ``n_pos=N`` so every sample runs in one pass.
    """

    name: str
    description: str
    inputs: Dict[str, torch.Tensor]
    n_samples: int


REL_ERROR_EPSILON = 1e-6
"""Floor on ``|reference|`` when computing relative error.

Samples where ``|reference| < REL_ERROR_EPSILON`` are excluded from the
relative-error aggregates — relative error is ill-defined for a reference of
zero. ``rel_valid_samples`` records how many samples contributed.
"""


@dataclass(frozen=True)
class NoiseMeasurement:
    """Result of running one op against one ``InputDistribution``.

    ``machine`` is the op-library axis (``"relu"`` or ``"swiglu"``) —
    the two libraries share op names (both have a ``square``), so
    measurements are identified by ``(machine, op_name)``.
    """

    op_name: str
    module: str
    distribution_name: str
    distribution_description: str
    n_samples: int
    max_abs_error: float
    mean_abs_error: float
    p99_abs_error: float
    max_rel_error: float
    mean_rel_error: float
    p99_rel_error: float
    rel_valid_samples: int
    worst_input: Dict[str, float]
    notes: str = ""
    machine: str = "relu"


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_op_isolated(
    *,
    op_name: str,
    module: str,
    build_graph: Callable[[Dict[str, Node]], Node],
    input_specs: Dict[str, int],
    reference_fn: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    distribution: InputDistribution,
    notes: str = "",
    machine: str = "relu",
) -> NoiseMeasurement:
    """Measure max/mean/p99 absolute error of ``build_graph`` on one distribution.

    Args:
        op_name: human name for the op (matches the function in ``torchwright/ops``).
        module: dotted module path where the op is defined.
        build_graph: takes ``{name: InputNode}`` and returns the output ``Node``.
            Called once per measurement.
        input_specs: ``{name: width}`` for each input to create via
            :func:`create_input`.
        reference_fn: exact-math reference, takes the same input-tensor dict
            and returns an ``(N, d_out)`` tensor of oracle values.
        distribution: the sample batch to run.
        notes: free-form text carried through to JSON/markdown.
        machine: op-library axis (``"relu"`` or ``"swiglu"``), carried
            through to the measurement record.
    """
    torch.manual_seed(0)

    input_nodes = {
        name: create_input(name, width) for name, width in input_specs.items()
    }
    output_node = build_graph(input_nodes)

    for name, tensor in distribution.inputs.items():
        if name not in input_specs:
            raise KeyError(f"distribution has input {name!r} not in input_specs")
        if tensor.shape[0] != distribution.n_samples:
            raise ValueError(
                f"input {name!r} has {tensor.shape[0]} rows, "
                f"expected {distribution.n_samples}"
            )
        if tensor.shape[1] != input_specs[name]:
            raise ValueError(
                f"input {name!r} has width {tensor.shape[1]}, "
                f"expected {input_specs[name]}"
            )

    with torch.no_grad():
        compiled = output_node.compute(
            n_pos=distribution.n_samples,
            input_values={
                k: v.to(torch.float32) for k, v in distribution.inputs.items()
            },
        )
        reference = reference_fn(distribution.inputs).to(torch.float32)

    if compiled.shape != reference.shape:
        raise ValueError(
            f"shape mismatch: compiled {tuple(compiled.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )

    errs = (compiled - reference).abs()
    per_sample = errs.flatten(start_dim=1).max(dim=1).values
    worst_idx = int(torch.argmax(per_sample).item())
    worst_input = {
        name: [float(v) for v in distribution.inputs[name][worst_idx].tolist()]
        for name in distribution.inputs
    }
    worst_input_flat: Dict[str, float] = {}
    for name, vals in worst_input.items():
        if len(vals) == 1:
            worst_input_flat[name] = vals[0]
        else:
            for i, v in enumerate(vals):
                worst_input_flat[f"{name}[{i}]"] = v

    rel_mask = reference.abs() >= REL_ERROR_EPSILON
    rel_valid = int(rel_mask.sum().item())
    if rel_valid > 0:
        rel_errs = errs[rel_mask] / reference.abs()[rel_mask]
        max_rel = float(rel_errs.max().item())
        mean_rel = float(rel_errs.mean().item())
        p99_rel = float(torch.quantile(rel_errs, 0.99).item())
    else:
        max_rel = mean_rel = p99_rel = float("nan")

    return NoiseMeasurement(
        op_name=op_name,
        module=module,
        distribution_name=distribution.name,
        distribution_description=distribution.description,
        n_samples=distribution.n_samples,
        max_abs_error=float(errs.max().item()),
        mean_abs_error=float(errs.mean().item()),
        p99_abs_error=float(torch.quantile(errs.flatten(), 0.99).item()),
        max_rel_error=max_rel,
        mean_rel_error=mean_rel,
        p99_rel_error=p99_rel,
        rel_valid_samples=rel_valid,
        worst_input=worst_input_flat,
        notes=notes,
        machine=machine,
    )


# ---------------------------------------------------------------------------
# Docstring footer patching
# ---------------------------------------------------------------------------


def _find_docstring_span(source: str, func_name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            if not node.body:
                continue
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                assert first.end_lineno is not None
                return first.lineno, first.end_lineno
    raise LookupError(f"no docstring found for function {func_name!r}")


def render_footer_block(
    *,
    max_abs_error: float,
    max_rel_error: float,
    total_samples: int,
    commit: str,
    indent: str,
) -> List[str]:
    """Return the lines (no trailing newlines) of a noise-footer block.

    The footer is deliberately distribution-agnostic: it reports the worst
    observed abs and rel error across every distribution the op was measured
    on. Per-distribution context lives in ``docs/numerical_noise.md``, not
    in the op's docstring.
    """
    body = indent + "   "
    rel_txt = "n/a" if _is_nan(max_rel_error) else _fmt(max_rel_error)
    return [
        "",
        f"{indent}{NOISE_FOOTER_MARKER}",
        "",
        f"{body}Max error: {_fmt(max_abs_error)} abs, {rel_txt} rel "
        f"over {total_samples} samples;",
        f"{body}measured at commit {commit}. See docs/numerical_noise.md.",
    ]


def update_docstring_footer(
    source: str,
    func_name: str,
    *,
    max_abs_error: float,
    max_rel_error: float,
    total_samples: int,
    commit: str,
) -> str:
    """Insert or replace the noise-footer block in ``func_name``'s docstring.

    Idempotent: running twice at the same commit is a no-op. The marker line
    ``.. noise-footer::`` delimits the block; everything from the marker to
    the docstring's closing ``\"\"\"`` is replaced.
    """
    start, end = _find_docstring_span(source, func_name)
    lines = source.split("\n")
    doc_lines = lines[start - 1 : end]
    if not doc_lines:
        raise LookupError(f"empty docstring for {func_name!r}")

    closing = doc_lines[-1]
    indent = " " * (len(closing) - len(closing.lstrip()))

    marker_idx: Optional[int] = None
    for i, line in enumerate(doc_lines):
        if line.strip() == NOISE_FOOTER_MARKER:
            marker_idx = i
            break

    if marker_idx is not None:
        trim = marker_idx
        while trim > 0 and doc_lines[trim - 1].strip() == "":
            trim -= 1
        doc_lines = doc_lines[:trim] + [closing]

    footer = render_footer_block(
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        total_samples=total_samples,
        commit=commit,
        indent=indent,
    )
    new_doc = doc_lines[:-1] + footer + [doc_lines[-1]]
    new_lines = lines[: start - 1] + new_doc + lines[end:]
    return "\n".join(new_lines)


def _fmt(x: float) -> str:
    return f"{x:.4g}"


def _is_nan(x: float) -> bool:
    return x != x
