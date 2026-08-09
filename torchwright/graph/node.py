import dataclasses
import functools
import inspect
import math
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Optional, ParamSpec, TypeVar, cast

import torch

from torchwright.graph.value_type import NodeValueType, Range

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from torchwright.graph.affine_bound import AffineBound

global_node_id = 0


def reserve_node_id_above(nodes: "Iterable[Node]") -> None:
    """Advance ``global_node_id`` past every id in ``nodes``.

    ``Node.__eq__``/``__hash__`` key on ``node_id`` (see below), so two distinct
    nodes that happen to share an id compare equal and collide in any id-keyed
    dict/set — including the compiler's
    ``ResidualStreamMap._node_to_indices``.  In production the counter is
    monotonic and never reset, so a node minted *during* compile (e.g. the RoPE
    self-match constant-1 column in ``forward_compile``) is always born with an
    id strictly greater than every graph node and can never collide.  Tests
    reset the counter to 0 before each test for deterministic column assignment
    (see ``tests/conftest.py``); re-compiling a long-lived (module-scoped) graph
    then mints a helper node whose freshly-reset id collides with a graph node,
    silently overwriting that node's residual-map entry and orphaning its
    column (invariant I1 fires).  Calling this with the graph's full node set
    before minting any compile-internal node restores the monotonic invariant
    locally.  It only ever advances the counter, so it is a no-op whenever the
    counter is already ahead — which is every non-reset case, production
    included; it never touches the ids of the graph nodes themselves, so it
    cannot perturb their column assignment.
    """
    global global_node_id
    highest = max((n.node_id for n in nodes), default=-1)
    if global_node_id <= highest:
        global_node_id = highest + 1


def _verify_tensor_against_value_type(node: "Node", tensor: torch.Tensor) -> None:
    """Assert the actual tensor conforms to the node's declared value_type.

    Used by the optional runtime verifier (``TW_VERIFY_VALUE_TYPES``).
    All violations raise ``AssertionError``.
    """
    vt = node.value_type
    if tensor.numel() == 0:
        return
    t = tensor.detach()
    name = f"{node.node_type()}(id={node.node_id}, name='{node.name}')"

    r = vt.value_range
    actual_lo = float(t.min().item())
    actual_hi = float(t.max().item())
    tol = 1e-4
    if actual_lo < r.lo - tol or actual_hi > r.hi + tol:
        raise AssertionError(
            f"{name}: value_range mismatch — declared {r}, "
            f"observed [{actual_lo}, {actual_hi}]"
        )


_current_annotation: ContextVar[str | None] = ContextVar(
    "current_annotation",
    default=None,
)

# When True, ``compute`` skips running attached checks (``node.checks``).
# Used by callers that evaluate a graph on synthetic values — e.g. the
# univariate collapse pass seeds a subgraph's source with an offset grid
# whose "positions" are samples, not real data, so cross-position
# predicates (distinct-across, score gaps) would fire spuriously.
_checks_suppressed: ContextVar[bool] = ContextVar("checks_suppressed", default=False)


@contextmanager
def suppress_checks() -> "Iterator[None]":
    """Disable attached-check execution (``node.checks``) inside the block.

    Claims and bounds are unaffected — only the runtime predicates are
    skipped.  For evaluating graphs on synthetic inputs where the
    predicates' preconditions don't hold.
    """
    token = _checks_suppressed.set(True)
    try:
        yield
    finally:
        _checks_suppressed.reset(token)


@contextmanager
def annotate(label: str) -> "Iterator[None]":
    """Tag all nodes created inside this block with a hierarchical label.

    Nesting builds a ``/``-separated path. A component already active in the
    path is NOT repeated — so sibling functions in one subsystem calling each
    other (a natural pattern) stay ``a/b`` instead of accumulating
    ``a/b/a/b``::

        with annotate("render"):
            with annotate("texture"):
                # nodes get annotation = "render/texture"
            with annotate("render"):
                # still "render" — re-entering an active scope is a no-op

    ``label`` may itself be a ``/``-separated path (e.g. ``"pmrk/R_CheckPlane"``);
    each component is deduped independently.
    """
    current = _current_annotation.get()
    parts = current.split("/") if current else []
    seen = set(parts)
    for comp in label.split("/"):
        if comp and comp not in seen:
            parts.append(comp)
            seen.add(comp)
    new = "/".join(parts) if parts else current
    token = _current_annotation.set(new)
    try:
        yield
    finally:
        _current_annotation.reset(token)


_P = ParamSpec("_P")
_R = TypeVar("_R")


def annotated(label: str) -> "Callable[[Callable[_P, _R]], Callable[_P, _R]]":
    """Decorator form of :func:`annotate`.

    Tags every node created during the wrapped function's execution with
    ``label`` (hierarchical, same nesting rules as :func:`annotate`)::

        @annotated("wall projection")
        def project_segs(...):
            ...

    Equivalent to wrapping the whole body in ``with annotate(label):`` but
    without re-indenting it.
    """

    def deco(fn: "Callable[_P, _R]") -> "Callable[_P, _R]":
        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with annotate(label):
                return fn(*args, **kwargs)

        return wrapper

    return deco


@dataclasses.dataclass(eq=False)
class OpScopeRecord:
    """Construction-time provenance for one op-library call.

    Created by :func:`op_scope` on every decorated call — there is no
    dedup (unlike :func:`annotate`): two identical calls produce two
    records, because each call is a distinct op instance.  ``eq=False``
    keeps identity hashing, so records can key the capture dicts the
    schematic builds from them.

    Records hold ``node_id`` integers only, never ``Node`` references:
    the clone pass's stray-reference guard
    (``graph_clone._iter_stray_node_refs``) cannot see inside dataclass
    fields, so a ``Node``-typed field here would silently leak source
    references into compiler-private clones.

    Attribution rules: a node's membership (``Node.op_region``) is the
    innermost record active at its construction, stamped once in
    ``Node.__init__`` and never overwritten — a node returned unchanged
    through an outer decorated call appears in that outer record's
    ``result_ids`` but keeps its creator's membership.  Consumers read
    ``operand_ids``/``result_ids`` for dataflow, not membership.
    """

    #: Derived op name: module path under ``torchwright.ops`` plus the
    #: function qualname (``linear.add``, ``relu.arithmetic_ops.compare``).
    op: str
    #: JSON-safe non-Node arguments by parameter name (defaults included).
    params: dict[str, Any]
    #: Enclosing decorated call at entry, or None at top level.
    parent: "OpScopeRecord | None"
    #: Ambient :func:`annotate` path at call entry.
    annotation: str | None
    #: ``node_id`` of every Node-typed argument, in signature order.
    operand_ids: tuple[int, ...]
    #: ``node_id`` of every Node-typed return value (set after the call).
    result_ids: tuple[int, ...] = ()


_current_op_scope: ContextVar[OpScopeRecord | None] = ContextVar(
    "current_op_scope",
    default=None,
)

# Bounds for _sanitize_op_params: container elements beyond the cap (and
# nesting beyond the depth cap) collapse to a summary string, so DOOM-scale
# arguments (vocab lists, breakpoint grids) cannot bloat the record.  The
# element cap follows schematic_capture._INLINE_LITERAL_LIMIT's "small enough to keep
# inline" stance.
_PARAM_ELEMENT_CAP = 64
_PARAM_DEPTH_CAP = 3


def _scalar_param(value: object) -> object:
    """Encode a scalar; non-finite floats become strings.

    The string encoding is mandatory, not cosmetic: the schematic is
    dumped with ``allow_nan=False``, so a raw ``inf`` param would crash
    manifest emission far from the op call that captured it.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "+inf" if value > 0 else "-inf"
    return value


def _frozen_dataclass_fields(value: object) -> dict[str, object] | None:
    """Shallow field dict for a frozen dataclass instance, else None.

    Deliberately not ``dataclasses.asdict`` — that deep-copies tensor
    fields, and records live as long as the graph.
    """
    if not dataclasses.is_dataclass(value) or isinstance(value, type):
        return None
    params = getattr(type(value), "__dataclass_params__", None)
    if params is None or not params.frozen:
        return None
    return {
        field.name: getattr(value, field.name) for field in dataclasses.fields(value)
    }


def _sanitize_op_params(value: object, depth: int = 0) -> object:
    """Encode one op argument as JSON-safe data — total, recursive, capped.

    Every branch returns something ``json.dumps(..., allow_nan=False)``
    accepts.  Tensors are summarized by shape/dtype and never retained.
    Containers recurse up to the element and depth caps; oversized,
    non-string-keyed, or unrecognized values collapse to a summary
    string.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return _scalar_param(value)
    if isinstance(value, torch.Tensor):
        return {
            "tensor": {
                "shape": list(value.shape),
                "dtype": str(value.dtype).removeprefix("torch."),
            }
        }
    if depth < _PARAM_DEPTH_CAP:
        if isinstance(value, (list, tuple)) and len(value) <= _PARAM_ELEMENT_CAP:
            return [_sanitize_op_params(item, depth + 1) for item in value]
        if (
            isinstance(value, dict)
            and len(value) <= _PARAM_ELEMENT_CAP
            and all(isinstance(key, str) for key in value)
        ):
            return {
                key: _sanitize_op_params(item, depth + 1) for key, item in value.items()
            }
        fields = _frozen_dataclass_fields(value)
        if fields is not None:
            return {
                name: _sanitize_op_params(item, depth + 1)
                for name, item in fields.items()
            }
    size = f"[{len(value)}]" if isinstance(value, (list, tuple, dict)) else ""
    return f"<{type(value).__name__}{size}>"


def _node_ids_in(value: object) -> list[int] | None:
    """Node ids in ``value`` (one container level deep), or None if none.

    The one-level depth matches the clone pass's stray-reference walk
    (``graph_clone._iter_stray_node_refs``): an op argument that buries
    nodes deeper than one list/tuple level is invisible to both.
    """
    if isinstance(value, Node):
        return [value.node_id]
    if isinstance(value, (list, tuple)):
        ids = [item.node_id for item in value if isinstance(item, Node)]
        if ids:
            return ids
    return None


def op_scope(fn: "Callable[_P, _R]") -> "Callable[_P, _R]":
    """Stamp every node created inside ``fn`` with op-call provenance.

    Bare decorator for op-library constructors.  The op name is derived
    from the function's module and qualname (``linear.add``,
    ``swiglu.map_select.select``), so twin-machine names never collide
    and there is no hand-written label to drift.  Each call captures a
    fresh :class:`OpScopeRecord`: ``Node``-typed arguments (scanned one
    container level deep) become ``operand_ids``, every other argument is
    sanitized into ``params`` (defaults included; a parameter named
    ``self`` is skipped, so methods decorate cleanly), and ``Node``-typed
    returns are recorded as ``result_ids`` after the call.  Nested
    decorated calls chain through ``parent``.

    The record is pure metadata: it does not participate in canonical
    ids, fingerprints, or the schedule cache (the same neutrality
    contract ``node.checks`` has).
    """
    signature = inspect.signature(fn)
    module = fn.__module__.removeprefix("torchwright.ops.")
    op_name = f"{module}.{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        operand_ids: list[int] = []
        params: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            if name == "self":
                continue
            node_ids = _node_ids_in(value)
            if node_ids is not None:
                operand_ids.extend(node_ids)
            else:
                params[name] = _sanitize_op_params(value)
        record = OpScopeRecord(
            op=op_name,
            params=params,
            parent=_current_op_scope.get(),
            annotation=_current_annotation.get(),
            operand_ids=tuple(operand_ids),
        )
        token = _current_op_scope.set(record)
        try:
            result = fn(*args, **kwargs)
        finally:
            _current_op_scope.reset(token)
        record.result_ids = tuple(_node_ids_in(result) or ())
        return result

    return wrapper


class Node:
    """Base class for all computation graph nodes.

    Each node represents one operation in a dataflow graph that will be
    compiled into transformer weights. Nodes are connected by their
    ``inputs`` list (upstream dependencies).

    Attributes:
        d_output: Width of this node's output vector.
        inputs: Upstream nodes whose outputs feed into this node.
        node_id: Auto-incremented unique identifier.
        name: Optional human-readable label (for debugging / repr).
        annotation: Hierarchical label set by the ``annotate`` context manager.
        op_region: Innermost :func:`op_scope` record active at construction
            (None outside any decorated op call).  Stamped once here and
            never overwritten — a node returned through an outer decorated
            call keeps its creator's membership.  Pure metadata, like
            ``checks``: invisible to canonical ids, fingerprints, and the
            schedule cache.
        checks: Runtime predicates attached to this node's value
            (``torchwright.graph.misc.Check``); run during ``compute``
            and against compiled values on ``debug=True`` forwards.
            Attach via the helpers in ``torchwright.graph.asserts``.
        claimed_type: Running intersection of every range claim attached
            to this node (claims commute).  Applied to the derived
            caches by ``refresh_node_caches`` — the single choke point —
            so claims survive any bounds recompute by construction.
        integer_claim: True when an ``assert_integer`` claim is attached
            (the value is integer-grained).  ``NodeValueType`` stays
            range-only, so integer-ness travels here; the univariate
            collapse pass gates on it.
    """

    inputs: list["Node"]
    d_output: int
    node_id: int
    name: str
    annotation: str | None
    op_region: OpScopeRecord | None
    scheduling_predecessors: set["Node"]
    checks: list
    claimed_type: NodeValueType | None
    integer_claim: bool
    # Derived caches, set eagerly in ``__init__`` and recomputed only via
    # ``affine_rules.refresh_node_caches`` (the claim-application choke point).
    _structural_type: NodeValueType
    _affine_bound: "AffineBound"
    # Semantic affine override installed by an op via
    # ``affine_rules._apply_semantic_override`` (None for most nodes).
    # Public: read and written by the optimizer folds and the graph
    # cloner, not just the installing op.
    semantic_affine_override: Optional["AffineBound"]
    # Weight-support cache attached lazily by
    # ``compiler.realization.live_weight_row_ranges`` (and injected directly
    # onto CP-SAT snapshot stand-in nodes); absent until first query.
    _live_weight_row_ranges: tuple[tuple[int, int], ...]

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        original = cls.__dict__.get("compute")
        if original is None or getattr(original, "tw_verified", False):
            return

        def wrapped(
            self: "Node", n_pos: int, input_values: dict, *args: object, **kw: object
        ) -> torch.Tensor:
            result = original(self, n_pos, input_values, *args, **kw)
            if os.environ.get("TW_VERIFY_VALUE_TYPES") and isinstance(
                result, torch.Tensor
            ):
                _verify_tensor_against_value_type(self, result)
            # Attached checks run on every compute of the node's value —
            # this is what makes reference eval exercise every assert
            # (memoised evaluators call compute once per node, so each
            # check runs once per eval).
            if (
                self.checks
                and not _checks_suppressed.get()
                and isinstance(result, torch.Tensor)
            ):
                for check in self.checks:
                    check.run(result, self)
            return result

        wrapped.__name__ = original.__name__
        wrapped.__qualname__ = original.__qualname__
        wrapped.__doc__ = original.__doc__
        cast("Any", wrapped).tw_verified = True
        type.__setattr__(cls, "compute", wrapped)

    def __init__(self, d_output: int, inputs: list["Node"], name: str = "") -> None:
        global global_node_id
        self.d_output = d_output
        self.inputs = inputs
        self.node_id = global_node_id
        self.name = name
        self.annotation = _current_annotation.get()
        self.op_region = _current_op_scope.get()
        # Scheduling-only predecessors (not data inputs).  Populated by
        # ``torchwright.graph.scheduling_hints.sequential_scope`` and
        # similar helpers; honored by ``GraphAnalyzer.is_ready`` so the
        # node isn't scheduled until every listed predecessor is in
        # ``computed_nodes``.  Empty by default.
        self.scheduling_predecessors: set[Node] = set()
        global_node_id += 1
        # Semantic affine override installed by an op via
        # ``affine_rules._apply_semantic_override`` (None for most nodes).
        # Stored persistently so ``refresh_node_caches`` can re-apply it
        # after recomputing the propagated bound — a recompute that dropped
        # it would silently loosen every downstream bound.
        self.semantic_affine_override = None
        # Check/claim metadata (see class docstring).  Empty at birth;
        # the asserts.py helpers attach after construction.
        self.checks = []
        self.claimed_type = None
        self.integer_claim = False
        # Derived caches (_structural_type / _affine_bound) go through
        # refresh_node_caches — the same choke point every later
        # recompute uses, so claim application can never be skipped.
        from torchwright.graph.affine_rules import refresh_node_caches

        refresh_node_caches(self)

    @property
    def value_type(self) -> NodeValueType:
        r = self._affine_bound.to_scalar_range()
        sr = self._structural_type.value_range
        if sr.lo > r.lo or sr.hi < r.hi:
            # The structural type is tighter on at least one side; take the
            # intersection.  But the affine bound is float64 interval
            # arithmetic whose rounding can push a bound a few ULPs past the
            # structural range — e.g. a value structurally in [0, 1] whose
            # affine bound collapses to the point 1+2^-26.  That makes the two
            # *fp-disjoint* even though both soundly bound the same scalar, so
            # `Range.intersect` (which treats any crossing as a bug) would
            # raise.  Clamp the affine bound into the structural range instead,
            # guarded by a magnitude-scaled tolerance so a *gross* disjointness
            # (a genuinely unsound bound, not rounding) still raises.
            lo = max(r.lo, sr.lo)
            hi = min(r.hi, sr.hi)
            if lo > hi:
                scale = max(1.0, abs(sr.lo), abs(sr.hi), abs(r.lo), abs(r.hi))
                if lo - hi > 1e-6 * scale:
                    raise ValueError(
                        f"Affine bound {r} is disjoint from structural type "
                        f"{sr} by {lo - hi:.3e}, beyond fp-rounding noise; the "
                        f"affine analysis and the per-op value-type rule "
                        f"disagree, which indicates a real soundness bug."
                    )
                # fp-noise crossing: collapse onto the structural boundary
                # (the exact, authoritative bound) the affine bound rounded past.
                lo = hi = min(max(hi, sr.lo), sr.hi)
            r = Range(lo, hi)
        return NodeValueType(value_range=r)

    @property
    def affine_bound(self) -> "AffineBound":
        return self._affine_bound

    def compute_value_type(self) -> NodeValueType:
        """Return the static value-type of this node's output.

        Default: ``unknown()`` (fail-closed). Subclasses override to
        propagate properties from their inputs or declare constants.
        Called eagerly from ``__init__`` so rule errors surface at graph
        build time.
        """
        return NodeValueType.unknown()

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        return self.d_output

    def replace_input(self, old_input: "Node", new_input: "Node") -> None:
        for i, _ in enumerate(self.inputs):
            if self.inputs[i] == old_input:
                self.inputs[i] = new_input

    def node_type(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        type_name = self.node_type()
        if len(self.inputs) == 0:
            return f"{type_name}(id={self.node_id}, name='{self.name}', d={len(self)})"
        if len(self.inputs) == 1:
            inp = self.inputs[0]
            inp_type_name = inp.node_type()
            inp_repr = (
                f"{inp_type_name}(id={inp.node_id}, name='{inp.name}', d={len(inp)})"
            )
            return (
                f"{type_name}(id={self.node_id}, name='{self.name}', "
                f"inp={inp_repr}, d={len(self)})"
            )
        inp_strings = []
        for i, inp in enumerate(self.inputs):
            inp_type_name = inp.node_type()
            inp_strings.append(
                f"inp{i}={inp_type_name}(id={inp.node_id}, name='{inp.name}', "
                f"d={len(inp)})"
            )
        inp_str = ", ".join(inp_strings)
        return (
            f"{type_name}(id={self.node_id}, name='{self.name}', "
            f"{inp_str}, d={len(self)})"
        )

    def num_params(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return self.node_id == cast("Node", other).node_id

    def __hash__(self) -> int:
        return self.node_id

    def __deepcopy__(self, memo: dict) -> "Node":
        # Nodes are identity-bearing graph singletons (``__hash__``/``__eq__``
        # key on ``node_id``, and the scheduler distinguishes special nodes such
        # as ``pos_encoding`` by ``is`` identity).  Deep-copying a structure that
        # merely *references* nodes — e.g. cloning a ``ResidualStreamMap`` to
        # isolate the warm-start scheduler's mutations from the real map — must
        # preserve those references, not clone the graph.  Returning ``self``
        # keeps node identity intact while the surrounding container (dicts,
        # lists, sets) is still deep-copied normally.  Without this, the
        # warm-start's ``copy.deepcopy(residual_map)`` cloned ``pos_encoding``,
        # breaking the ``n is self.pos_encoding`` reservation check in
        # ``LayerScheduler`` so the warm-start freed pos early and produced a
        # hint the CP-SAT model (which reserves pos) rejected as infeasible.
        memo[id(self)] = self
        return self
