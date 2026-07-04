from typing import Callable, List, Dict, Optional, Tuple
from torchwright.graph import Node
from torchwright.graph.value_type import NodeValueType

import torch

# A predicate maps a value tensor to ``(ok, detail)`` where ``detail`` is a
# short human-readable hint included in the assertion message when ``ok``
# is False.  Predicates receive the full ``(n_pos, d_output)`` tensor;
# position-gating (e.g. "only at WALL positions") is the *caller*'s
# responsibility — wrap the value in ``select(is_wall, ...)`` before
# asserting so the predicate operates on the already-gated tensor.
Predicate = Callable[[torch.Tensor], Tuple[bool, str]]


class InputNode(Node):
    def __init__(
        self,
        d_output_or_name,
        d_output_or_nothing=None,
        name: str = "",
        *,
        value_range: Tuple[float, float],
    ):
        # Support both old and new constructor patterns:
        # - InputNode(d_output) - new anonymous pattern
        # - InputNode(d_output, name=name) - new named pattern
        # - InputNode(name, d_output) - legacy pattern
        if isinstance(d_output_or_name, str):
            # Legacy pattern: InputNode(name, d_output)
            if d_output_or_nothing is None:
                raise ValueError("d_output is required when name is the first argument")
            d_output = d_output_or_nothing
            name = d_output_or_name
        else:
            # New pattern: InputNode(d_output) or InputNode(d_output, name=name)
            d_output = d_output_or_name
            if d_output_or_nothing is not None and isinstance(d_output_or_nothing, str):
                name = d_output_or_nothing
        self._declared_value_range = value_range
        super().__init__(d_output, [], name=name)
        from torchwright.graph.session import current_session

        current_session().register_input(self)

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def compute(
        self, n_pos: int, input_values: dict, name: str | None = None
    ) -> torch.Tensor:
        # Use provided name or fall back to stored name
        lookup_name = name if name is not None else self.name
        if not lookup_name:
            raise ValueError(
                "InputNode has no name set and none was provided to compute()"
            )
        if lookup_name not in input_values:
            raise ValueError(f"Did not specify value for input variable {lookup_name}")

        val = input_values[lookup_name]
        assert isinstance(val, torch.Tensor)
        assert val.shape == (
            n_pos,
            self.d_output,
        ), "Input must be of shape (n_pos, d_output)"
        return val

    def __len__(self):
        return self.d_output


class Concatenate(Node):
    def __init__(self, inputs: List[Node]):
        super().__init__(sum(len(x) for x in inputs), inputs)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.cat([x.compute(n_pos, input_values) for x in self.inputs], dim=-1)

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def flatten_inputs(self: Node) -> List[Node]:
        # Flatten concatenation and return the list of nodes
        inputs = []
        for n in self.inputs:
            if isinstance(n, Concatenate):
                inputs += n.flatten_inputs()
            else:
                inputs.append(n)
        return inputs


class Add(Node):
    def __init__(self, input1: Node, input2: Node, name: str = ""):
        super().__init__(len(input1), [input1, input2], name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        input1, input2 = self.inputs
        return input1.compute(n_pos, input_values) + input2.compute(n_pos, input_values)

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def other_input(self, node: Node):
        if self.inputs[0] == node:
            return self.inputs[1]
        elif self.inputs[1] == node:
            return self.inputs[0]
        else:
            raise ValueError("Invalid node choice")


class LiteralValue(Node):
    def __init__(self, value: torch.Tensor, name: str = ""):
        assert len(value.shape) == 1
        self.value = value
        super().__init__(len(value), [], name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.value.unsqueeze(0).expand(n_pos, -1)
        assert x.shape == (n_pos, len(self))
        return x

    def compute_value_type(self) -> NodeValueType:
        from torchwright.graph.value_type import Range

        v = self.value
        if v.numel() == 0:
            return NodeValueType()
        lo = float(v.min().item())
        hi = float(v.max().item())
        return NodeValueType(value_range=Range(lo, hi))

    def is_zero(self):
        return self.value.eq(0).all()

    def node_type(self):
        if self.is_zero():
            return "Zero"
        else:
            return "LiteralValue"


class Placeholder(Node):
    """Zero-width sentinel used as a stand-in when a real node is not yet available."""

    def __init__(self, d: int = 0):
        super().__init__(d, [])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.zeros(self.d_output)


class ValueLogger(Node):
    def __init__(self, inp: Node, name: str):
        self.name = name
        super().__init__(len(inp), [inp])

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        inp = self.inputs[0]
        x = inp.compute(n_pos, input_values)
        print(f"ValueLogger({self.name}): shape={x.shape} value={x}")
        return x

    def compute_value_type(self) -> NodeValueType:
        return self.inputs[0].value_type


class Assert(Node):
    """Pass-through node that validates its input's value against a predicate.

    Invisible to the compiler — ``lower()`` strips wrappers from the
    compiler-private copy (the source graph keeps them), so compiled
    transformer weights are identical with or without Asserts.  During
    reference evaluation (``reference_eval``) and compiled-graph probing
    (``probe_compiled``), runs the predicate on the input value and
    raises ``AssertionError`` on rejection.

    The raised message incorporates this Assert's ``annotation`` (set by
    the surrounding ``annotate()`` context manager) plus the predicate's
    ``detail`` string, so failures pinpoint both the site and the value.

    When ``claimed_type`` is supplied, the Assert also *promotes* the
    wrapped node's static type: downstream graph analysis sees the
    claimed ``NodeValueType`` on this Assert's output, and the lowering
    strip tightens the copied wrapped node's ``value_type`` to the
    intersection of its own inferred type and this claim.  The runtime
    predicate is the safety net that keeps the claim honest.
    """

    def __init__(
        self,
        inp: Node,
        predicate: Predicate,
        message: str = "",
        claimed_type: Optional[NodeValueType] = None,
    ):
        self.predicate = predicate
        self.message = message
        # Stashed for compute_value_type (runs inside super().__init__).
        self._claimed_type = claimed_type
        super().__init__(len(inp), [inp])

    @property
    def claimed_type(self) -> Optional[NodeValueType]:
        return self._claimed_type

    def compute_value_type(self) -> NodeValueType:
        if self._claimed_type is not None:
            return NodeValueType()
        return self.inputs[0].value_type

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.inputs[0].compute(n_pos, input_values)
        self._check(x)
        return x

    def _check(self, x: torch.Tensor) -> None:
        """Run the predicate; raise AssertionError with context on failure.

        Shared by ``compute`` (reference-eval path) and ``probe_compiled``
        (compiled-graph path) so both paths produce identical failure
        messages.
        """
        ok, detail = self.predicate(x)
        if not ok:
            site = self.annotation or f"node_{self.node_id}"
            msg_parts = [f"Assert failed at {site}"]
            if self.message:
                msg_parts.append(self.message)
            msg_parts.append(f"({detail})")
            raise AssertionError(": ".join(msg_parts[:-1]) + " " + msg_parts[-1])


class DebugWatch(Node):
    """Pass-through node that prints when a predicate fires.

    Like Assert but observational: prints instead of raising.  Stripped
    from the compiler-private copy alongside Assert nodes (the source
    graph keeps both).  Exercised during both ``compute()`` (oracle
    path) and compiled debug forward when ``debug=True``.
    """

    def __init__(self, inp: Node, predicate: Predicate, message: str = ""):
        self.predicate = predicate
        self.message = message
        super().__init__(len(inp), [inp])

    def compute_value_type(self) -> NodeValueType:
        return self.inputs[0].value_type

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        x = self.inputs[0].compute(n_pos, input_values)
        self._check(x)
        return x

    def _check(self, x: torch.Tensor) -> None:
        ok, detail = self.predicate(x)
        if not ok:
            site = self.annotation or f"node_{self.node_id}"
            msg_parts = [f"DebugWatch at {site}"]
            if self.message:
                msg_parts.append(self.message)
            if detail:
                msg_parts.append(f"({detail})")
            print(": ".join(msg_parts))
