from collections.abc import Callable
from dataclasses import dataclass

import torch

from torchwright.graph.node import Node
from torchwright.graph.value_type import NodeValueType

# A predicate maps a value tensor to ``(ok, detail)`` where ``detail`` is a
# short human-readable hint included in the assertion message when ``ok``
# is False.  Predicates receive the full ``(n_pos, d_output)`` tensor;
# position-gating (e.g. "only at WALL positions") is the *caller*'s
# responsibility — wrap the value in ``select(is_wall, ...)`` before
# asserting so the predicate operates on the already-gated tensor.
Predicate = Callable[[torch.Tensor], tuple[bool, str]]


@dataclass
class Check:
    """One runtime predicate attached to a node (``node.checks``).

    A check is a *fact about the node's value*: the predicate runs on the
    node's ``(n_pos, d_output)`` tensor during ``compute`` (reference
    eval) and against the node's compiled value during ``debug=True``
    forwards.  ``kind`` selects the failure behavior — ``"assert"``
    raises ``AssertionError``, ``"watch"`` prints.  ``annotation`` is the
    ambient ``annotate()`` path captured when the check was attached, so
    failure messages pinpoint the attach site.

    Checks hold no node references (the graph-clone stray-reference
    guard relies on this); predicate closures are shared by reference,
    like weight payloads.
    """

    predicate: Predicate
    message: str = ""
    kind: str = "assert"  # "assert" (raises) | "watch" (prints)
    annotation: str | None = None

    def run(self, x: torch.Tensor, node: Node) -> None:
        """Run the predicate on ``x``; raise (assert) or print (watch) on failure.

        Shared by the reference-eval path (``Node.compute`` runs every
        check on the freshly computed value) and the compiled path
        (``debug/extraction.check_debug_predicates`` runs them on
        residual-stream values) so both produce identical messages.
        """
        ok, detail = self.predicate(x)
        if ok:
            return
        site = self.annotation or node.annotation or f"node_{node.node_id}"
        if self.kind == "assert":
            msg_parts = [f"Assert failed at {site}"]
            if self.message:
                msg_parts.append(self.message)
            msg_parts.append(f"({detail})")
            raise AssertionError(": ".join(msg_parts[:-1]) + " " + msg_parts[-1])
        msg_parts = [f"DebugWatch at {site}"]
        if self.message:
            msg_parts.append(self.message)
        if detail:
            msg_parts.append(f"({detail})")
        print(": ".join(msg_parts))


class InputNode(Node):
    def __init__(
        self,
        d_output_or_name,
        d_output_or_nothing=None,
        name: str = "",
        *,
        value_range: tuple[float, float],
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
    def __init__(self, inputs: list[Node]):
        super().__init__(sum(len(x) for x in inputs), inputs)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        return torch.cat([x.compute(n_pos, input_values) for x in self.inputs], dim=-1)

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def flatten_inputs(self: Node) -> list[Node]:
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
        if self.inputs[1] == node:
            return self.inputs[0]
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


# Assert and DebugWatch wrapper nodes used to live here.  A runtime
# check is node *metadata* now — ``Node.checks`` / ``Node.claimed_type``
# (docs/assert_metadata_plan.md); attach with the helpers in
# ``torchwright.graph.asserts``.
