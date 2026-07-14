import pytest

from torchwright.compiler.forward.cpsat_scheduler import (
    Costs,
    ScheduleAssignment,
    choose_dominating_assignment,
)


def _assignment(layer: int, *, cancel: int = 2) -> ScheduleAssignment:
    return ScheduleAssignment(
        node_to_layer={7: layer},
        node_to_cancel_layer={7: cancel},
        node_to_routing={7: "attn"},
        node_to_cancel_mech={7: "attn"},
        n_layers=layer + 1,
    )


def test_assignment_defensively_freezes_and_canonicalizes_mappings():
    layers = {9: 1, 7: 0}
    assignment = ScheduleAssignment(
        node_to_layer=layers,
        node_to_cancel_layer={9: 2, 7: 1},
        node_to_routing={9: "mlp", 7: "attn"},
        n_layers=2,
    )
    layers[7] = 99
    assert assignment.node_to_layer[7] == 0
    assert tuple(assignment.node_to_layer) == (7, 9)
    with pytest.raises(TypeError):
        assignment.node_to_layer[7] = 1


def test_pure_depth_candidate_must_dominate_incumbent():
    incumbent = _assignment(0, cancel=1)
    worse = _assignment(2, cancel=3)
    assert choose_dominating_assignment(Costs(), incumbent, worse) is incumbent
    assert choose_dominating_assignment(Costs(), incumbent, None) is incumbent


def test_pure_depth_tie_uses_canonical_assignment_key():
    incumbent = _assignment(0, cancel=1)
    candidate = _assignment(0, cancel=0)
    chosen = choose_dominating_assignment(Costs(), incumbent, candidate)
    assert chosen.canonical_key() == min(
        incumbent.canonical_key(), candidate.canonical_key()
    )
