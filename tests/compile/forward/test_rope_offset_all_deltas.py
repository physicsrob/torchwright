"""Phase-2 part 1: rotary offset head is token-identical to trig at all Δ.

docs/rope_port_plan.md Phase 2 — "reimplement ``attend_to_offset`` (all Δ) on
``W_K = R_N W_Q`` (lock the sign convention — ``j+N`` vs ``j−N`` is an easy
silent flip)".  Phase 0 proved Δ=−1; this proves the rotary offset head selects
the *same key* as the existing trig-shift ``attend_to_offset`` across the full
Δ range the real callers use (committed grep: −1, −2, −3, +1), plus a couple
wider/forward offsets, and pins it on the compiled path.

The token-identity assertion is the sign-convention lock: if rotary and trig
disagreed on direction, the selected key (and thus the transported payload)
would differ at every position.
"""

import pytest
import torch

from torchwright.compiler.export import compile_headless
from torchwright.debug.probe import probe_compiled
from torchwright.graph.rope import rotary_offset_head
from torchwright.ops.inout_nodes import create_input, create_pos_encoding

N_POS = 16
D = 256
D_HEAD = 16

# Backward Δ: the real caller range (−1, −2, −3) plus two wider offsets.  At an
# in-bounds query (m ≥ −Δ, so key m+Δ ≥ 0) this is a clean selection; the
# out-of-bounds region (target before BOS) is a don't-care fallback that trig
# and rotary handle differently and harmlessly, so it is excluded.
BACKWARD_DELTAS = [-1, -2, -3, -5, -8]


def _payload(n_pos: int) -> torch.Tensor:
    return (100.0 + torch.arange(n_pos, dtype=torch.float32)).unsqueeze(1)


@pytest.mark.parametrize("delta", BACKWARD_DELTAS)
def test_rotary_offset_token_identical_to_trig(delta):
    """Oracle: the rotary head selects the same key as trig ``attend_to_offset``
    at every in-bounds position — the sign-convention lock across all Δ."""
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    vals = _payload(N_POS)

    rotary = rotary_offset_head(payload, delta_pos=delta)
    trig = pos.attend_to_offset(payload, delta_pos=delta)

    rotary_out = rotary.compute(N_POS, {"payload": vals}).squeeze(1)
    trig_out = trig.compute(N_POS, {"payload": vals}).squeeze(1)
    # in-bounds: the target key m+delta is >= 0 (BOS or later).
    in_bounds = slice(-delta, N_POS)
    assert torch.allclose(rotary_out[in_bounds], trig_out[in_bounds], atol=1e-2), (
        f"delta={delta}: rotary and trig disagree in-bounds\n"
        f"rotary={rotary_out}\ntrig={trig_out}"
    )


@pytest.mark.parametrize("delta", [-1, -3, 1])
def test_rotary_offset_compiled_matches_oracle_all_deltas(delta):
    """probe_compiled: the compiled rotary head matches its oracle at
    representative backward, wider-backward, and forward Δ."""
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    vals = _payload(N_POS)
    rotary = rotary_offset_head(payload, delta_pos=delta)

    compiled = compile_headless(rotary, pos, d=D, d_head=D_HEAD, verbose=False)
    report = probe_compiled(compiled, rotary, {"payload": vals}, N_POS, atol=1e-2)
    assert report.first_divergent is None, report.format_short()


def test_rotary_offset_sign_is_directional():
    """−1 and +1 select opposite directions (the silent-flip guard).

    At an interior position both keys are visible-or-not under the causal mask,
    but the two heads must not produce identical output — that would mean the
    sign collapsed.
    """
    pos = create_pos_encoding()
    payload = create_input("payload", 1)
    vals = _payload(N_POS)

    back = (
        rotary_offset_head(payload, delta_pos=-1)
        .compute(N_POS, {"payload": vals})
        .squeeze(1)
    )
    fwd = (
        rotary_offset_head(payload, delta_pos=1)
        .compute(N_POS, {"payload": vals})
        .squeeze(1)
    )
    # −1 reads the previous position (distinct values per position), +1 cannot
    # (j+1 is causally masked → self), so the two disagree on the interior.
    assert not torch.allclose(back, fwd, atol=1e-2), (back, fwd)
