"""output_sequence emission gating — the short-prompt reproducer.

Slot ``i`` used to be gated by ``attend_to_offset(trigger, delta_pos=-i)``.
For a prompt shorter than the output length, the deep slots target a
position before BOS; with no real key to match, the sharp positional
softmax locks onto an arbitrary in-range key, and where that key reads
the trigger as true the deep slot's value is summed into the emission —
the output at the trigger position becomes the superposition
``seq[0] + seq[k]`` instead of ``seq[0]``.  The gating now rides the
near-marker step counter (``count_since_marker``), which is immune to
out-of-range aliasing; this test pins both the short-prompt fix and the
unchanged in-range emission walk, on both machines.
"""

import pytest
import torch

from torchwright.debug.probe import reference_eval
from torchwright.ops.inout_nodes import (
    create_embedding,
    create_literal_value,
    create_rope_config,
)
from torchwright.ops.relu.logic_ops import equals_vector as relu_equals_vector
from torchwright.ops.relu.sequence_ops import output_sequence as relu_output_sequence
from torchwright.ops.swiglu.logic_ops import equals_vector as swiglu_equals_vector
from torchwright.ops.swiglu.sequence_ops import (
    output_sequence as swiglu_output_sequence,
)

D_HEAD = 16
N_SLOTS = 6

_MACHINES = {
    "relu": (relu_output_sequence, relu_equals_vector),
    "swiglu": (swiglu_output_sequence, swiglu_equals_vector),
}


def _build(machine):
    """output_sequence emitting the digit embeddings 0..5 after "=" fires."""
    output_sequence, equals_vector = _MACHINES[machine]
    embedding = create_embedding(vocab=list("012345") + ["="])
    rope = create_rope_config(d_head=D_HEAD, max_positions=512)
    trigger = equals_vector(embedding, embedding.get_embedding("="))
    seq = [
        create_literal_value(embedding.get_embedding(str(i))) for i in range(N_SLOTS)
    ]
    default = embedding.get_embedding("=")
    out = output_sequence(rope, trigger, seq, default)
    return embedding, out, default


def _eval(embedding, out, tokens):
    tok = embedding.tokenizer
    ids = torch.tensor([[float(tok.get_token_id(t))] for t in tokens])
    cache = reference_eval(out, {embedding.input_name: ids}, len(tokens))
    return cache[out]


@pytest.mark.parametrize("machine", ["relu", "swiglu"])
def test_output_sequence_short_prompt_no_slot_leak(machine):
    """Trigger at position 2 with 6 slots: slots 3..5 target positions before
    BOS and must contribute nothing — the trigger position emits exactly
    seq[0], not a superposition.
    """
    embedding, out, default = _build(machine)
    values = _eval(embedding, out, ["0", "1", "="])
    torch.testing.assert_close(
        values[2], embedding.get_embedding("0"), atol=0.1, rtol=0.0
    )
    # Pre-trigger positions emit the default.
    for p in range(2):
        torch.testing.assert_close(values[p], default, atol=0.1, rtol=0.0)


@pytest.mark.parametrize("machine", ["relu", "swiglu"])
def test_output_sequence_in_range_walk(machine):
    """The default path is unchanged: from the trigger onward, position
    trigger+i emits seq[i].
    """
    embedding, out, _ = _build(machine)
    # Positions 2..5 emit seq[0..3] (the post-trigger input tokens are the
    # autoregressive echo of the emission; the gating ignores them).
    values = _eval(embedding, out, ["0", "1", "=", "0", "1", "2"])
    for i in range(4):
        torch.testing.assert_close(
            values[2 + i], embedding.get_embedding(str(i)), atol=0.1, rtol=0.0
        )
