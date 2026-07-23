import torch

from torchwright.graph import Node
from torchwright.graph.rope import ROPE_BASE, apply_rope, rope_cos_sin

# Causal mask sentinel: future positions are filled with this value before
# softmax.  Must be large enough that no valid logit ever falls below it,
# otherwise the softmax will prefer "hidden" future positions over the
# real current position.  With _QUERY_GAIN = 8, _VALIDITY_DIRECT = 1000,
# _UNMASKED_PENALTY = 1000, and |score| up to 120, the worst valid
# logit is around −3000, still far above −1e6.  Retained at −1e6 so
# that existing oracle/compiled parity is preserved.
CAUSAL_MASK_SENTINEL = -1e6


def _assert_no_dead_children(
    query_in: Node,
    key_in: Node,
    value_in: Node,
    query_matrix: torch.Tensor,
    key_matrix: torch.Tensor,
    value_matrix: torch.Tensor,
) -> None:
    """Catch Concatenate children that contribute nothing to any projection.

    For each Concatenate input, walk its flattened children. A child is
    "dead" if its row slab is all-zero in every matrix whose input shares
    this Concatenate — i.e. nothing reads it on any of query/key/value.
    When Q, K, and V share the same residual-stream Concatenate (a common
    pattern), rows read by only one matrix are still live because the
    other matrices' zero rows represent "this matrix doesn't care",
    not "nothing cares".
    """
    from torchwright.graph import Concatenate

    roles = [
        ("query", query_in, query_matrix),
        ("key", key_in, key_matrix),
        ("value", value_in, value_matrix),
    ]

    seen: set[int] = set()
    for _role, input_node, _matrix in roles:
        if id(input_node) in seen or not isinstance(input_node, Concatenate):
            continue
        seen.add(id(input_node))

        shared = [(r, m) for (r, i, m) in roles if i is input_node]
        shared_roles = [r for r, _ in shared]

        row = 0
        for child in input_node.flatten_inputs():
            live = any(bool((m[row : row + len(child)] != 0).any()) for _r, m in shared)
            if not live:
                raise AssertionError(
                    f"Attn input shared across {shared_roles} has a dead "
                    f"child at rows {row}..{row + len(child)} "
                    f"({type(child).__name__}, width {len(child)}) — "
                    f"all-zero in "
                    f"{' and '.join(r + '_matrix' for r in shared_roles)}. "
                    f"Nothing reads it; remove it from the Concatenate."
                )
            row += len(child)


class Attn(Node):
    """Single causal attention head with explicit Q/K/V/O weight matrices.

    Computes causal (lower-triangular masked) attention:
    ``softmax(Q @ K^T, masked) @ V @ O``

    Inputs are three nodes: ``query_in``, ``key_in``, ``value_in``.
    """

    # query_matrix shape (d_query_in, d_qk)
    query_matrix: torch.Tensor

    # key_matrix shape (d_key_in, d_qk)
    key_matrix: torch.Tensor

    # value_matrix shape (d_value_in, d_v)
    value_matrix: torch.Tensor

    # output_matrix shape (d_v, d_output)
    output_matrix: torch.Tensor

    def __init__(
        self,
        query_in: Node,
        key_in: Node,
        value_in: Node,
        query_matrix: torch.Tensor,
        key_matrix: torch.Tensor,
        value_matrix: torch.Tensor,
        output_matrix: torch.Tensor,
        rope_base: float = ROPE_BASE,
        rope_d_rot: int | None = None,
    ):
        self.d_qk = query_matrix.shape[1]
        self.d_v = value_matrix.shape[1]
        self.d_query_in = query_matrix.shape[0]
        self.d_key_in = key_matrix.shape[0]
        self.d_value_in = value_matrix.shape[0]

        assert key_matrix.shape[1] == self.d_qk
        assert output_matrix.shape[0] == self.d_v

        _assert_no_dead_children(
            query_in,
            key_in,
            value_in,
            query_matrix,
            key_matrix,
            value_matrix,
        )

        # The rotary front d_rot is paired by rotate_half (dim p with p+d_rot/2),
        # so d_rot must be even.  d_rot defaults to d_qk (full rotary); the last
        # d_qk - d_rot dims are the unrotated NoPE tail (vanilla partial rotary).
        rope_d_rot = self.d_qk if rope_d_rot is None else rope_d_rot
        if rope_d_rot % 2 != 0 or not (0 < rope_d_rot <= self.d_qk):
            raise ValueError(
                f"Attn rope_d_rot must be even and in (0, d_qk={self.d_qk}]; got "
                f"{rope_d_rot}.  The rotary front is rotate_half over d_rot, pairing "
                f"dim p with dim p+d_rot/2."
            )

        self.query_matrix = query_matrix
        self.key_matrix = key_matrix
        self.value_matrix = value_matrix
        self.output_matrix = output_matrix
        self.rope_base = rope_base
        self.rope_d_rot = rope_d_rot
        super().__init__(output_matrix.shape[1], inputs=[query_in, key_in, value_in])

    def compute_value_type(self):
        from torchwright.graph.value_type import NodeValueType

        return NodeValueType()

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        query_in_node, key_in_node, value_in_node = self.inputs
        query_in = query_in_node.compute(n_pos, input_values)
        key_in = key_in_node.compute(n_pos, input_values)
        value_in = value_in_node.compute(n_pos, input_values)

        assert query_in.shape == (n_pos, self.d_query_in)
        assert key_in.shape == (n_pos, self.d_key_in)
        assert value_in.shape == (n_pos, self.d_value_in)

        key_values = torch.matmul(key_in, self.key_matrix)
        # key_values shape is (pos, d_qk)
        query_values = torch.matmul(query_in, self.query_matrix)
        # query_values shape is (pos, d_qk)

        # RoPE: rotate Q and K by absolute position (row index) before the
        # dot product, so the logit depends on the relative offset (i - j).
        # rotate_half over the rotary front d_rot (= d_qk for full rotary) on the
        # one global grid the compiled component and the ONNX/HF runtimes also
        # apply; the last d_qk - d_rot dims are the unrotated NoPE tail.
        positions = torch.arange(n_pos, device=query_values.device)
        cos, sin = rope_cos_sin(positions, self.rope_d_rot, self.rope_base)
        cos = cos.to(query_values.dtype)
        sin = sin.to(query_values.dtype)
        query_values = apply_rope(query_values, cos, sin)
        key_values = apply_rope(key_values, cos, sin)

        attn_logits = query_values.matmul(key_values.t())
        # attn_logits shape is (query pos, key pos)

        # Apply attention mask
        mask = torch.triu(torch.ones_like(attn_logits), diagonal=1)
        attn_logits = torch.where(
            mask == 1,
            torch.full_like(attn_logits, CAUSAL_MASK_SENTINEL),
            attn_logits,
        )

        attn = torch.softmax(attn_logits, dim=1)
        value_values = torch.matmul(value_in, self.value_matrix)
        # value_values shape is (pos, d_v)
        values = attn.matmul(value_values)
        # values shape is now (query pos, d_v)

        values_output = values.matmul(self.output_matrix)
        # values shape is now (query pos, d_output)
        return values_output

    def num_params(self):
        return (
            self.query_matrix.numel()
            + self.key_matrix.numel()
            + self.value_matrix.numel()
            + self.output_matrix.numel()
        )
