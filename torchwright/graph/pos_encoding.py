from torchwright.graph import Node, Concatenate, Attn, LiteralValue
from torchwright.graph.value_type import NodeValueType
import torch
import math

attention_hardness = 100.0  # Scales attention to be 1.0 and 0.0 everywhere else


class PosEncoding(Node):
    """Sinusoidal positional encoding plus a raw integer position counter.

    Column layout (``d_pos`` must be odd):

    - columns ``0 .. d_pos-2`` (an even count, ``trig_width``): complete
      sin/cos pairs.
    - column ``d_pos-1`` (``counter_col``): the raw integer position
      ``0, 1, 2, …``.

    Keeping the counter in its own column leaves a full block of sin/cos
    pairs for the trig-identity shift (:func:`trig_shift_matrix`,
    :meth:`attend_to_offset`) and gives an exact position via one Linear
    extraction (:meth:`get_position_scalar`).
    """

    def __init__(self, d_pos: int):
        if d_pos % 2 == 0:
            raise ValueError(
                f"PosEncoding requires an odd d_pos (got {d_pos}). The trig "
                f"block occupies the even-width columns 0..d_pos-2 and the "
                f"raw position counter occupies the last column d_pos-1; an "
                f"even d_pos would leave an incomplete sin/cos pair. Use "
                f"{d_pos + 1} or {d_pos - 1}."
            )
        self.d_pos = d_pos
        super().__init__(d_pos, [])

    # --- Layout -----------------------------------------------------------

    @property
    def trig_width(self) -> int:
        """Number of sin/cos columns (an even count): columns ``0..d_pos-2``."""
        return self.d_pos - 1

    @property
    def trig_slice(self) -> slice:
        """Slice covering the sin/cos columns."""
        return slice(0, self.d_pos - 1)

    @property
    def counter_col(self) -> int:
        """Column holding the raw integer position counter (the last column)."""
        return self.d_pos - 1

    # --- Encoding ---------------------------------------------------------

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def get_pos_encoding(self, n_pos: int):
        pe = torch.zeros((n_pos, self.d_pos))
        tw = self.trig_width
        # Frequencies built from trig_width (not d_pos) so the grid is
        # independent of the extra counter column.
        div_term = torch.exp(torch.arange(0, tw, 2) * -(math.log(10000.0) / tw))
        for pos in range(n_pos):
            pe[pos, 0:tw:2] = torch.sin(pos * div_term)
            pe[pos, 1:tw:2] = torch.cos(pos * div_term)
        # Raw integer position counter in its own (last) column.  One Linear
        # extraction gives exact position indices — no asin inversion.
        pe[:, self.counter_col] = torch.arange(n_pos, dtype=pe.dtype)
        return pe

    def compute(self, n_pos: int, input_values: dict):
        return self.get_pos_encoding(n_pos)

    def get_position_scalar(self) -> Node:
        """Recover the position index as a 1-D scalar node.

        Extracts the raw integer counter at ``counter_col``.  Exact for all
        positions.
        """
        from torchwright.graph.linear import Linear

        weight = torch.zeros(self.d_pos, 1)
        weight[self.counter_col, 0] = 1.0
        return Linear(self, weight, name="position_scalar")

    def attend_to_offset(self, value: Node, delta_pos=-1) -> Node:
        if delta_pos == 0:
            # NOOP -- supporting this simplifies some use cases.
            return value

        tw = self.trig_width
        d_v = len(value)

        # Q/K match positions over the trig block only (d_qk == trig_width);
        # the raw counter never participates in trig matching.
        query_matrix = torch.zeros((len(self), tw))
        query_matrix[self.trig_slice, :] = attention_hardness * torch.eye(tw)

        # Shift the key's trig block by -delta_pos so key position i presents
        # trig(i - delta_pos).  The logit peaks when i - delta_pos == j, i.e.
        # i == j + delta_pos (delta_pos = -1 -> the previous position).
        key_matrix = torch.zeros((len(self), tw))
        key_matrix[self.trig_slice, :] = trig_shift_matrix(-delta_pos, tw)

        # V/O transport the payload independently; values wider than the
        # encoding are split across physical heads by the compiler.
        value_matrix = torch.eye(d_v)
        output_matrix = torch.eye(d_v)

        return Attn(
            query_in=self,
            key_in=self,
            value_in=value,
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=value_matrix,
            output_matrix=output_matrix,
        )

    def get_prev_value(
        self,
        value: Node,
        cond: Node,
        *,
        max_pos: int = 256,
        recency_logit_gap: float = 8.0,
    ) -> Node:
        """Most-recent previous ``value`` at a position where ``cond`` is true.

        For each query position, selects ``value`` at the most recent causal
        position where ``cond == 1.0`` (false positions must be ``<= 0``).
        Recency is ranked by the raw position counter: among true positions
        the largest counter (most recent) wins by ``recency_logit_gap`` of
        logit per position.  A condition gate that dominates the full recency
        span makes any true key beat any false key.

        Numerical limit.  The gate forces peak logits near
        ``3 * recency_logit_gap * max_pos``.  On A100 TF32 (~1e-3 relative
        precision) the per-position recency gap stays resolvable only while
        ``max_pos`` is a few hundred; the default 256 is safe, ~512 is the
        ceiling.  Larger reach needs an fp32 matmul for this head.  Runtime
        positions must be ``<= max_pos``.
        """
        assert len(cond) == 1, "get_prev_value expects a 1-D boolean cond"
        d_v = len(value)

        # Key columns: [cond, the positional encoding].  Only the cond row
        # and the encoding's counter column are read; the rest of the
        # encoding is inert here.
        key_in = Concatenate([cond, self])

        # Gate must exceed the full recency span (recency_logit_gap * max_pos)
        # so any true key beats any false key regardless of position.  The
        # factor of two keeps a wide margin above the span while leaving the
        # peak logit (~3 * recency_logit_gap * max_pos) within the TF32 range
        # where the per-position recency gap still resolves.
        gate = 2.0 * recency_logit_gap * max_pos

        # Query is the exact constant 1.0, projected to [gate, recency_gap].
        query_one = LiteralValue(torch.tensor([1.0]), name="prev_value_query_one")
        query_matrix = torch.tensor([[gate, recency_logit_gap]])  # (1, 2)

        # logit(j, i) = gate * cond_i + recency_logit_gap * counter_i.
        key_matrix = torch.zeros((len(key_in), 2))
        key_matrix[0, 0] = 1.0  # cond -> gate column
        key_matrix[1 + self.counter_col, 1] = 1.0  # counter -> recency column

        value_matrix = torch.eye(d_v)
        output_matrix = torch.eye(d_v)

        return Attn(
            query_in=query_one,
            key_in=key_in,
            value_in=value,
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=value_matrix,
            output_matrix=output_matrix,
        )


def trig_shift_matrix(k: int, trig_width: int) -> torch.Tensor:
    """Row-convention positional shift over the trig block.

    Returns a ``(trig_width, trig_width)`` matrix ``S`` such that

        trig(pos) @ S == trig(pos + k)

    for the sin/cos pairs, via ``sin(A+B) = sinA cosB + cosA sinB`` and
    ``cos(A+B) = cosA cosB - sinA sinB``.  ``trig_width`` must be even and
    must match the value used by :meth:`PosEncoding.get_pos_encoding` (the
    frequency grid is shared).

    :meth:`PosEncoding.attend_to_offset` feeds ``trig_shift_matrix(-delta_pos)``
    to the key side, so query position ``j`` attends to key ``j + delta_pos``.
    """
    S = torch.zeros((trig_width, trig_width))
    div_term = torch.exp(
        torch.arange(0, trig_width, 2) * -(math.log(10000.0) / trig_width)
    )
    for i in range(trig_width // 2):
        theta = div_term[i]
        c = torch.cos(theta * k)
        s = torch.sin(theta * k)
        # Row pair (sin, cos) right-multiplied by [[c, -s], [s, c]] yields
        # (sin(theta(pos+k)), cos(theta(pos+k))).
        S[2 * i + 0, 2 * i + 0] = c
        S[2 * i + 0, 2 * i + 1] = -s
        S[2 * i + 1, 2 * i + 0] = s
        S[2 * i + 1, 2 * i + 1] = c
    return S
