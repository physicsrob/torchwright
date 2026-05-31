from torchwright.graph import Node, Concatenate, Attn
from torchwright.graph.value_type import NodeValueType
import torch
import math

attention_hardness = 100.0  # Scales attention to be 1.0 and 0.0 everywhere else


class PosEncoding(Node):
    def __init__(self, d_pos: int):
        self.d_pos = d_pos
        super().__init__(d_pos, [])

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def get_pos_encoding(self, n_pos: int):
        pe = torch.zeros((n_pos, self.d_pos))
        div_term = torch.exp(
            torch.arange(0, self.d_pos, 2) * -(math.log(10000.0) / self.d_pos)
        )
        for pos in range(n_pos):
            pe[pos, 0::2] = torch.sin(pos * div_term)
            pe[pos, 1::2] = torch.cos(pos * div_term)
        # Replace the slowest sin (column d_pos-2) with a raw integer
        # position counter.  One Linear extraction gives exact position
        # indices — no piecewise-linear asin inversion needed.
        pe[:, self.d_pos - 2] = torch.arange(n_pos, dtype=pe.dtype)
        return pe

    def compute(self, n_pos: int, input_values: dict):
        return self.get_pos_encoding(n_pos)

    def slow_sin_freq(self) -> float:
        """Angular frequency of the slowest sinusoidal component of the
        positional encoding (the ``sin`` at row ``d_pos - 2``).

        For positions ``p`` with ``p · freq < π/2`` — i.e. roughly
        ``p < π/(2·freq)`` — the slow sin is monotonically increasing,
        and ``sin(p · freq) ≈ p · freq`` within a few percent. That makes
        it a cheap linear-in-position proxy on the key side of an
        attention head, without having to invert back through
        :meth:`get_position_scalar`.
        """
        return math.exp((self.d_pos - 2) * -(math.log(10000.0) / self.d_pos))

    def get_position_scalar(self, max_pos: int = 2048) -> Node:
        """Recover position index as a 1D scalar node.

        Extracts the raw integer counter at column ``d_pos - 2``.
        Exact for all positions.
        """
        from torchwright.graph.linear import Linear

        weight = torch.zeros(self.d_pos, 1)
        weight[self.d_pos - 2, 0] = 1.0
        return Linear(self, weight, name="position_scalar")

    def attend_to_offset(self, value: Node, delta_pos=-1) -> Node:
        if delta_pos == 0:
            # NOOP -- supporting this simplifies some use cases.
            return value

        # Q/K match positions in positional-encoding space; V/O transports
        # the payload independently and may be wider than the position vector.
        d_qk = self.d_pos
        d_v = len(value)

        # key_matrix shape (d_key_in, d_qk)
        delta = get_pos_delta_matrix(delta_pos, self.d_pos)
        # The last sin/cos pair (columns d_pos-2, d_pos-1) contains the
        # raw position counter instead of a sinusoid.  Zero it out so the
        # counter doesn't participate in the trig-identity attention.
        delta[self.d_pos - 2 :, :] = 0.0
        delta[:, self.d_pos - 2 :] = 0.0
        key_matrix = torch.zeros((len(self), d_qk))
        key_matrix[:, : self.d_pos] = delta

        # query_matrix shape (d_query_in, d_qk)
        query_matrix = attention_hardness * torch.eye(len(self), d_qk)
        query_matrix[self.d_pos - 2 :, :] = 0.0

        # value_matrix shape (d_value_in, d_v)
        value_matrix = torch.eye(d_v)

        # output_matrix shape (d_v, d_output)
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

    def get_prev_value(self, value: Node, cond: Node) -> Node:
        """
        Get the most recent previous value when cond is true.  cond must be 1.0 for true values,
        otherwise we won't guarantee that we return the maximum value.
        """
        # The penultimate component of the position encoding is approximately
        # sin(pos / 10000.0), which for our purposes is monotonically increasing.

        key_in = Concatenate([self, cond])
        d_head = self.d_pos

        assert self.d_pos <= d_head
        assert len(value) <= d_head
        assert len(cond) == 1

        # key_matrix shape (d_key_in, d_head)
        key_matrix = torch.zeros((len(key_in), d_head))
        key_matrix[-1, 0] = 1.0  # Cond
        # Column d_pos-2 is the raw position counter.  Scale it by the
        # old slow-sin frequency so the key contribution matches the
        # magnitude of the sinusoidal value it replaced.
        key_matrix[self.d_pos - 2, 0] = 100.0 * self.slow_sin_freq()

        # query_matrix shape (d_query_in, d_head)
        query_matrix = attention_hardness * torch.ones(len(self), d_head)
        # Exclude the counter from the query — its large magnitude
        # would dominate the sinusoidal position-matching logit.
        query_matrix[self.d_pos - 2, :] = 0.0

        # value_matrix shape (d_value_in, d_head)
        value_matrix = torch.eye(len(value), d_head)

        # output_matrix shape (d_head, d_output)
        output_matrix = torch.eye(d_head, len(value))

        return Attn(
            query_in=self,
            key_in=key_in,
            value_in=value,
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            value_matrix=value_matrix,
            output_matrix=output_matrix,
        )


def get_pos_delta_matrix(delta_pos: int, d: int):
    # Calculate a matrix M, such that M * pos_encoding(pos) = pos_encoding(pos + delta_pos)
    # We can do this using the trig identity sin(A+B) = sin(A)cos(B) + cos(A)sin(B)
    M = torch.zeros((d, d))

    # div_term must match from get_pos_encoding.  Note it's length d//2.
    div_term = torch.exp(torch.arange(0, d, 2) * -(math.log(10000.0) / d))

    # It's helpful to imagine d==2, in which case:
    # pos_encoding(pos) = [[sin(pos * div_term[0])], [cos(pos * div_term[0])]]

    for i in range(d // 2):
        k = div_term[i]
        # Loop through one sin/cos pair at a time
        # We want:
        # M*pos_encoding(pos) = pos_encoding(pos + delta_pos)
        # In the odd row case, we're calculating the new sin values
        # sin(k*pos + k*delta_pos) = sin(k*pos)*cos(k*delta_pos) + cos(k*pos)*sin(k*delta_pos)
        # Which is a linear combination, so can be expressed using our matrix.
        M[2 * i + 0, 2 * i + 0] = torch.cos(k * delta_pos)
        M[2 * i + 0, 2 * i + 1] = torch.sin(k * delta_pos)

        # In the even row case, we're calculating the new cosine values
        # cos(A + B) = cos(A)cos(B) − sin(A)sin(B)
        # cos(k*pos + k*delta_pos) = cos(k*pos)*cos(k*delta_pos) - sin(k*pos)*sin(k*delta_pos)
        M[2 * i + 1, 2 * i + 1] = torch.cos(k * delta_pos)
        M[2 * i + 1, 2 * i + 0] = -torch.sin(k * delta_pos)

    return M
