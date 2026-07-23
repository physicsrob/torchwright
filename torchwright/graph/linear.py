import torch

from torchwright.graph.node import Node
from torchwright.graph.value_type import NodeValueType


class Linear(Node):
    """Affine transform: ``y = x @ output_matrix + output_bias``.

    The compiler may realise this as either an MLP slice or an attention
    head attending to the current position, depending on context.

    Attributes:
        output_matrix: Weight matrix, shape ``(d_input, d_output)``.
        output_bias: Bias vector, shape ``(d_output,)``.
    """

    output_matrix: torch.Tensor  # d_input x d_output
    output_bias: torch.Tensor  # d_output

    def __init__(
        self,
        input_node: Node,
        output_matrix: torch.Tensor,
        output_bias: torch.Tensor | None = None,
        name: str = "",
    ):
        # output_matrix shape (d_input, d_output)
        self.d_input = output_matrix.shape[0]
        self.d_output = output_matrix.shape[1]
        assert len(input_node) == self.d_input
        self.output_matrix = output_matrix

        if output_bias is None:
            self.output_bias = torch.zeros(self.d_output)
        else:
            assert len(output_bias) == self.d_output
            self.output_bias = output_bias

        super().__init__(self.d_output, [input_node], name=name)

    def compute(self, n_pos: int, input_values: dict) -> torch.Tensor:
        value_in = self.inputs[0].compute(n_pos, input_values)

        assert value_in.shape == (n_pos, self.d_input)
        return torch.matmul(value_in, self.output_matrix) + self.output_bias

    def compute_value_type(self) -> NodeValueType:
        return NodeValueType()

    def num_params(self):
        return self.d_input * self.d_output + self.d_output
