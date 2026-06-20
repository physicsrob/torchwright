from typing import List, Optional, Tuple

from torchwright.graph import Node, InputNode, LiteralValue, Embedding, PosEncoding
import torch

from torchwright.graph.embedding import Unembedding

_DEFAULT_VALUE_RANGE = (-1e4, 1e4)


def create_input(
    name_or_width,
    width: int | None = None,
    *,
    value_range: Optional[Tuple[float, float]] = None,
) -> Node:
    """
    Create an input node with optional name and specified dimension.

    Supports two call patterns:
    - create_input(width) -> anonymous InputNode with given width
    - create_input(name, width) -> named InputNode (legacy pattern)

    Args:
    - name_or_width: Either the input name (str) or width (int)
    - width: Width when name is provided (optional)
    - value_range: (lo, hi) bound on the input tensor values.
      Defaults to (-1e4, 1e4) if not specified.

    Returns:
    - Node: The created input node.
    """
    if value_range is None:
        value_range = _DEFAULT_VALUE_RANGE
    if isinstance(name_or_width, int):
        # New pattern: create_input(width)
        return InputNode(name_or_width, value_range=value_range)
    else:
        # Legacy pattern: create_input(name, width)
        if width is None:
            raise ValueError("width is required when name is provided")
        return InputNode(width, name=name_or_width, value_range=value_range)


def create_literal_value(vector: torch.Tensor, name: str = "") -> Node:
    """
    Create a node with a literal value.

    Args:
    - vector (torch.Tensor): Tensor representing the literal value.

    Returns:
    - Node: Node with the specified literal value.
    """
    return LiteralValue(vector, name)


def create_embedding(vocab: List[str]) -> Embedding:
    """
    Create an embedding input.

    Args:
    - vocab (List[str]): List of vocab words.

    Returns:
    - Node: Embedding node.
    """
    return Embedding(vocab)


def create_onehot_embedding(vocab: List[str]) -> Embedding:
    """Create a one-hot embedding: token ``i`` maps to the ``i``-th unit vector.

    The table is the identity ``eye(len(vocab))`` and there are no special
    tokens, so ``d_embed == len(vocab)`` and every embedding row is an exact
    one-hot.  This makes lookups exact integer counting (see
    :func:`torchwright.ops.onehot_table.onehot_lookup`) and makes the unembed
    an identity — ``argmax(output · eᵢ)`` over the equal-norm rows selects the
    hot column directly, so argmax-decode is exact.

    Args:
        vocab: Ordered token list; the row index of a token is its position
            here.

    Returns:
        Embedding node with an identity table and no ``<unk>`` prefix.
    """
    n = len(vocab)
    return Embedding(vocab, d_embed=n, table=torch.eye(n), special_tokens=[])


def create_unembedding(inp: Node, embedding: Embedding) -> Unembedding:
    """
    Create an unembedding output.

    Args:
    - inp (Node): Node with embedding vector to unembed
    - embedding (Embedding): Embedding instance to use for unembedding.

    Returns:
    - Unembedding
    """
    return Unembedding(inp, embedding)


def create_pos_encoding() -> PosEncoding:
    """
    Create a position encoding.

    16 trig columns plus one raw position counter (``d_pos = 17``).  The
    16-wide trig grid matches a default ``d_head = 16`` exactly.

    Returns:
    - Node: PosEncoding node.
    """
    return PosEncoding(d_pos=17)
