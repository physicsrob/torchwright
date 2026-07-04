from typing import List, Optional, Tuple

from torchwright.graph import Node, InputNode, LiteralValue, Embedding, RopeConfig
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
    :func:`torchwright.ops.relu.onehot_table.onehot_lookup`) and makes the unembed
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


def create_rope_config(
    d_head: int = 64, max_positions: int = 4096, d_rot: int | None = None
) -> RopeConfig:
    """Create the RoPE substrate a graph is built against.

    Replaces the old ``create_pos_encoding`` (``docs/rope_port_plan.md`` §7):
    position is a rotation applied inside attention, not a residual node, so a
    graph carries a :class:`~torchwright.graph.RopeConfig` (the ``d_head`` /
    ``base`` / ``max_positions`` the rotary builders need) instead of a
    ``PosEncoding`` node.  The ``d_head`` here **must** match the ``d_head``
    passed to the compile entry points (they assert it).

    Args:
        d_head: head width (= compiled ``d_head``).  Must be even and large
            enough that the widest content head's columns fit on the slow planes
            (``W <= d_head/2``) and the recency plane exists.
        max_positions: rollout length the recency plane is sized never to wrap
            over (the cache cap).
        d_rot: partial-rotary width (vanilla HF ``partial_rotary_factor``).  The
            first ``d_rot`` dims rotate; the rest are the unrotated NoPE tail.
            ``None`` (default) is full rotary (``d_rot = d_head``).  The
            content/recency/global-recency heads assume the full grid and reject
            a partial ``d_rot``; the rotary offset head supports it.

    Returns:
        RopeConfig.
    """
    return RopeConfig(d_head=d_head, max_positions=max_positions, d_rot=d_rot)
