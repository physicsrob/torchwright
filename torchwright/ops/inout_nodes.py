import torch

from torchwright.graph import Embedding, InputNode, LiteralValue, Node, RopeConfig
from torchwright.graph.embedding import Unembedding, unk_token

_DEFAULT_VALUE_RANGE = (-1e4, 1e4)


def create_input(
    name_or_width: str | int,
    width: int | None = None,
    *,
    value_range: tuple[float, float] | None = None,
) -> Node:
    """Create an input node with optional name and specified dimension.

    Supports two call patterns:
    - create_input(width) -> anonymous InputNode with given width
    - create_input(name, width) -> named InputNode (legacy pattern)

    Args:
        name_or_width: Either the input name (str) or width (int).
        width: Width when name is provided (optional).
        value_range: (lo, hi) bound on the input tensor values.
            Defaults to (-1e4, 1e4) if not specified.

    Returns:
        Node: The created input node.
    """
    if value_range is None:
        value_range = _DEFAULT_VALUE_RANGE
    if isinstance(name_or_width, int):
        # New pattern: create_input(width)
        return InputNode(name_or_width, value_range=value_range)
    # Legacy pattern: create_input(name, width)
    if width is None:
        raise ValueError("width is required when name is provided")
    return InputNode(width, name=name_or_width, value_range=value_range)


def create_literal_value(vector: torch.Tensor, name: str = "") -> Node:
    """Create a node with a literal value.

    Args:
        vector: Tensor representing the literal value.
        name: Node name for debugging.

    Returns:
        Node: Node with the specified literal value.
    """
    return LiteralValue(vector, name)


def create_embedding(vocab: list[str]) -> Embedding:
    """Create an embedding input.

    Args:
        vocab: List of vocab words.

    Returns:
        Node: Embedding node.
    """
    return Embedding(vocab)


def create_onehot_embedding(vocab: list[str]) -> Embedding:
    """Create a one-hot embedding with a zero-vector ``<unk>`` row.

    Every non-unknown token maps to an exact unit vector, while ``<unk>`` maps
    to zero.  If the caller did not include ``<unk>``, it is appended so every
    existing token id stays unchanged.  Thus ``d_embed`` is the number of
    non-unknown tokens and the table is rectangular: one row per vocabulary
    token, one column per semantic one-hot feature.  This makes lookups exact
    integer counting (see
    :func:`torchwright.ops.relu.onehot_table.onehot_lookup`) and makes the unembed
    an identity over every supported output token — ``argmax(output · eᵢ)``
    selects the hot column directly, while the zero unknown row cannot beat a
    valid unit-vector output.

    Args:
        vocab: Ordered token list; the row index of a token is its position
            here.

    Returns:
        Embedding node with one zero ``<unk>`` row.
    """
    tokens = list(vocab)
    if len(set(tokens)) != len(tokens):
        raise ValueError("one-hot embedding vocabulary tokens must be unique")
    if unk_token not in tokens:
        tokens.append(unk_token)

    semantic_tokens = [token for token in tokens if token != unk_token]
    semantic_ids = {token: i for i, token in enumerate(semantic_tokens)}
    table = torch.zeros(len(tokens), len(semantic_tokens))
    for row, token in enumerate(tokens):
        if token != unk_token:
            table[row, semantic_ids[token]] = 1.0
    return Embedding(
        tokens,
        d_embed=len(semantic_tokens),
        table=table,
        special_tokens=[],
    )


def create_unembedding(inp: Node, embedding: Embedding) -> Unembedding:
    """Create an unembedding output.

    Args:
        inp: Node with embedding vector to unembed.
        embedding: Embedding instance to use for unembedding.

    Returns:
        Unembedding
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
