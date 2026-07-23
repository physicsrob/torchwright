from numbers import Integral

from torchwright.graph import Node


def resolve_n_heads(
    d: int,
    d_head: int,
    n_heads: int | None,
    *,
    require_divisible: bool = True,
) -> int:
    """Resolve and validate the compile-time attention-head capacity.

    When ``n_heads`` is omitted, retain the historical geometry where the
    flattened attention width equals the residual width.  An explicit head
    count decouples those widths, so ``d`` need not be divisible by
    ``d_head``.
    """
    if not isinstance(d, Integral) or isinstance(d, bool) or d <= 0:
        raise ValueError(f"d must be a positive integer, got {d!r}")
    if not isinstance(d_head, Integral) or isinstance(d_head, bool) or d_head <= 0:
        raise ValueError(f"d_head must be a positive integer, got {d_head!r}")
    if n_heads is None:
        if require_divisible and d % d_head != 0:
            raise ValueError(
                f"d must be divisible by d_head when n_heads is omitted "
                f"(got d={d}, d_head={d_head}); pass n_heads explicitly to "
                f"decouple attention width from residual width"
            )
        return int(d // d_head)
    if not isinstance(n_heads, Integral) or isinstance(n_heads, bool) or n_heads <= 0:
        raise ValueError(f"n_heads must be a positive integer, got {n_heads!r}")
    return int(n_heads)


def get_ancestor_nodes(start_nodes: set[Node]) -> set[Node]:
    # Find all ancestors via iterative BFS
    result = set(start_nodes)
    queue = list(start_nodes)
    while queue:
        node = queue.pop()
        for inp in node.inputs:
            if inp not in result:
                result.add(inp)
                queue.append(inp)
    return result
