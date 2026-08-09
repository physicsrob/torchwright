"""Schematic format primitives — names, hashing, and the run encoding.

The one home for the constants and encodings shared by the producer
(``torchwright.compiler.schematic_capture``, ``torchwright.compiler.hf``)
and every consumer.  Deliberately stdlib-only at import: reading a
schematic must never pay for torch.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

SCHEMATIC_FORMAT = "torchwright.schematic.v1"
SCHEMATIC_FILENAME = "torchwright_schematic.json"
SCHEMATIC_SCHEMA_FILENAME = "torchwright_schematic_v1.schema.json"
SCHEMATIC_SUPPORT_FILENAME = "torchwright_schematic_support.npz"

#: The packaged JSON schema — the normative section inventory for v1.
SCHEMATIC_SCHEMA_SOURCE = Path(__file__).parent / SCHEMATIC_SCHEMA_FILENAME


def sha256_json(value: object) -> str:
    """Hash of the canonical JSON encoding of ``value``.

    This encoding (sorted keys, compact separators, raw unicode) is the
    contract between every schematic hash producer and validator — both
    sides must call this one function or freshly built bundles fail
    their own hash checks.
    """
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (bundle files are multi-GB)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def encode_cols(cols: list[int]) -> list[tuple[int, int]]:
    """Run-length encode an ORDERED column list as ``(start, length)`` runs.

    Column order is meaningful (column k holds component k of the node's
    value), so runs only merge consecutive ASCENDING indices — decoding
    reproduces the exact original order.
    """
    runs: list[tuple[int, int]] = []
    for c in cols:
        if runs and c == runs[-1][0] + runs[-1][1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((c, 1))
    return runs


def decode_cols(runs: list) -> list[int]:
    """Inverse of :func:`encode_cols` (accepts JSON-decoded lists).

    Order-preserving by contract: ``[[5, 2], [3, 1]]`` decodes to
    ``[5, 6, 3]`` — never sort the result.
    """
    cols: list[int] = []
    for start, length in runs:
        cols.extend(range(int(start), int(start) + int(length)))
    return cols


def column_runs(value: Sequence[int] | None) -> list[list[int]] | None:
    """Encode a column index list as the manifest's ``[start, length]`` runs.

    The one run-list encoding every manifest field uses; ``None`` stays
    ``None`` so optional fields serialize as JSON null.
    """
    return None if value is None else [list(run) for run in encode_cols(list(value))]
