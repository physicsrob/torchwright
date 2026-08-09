"""Parameter-support archive — the schematic's nonzero-coordinate sidecar.

``torchwright_schematic_support.npz`` records, per checkpoint tensor,
exactly which stored fp32 entries are nonzero: 2-D tensors as chunked
CSR (chunk-local ``indptr``), 1-D tensors as flat indices.  The npz
array keys are insertion-counter names (``tensor_%06d_...``) and are
only discoverable through the manifest's ``parameter_support.tensors``
records — never guess them.

numpy is imported inside functions so importing this module stays
dependency-free; opening an archive is the first thing that pays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from torchwright.schematic.validate import SchematicValidationError

if TYPE_CHECKING:
    import numpy as np

_MATRIX_NDIM = 2


def _validate_csr_support(
    *,
    name: str,
    shape: list[int],
    nnz: int,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> None:
    import numpy as np

    invalid = (
        len(shape) != _MATRIX_NDIM
        or len(indptr) != shape[0] + 1
        or indptr[0] != 0
        or indptr[-1] != nnz
        or bool((np.diff(indptr) < 0).any())
        or (len(indices) and (indices.min() < 0 or indices.max() >= shape[1]))
    )
    if invalid:
        raise SchematicValidationError(f"malformed CSR support for {name}")


def _validate_support_tensor(
    name: str,
    record: dict[str, Any],
    arrays: np.lib.npyio.NpzFile,
    available: set[str],
) -> None:
    shape = record.get("shape")
    nnz = record.get("nnz")
    if not isinstance(shape, list) or not isinstance(nnz, int):
        raise SchematicValidationError(f"invalid support metadata for {name}")
    if record.get("encoding") == "flat_indices":
        indices_key = record.get("indices")
        if indices_key not in available or len(arrays[indices_key]) != nnz:
            raise SchematicValidationError(f"invalid support indices for {name}")
        indices = arrays[indices_key]
        size = math.prod(shape)
        if len(indices) and (indices.min() < 0 or indices.max() >= size):
            raise SchematicValidationError(f"out-of-bounds flat support for {name}")
        return
    chunks = record.get("chunks")
    if record.get("encoding") != "csr_row_chunks" or not isinstance(chunks, list):
        raise SchematicValidationError(f"unknown support encoding for {name}")
    expected_row = total_nnz = 0
    for chunk in chunks:
        indptr_key = chunk.get("indptr")
        indices_key = chunk.get("indices")
        if indptr_key not in available or indices_key not in available:
            raise SchematicValidationError(f"missing support chunk for {name}")
        if chunk.get("row_start") != expected_row:
            raise SchematicValidationError(f"non-contiguous support chunks for {name}")
        chunk_rows, chunk_nnz = chunk["row_count"], chunk["nnz"]
        _validate_csr_support(
            name=name,
            shape=[chunk_rows, shape[1]],
            nnz=chunk_nnz,
            indptr=arrays[indptr_key],
            indices=arrays[indices_key],
        )
        expected_row += chunk_rows
        total_nnz += chunk_nnz
    if expected_row != shape[0] or total_nnz != nnz:
        raise SchematicValidationError(f"incomplete support chunks for {name}")


def _checked_support_path(directory: Path, support: dict[str, Any]) -> Path:
    filename = support.get("file")
    if not isinstance(filename, str) or not filename:
        raise SchematicValidationError("schematic parameter_support names no file")
    relative = Path(filename)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        raise SchematicValidationError(
            f"schematic has unsafe support path {filename!r}"
        )
    support_path = directory / relative
    if not support_path.is_file():
        raise SchematicValidationError(f"schematic support file is missing: {filename}")
    return support_path


def validate_support_archive(directory: Path, support: dict[str, Any]) -> None:
    """Structurally validate the support npz against its manifest records."""
    import numpy as np

    support_path = _checked_support_path(directory, support)
    with np.load(support_path, allow_pickle=False) as arrays:
        available = set(arrays.files)
        for name, record in support.get("tensors", {}).items():
            _validate_support_tensor(name, record, arrays, available)


@dataclass(frozen=True)
class SupportArchive:
    """Decoded view of the support npz, keyed by checkpoint tensor name.

    ``coordinates`` returns the exact nonzero coordinate set the builder
    recorded: ``{(flat,)}`` for 1-D tensors, ``{(row, col)}`` for 2-D.
    Sets are materialized in memory — fine at current model scale.
    """

    #: parameter_support.tensors records from the manifest.
    tensors: dict[str, dict[str, Any]]
    #: Loaded npz arrays, keyed by their insertion-counter names.
    arrays: dict[str, Any]

    @classmethod
    def load(cls, directory: Path, support: dict[str, Any]) -> SupportArchive:
        """Open and structurally validate a bundle's support archive."""
        import numpy as np

        validate_support_archive(directory, support)
        support_path = _checked_support_path(directory, support)
        with np.load(support_path, allow_pickle=False) as handle:
            arrays = {key: handle[key] for key in handle.files}
        return cls(tensors=dict(support.get("tensors", {})), arrays=arrays)

    def coordinates(self, tensor: str) -> set[tuple[int, ...]]:
        """Nonzero coordinates for one checkpoint tensor."""
        record = self.tensors.get(tensor)
        if record is None:
            raise SchematicValidationError(f"no support record for tensor {tensor!r}")
        if record["encoding"] == "flat_indices":
            return {(int(index),) for index in self.arrays[record["indices"]]}
        result: set[tuple[int, ...]] = set()
        for chunk in record["chunks"]:
            indptr = self.arrays[chunk["indptr"]]
            indices = self.arrays[chunk["indices"]]
            for local_row in range(chunk["row_count"]):
                start, end = indptr[local_row : local_row + 2]
                result.update(
                    (chunk["row_start"] + local_row, int(column))
                    for column in indices[start:end]
                )
        return result
