"""Format primitives: the ordered run encoding and the hash contract."""

import hashlib

from torchwright.schematic.format import (
    column_runs,
    decode_cols,
    encode_cols,
    sha256_file,
    sha256_json,
)

# sha256_json of {"b": 1, "a": [1, 2], "c": "π"} — pins the canonical
# encoding (sorted keys, compact separators, ensure_ascii=False).  If
# this moves, every shipped schematic fails its own hash checks.
_JSON_PIN = "9663ff95274ee201c71b09d45fec9aaebc9c8f4929dd1fb3aaafdea9682d333d"


def test_decode_preserves_order_never_sorts():
    # Column order is meaningful: column k holds component k.
    assert decode_cols([[5, 2], [3, 1]]) == [5, 6, 3]


def test_roundtrip_on_interleaved_and_descending():
    for cols in (
        [5, 6, 3],
        [9, 8, 7],
        [0],
        [],
        [2, 3, 4, 10, 11, 1],
        [1, 3, 5, 7],
    ):
        assert decode_cols([list(run) for run in encode_cols(cols)]) == cols


def test_encode_merges_only_ascending_consecutive():
    # Descending neighbors never merge — decoding must reproduce order.
    assert encode_cols([5, 4, 3]) == [(5, 1), (4, 1), (3, 1)]


def test_encode_exact_shapes():
    assert encode_cols([3, 4, 5]) == [(3, 3)]
    assert encode_cols([5, 6, 3]) == [(5, 2), (3, 1)]
    assert encode_cols([]) == []


def test_column_runs_wrapper():
    assert column_runs(None) is None
    assert column_runs((5, 6, 3)) == [[5, 2], [3, 1]]


def test_sha256_json_pin():
    assert sha256_json({"b": 1, "a": [1, 2], "c": "π"}) == _JSON_PIN


def test_sha256_json_key_order_invariant():
    assert sha256_json({"a": 1, "b": 2}) == sha256_json({"b": 2, "a": 1})


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    payload = bytes(range(256)) * 5000  # > one 1 MiB read block
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
