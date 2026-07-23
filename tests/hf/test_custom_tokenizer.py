"""Generic TorchwrightCustomTokenizer behavior (char-level + bos handling).

Fast and self-contained: writes a tiny vocab.json and exercises the tokenizer
directly — no ONNX artifact, no model. Covers the bos-prepend default, the
``add_bos_token`` save/load round-trip, and the no-bos (``bos_token=None``) path
that must never inject a ``None`` id.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("transformers")

from torchwright.compiler.hf.tokenization_torchwright_custom import (
    TorchwrightCustomTokenizer,
)

_VOCAB = ["<unk>", "<bos>", "<eos>", *list("0123456789+-*\n")]


@pytest.fixture
def vocab_file(tmp_path):
    p = tmp_path / "vocab.json"
    p.write_text(json.dumps(_VOCAB))
    return str(p)


def test_char_level_encode_decode_round_trip(vocab_file):
    tok = TorchwrightCustomTokenizer(
        vocab_file=vocab_file, bos_token="<bos>", eos_token="<eos>"
    )
    ids = tok("12*34\n")["input_ids"]
    # bos prepended, then one id per character.
    assert ids[0] == tok.bos_token_id
    assert ids[1:] == [tok._token_to_id[c] for c in "12*34\n"]
    assert tok.decode(ids, skip_special_tokens=True) == "12*34\n"


def test_add_bos_token_false_persists(tmp_path, vocab_file):
    tok = TorchwrightCustomTokenizer(
        vocab_file=vocab_file,
        bos_token="<bos>",
        eos_token="<eos>",
        add_bos_token=False,
    )
    assert tok("12")["input_ids"] == [tok._token_to_id[c] for c in "12"]  # no bos
    save_dir = tmp_path / "tok"
    tok.save_pretrained(save_dir)
    reloaded = TorchwrightCustomTokenizer.from_pretrained(save_dir)
    assert reloaded.add_bos_token is False
    assert reloaded("12")["input_ids"] == [reloaded._token_to_id[c] for c in "12"]


def test_no_bos_token_never_emits_none(vocab_file):
    # A model with no bos passes bos_token=None; add_bos_token must collapse to
    # False so no None id is ever prefixed into input_ids.
    tok = TorchwrightCustomTokenizer(
        vocab_file=vocab_file, bos_token=None, eos_token="<eos>", add_bos_token=True
    )
    assert tok.add_bos_token is False
    ids = tok("12")["input_ids"]
    assert None not in ids
    assert ids == [tok._token_to_id[c] for c in "12"]
