"""``TorchwrightTokenizer`` — a character-level tokenizer over a fixed
vocabulary.

The vocabulary is a JSON list of tokens in id order (``vocab.json``), one
entry per row of the model's embedding table. Every printable token is a
single character, so encoding is character-level: each character of the input
maps to one id. Multi-character control tokens (``<bos>`` / ``<eos>`` /
``<unk>``) are registered as special tokens and split out before the
character pass, exactly as ``transformers`` does for any added token.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class TorchwrightTokenizer(PreTrainedTokenizer):
    """Standalone slow ``PreTrainedTokenizer`` over a fixed vocabulary.

    A character-level encoder backed by the ``{token: id}`` bijection read from
    ``vocab.json`` (a JSON list of tokens in id order). ``bos`` is prepended on
    encode by default (``add_bos_token``); ``eos`` is the generation stop token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str | None = None,
        *,
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        add_bos_token: bool = True,
        **kwargs,
    ):
        if vocab_file is None:
            raise ValueError(
                "TorchwrightTokenizer needs vocab_file (vocab.json); load via "
                "from_pretrained on a saved bundle."
            )

        # Build the id<->token bijection BEFORE super().__init__ (which processes
        # special tokens and consults get_vocab()).
        self._tokens: list[str] = list(json.loads(Path(vocab_file).read_text()))
        self._token_to_id: dict[str, int] = {t: i for i, t in enumerate(self._tokens)}
        # transformers 5.x deliberately strips `add_bos_token` from the saved
        # tokenizer_config, so it can't round-trip under that name. Persist it
        # under our own key `prepend_bos` (set into init_kwargs below), which
        # save_pretrained keeps; on reload it arrives here as a kwarg.
        prepend_bos = kwargs.pop("prepend_bos", None)
        if prepend_bos is not None:
            add_bos_token = prepend_bos
        # Never prepend a bos that doesn't exist: a model with no bos passes
        # bos_token=None, and prepending its (None) id would inject None into
        # input_ids. Tie add_bos_token to bos actually being present.
        self.add_bos_token = bool(add_bos_token) and bos_token is not None

        # Byte-exact decode: no whitespace cleanup. setdefault (not a hardcoded
        # kwarg) so a value round-tripped from tokenizer_config on reload wins.
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        super().__init__(
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            **kwargs,
        )
        # Persisted under a non-stripped key so a bundle saved with
        # add_bos_token=False reloads with it.
        self.init_kwargs["prepend_bos"] = self.add_bos_token

    @property
    def vocab_size(self) -> int:
        return len(self._tokens)

    def get_vocab(self) -> dict[str, int]:
        return {**self._token_to_id, **self.added_tokens_encoder}

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        # Character-level: each character is its own token. Special / added
        # tokens are split out by the base tokenize() before this is called.
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        unk_id = self._token_to_id.get(self.unk_token, 0)
        return self._token_to_id.get(token, unk_id)

    def _convert_id_to_token(self, index: int) -> str:
        if 0 <= index < len(self._tokens):
            return self._tokens[index]
        return self.unk_token

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens)

    def _bos_prefix(self) -> list[int]:
        if self.add_bos_token and self.bos_token_id is not None:
            return [self.bos_token_id]
        return []

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        prefix = self._bos_prefix()
        out = prefix + token_ids_0
        if token_ids_1 is not None:
            out = out + prefix + token_ids_1
        return out

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1, already_has_special_tokens=True
            )
        prefix = [1] * len(self._bos_prefix())
        mask = prefix + [0] * len(token_ids_0)
        if token_ids_1 is not None:
            mask += prefix + [0] * len(token_ids_1)
        return mask

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str]:
        os.makedirs(save_directory, exist_ok=True)
        prefix = (filename_prefix + "-") if filename_prefix else ""
        vocab_path = os.path.join(
            save_directory, prefix + VOCAB_FILES_NAMES["vocab_file"]
        )
        Path(vocab_path).write_text(json.dumps(self._tokens))
        return (vocab_path,)
