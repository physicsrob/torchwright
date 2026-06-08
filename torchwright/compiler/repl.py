"""Interactive REPL that runs inference on an ONNX model.

Standalone -- does not depend on any torchwright graph code.
Requires: onnxruntime, numpy.
"""

import json
import sys
from dataclasses import dataclass
from typing import Any, Iterator, List

import numpy as np


class _Vocab:
    """Minimal tokenizer reconstructed from a vocab list."""

    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self._token_to_id = {t: i for i, t in enumerate(vocab)}

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, 0)  # 0 = <unk>

    def id_to_token(self, token_id: int) -> str:
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return self.vocab[0]


@dataclass
class _Model:
    """ONNX session plus the metadata the REPL loop needs for KV cache."""

    session: Any  # onnxruntime.InferenceSession
    vocab: _Vocab
    n_layers: int
    per_layer_n_heads: List[int]
    d_head: int


def generate(
    model: _Model,
    input_text: str,
    max_new_tokens: int = 10,
    bos_token: str = "<bos",
    eos_token: str = "<eos>",
) -> Iterator[str]:
    """Run autoregressive generation with a KV cache over an ONNX model.

    The ONNX graph is expected to expose the cached-forward interface:

        inputs:  token_ids, past_len, past_K_i, past_V_i
        outputs: logits,    delta_K_i, delta_V_i  (new rows only)

    Prefill is the first session.run() with empty past tensors. Each
    subsequent decode step feeds a single token and the accumulated past.

    Yields each generated token string as it is produced.
    """
    session = model.session
    vocab = model.vocab
    n_layers = model.n_layers
    per_layer_n_heads = model.per_layer_n_heads
    d_head = model.d_head

    tokens = [bos_token] + list(input_text)
    token_ids = np.array([vocab.token_to_id(t) for t in tokens], dtype=np.int64)

    past_K = [
        np.zeros((0, per_layer_n_heads[i], d_head), dtype=np.float32)
        for i in range(n_layers)
    ]
    past_V = [
        np.zeros((0, per_layer_n_heads[i], d_head), dtype=np.float32)
        for i in range(n_layers)
    ]
    past_len = 0

    out_names = ["logits"]
    for i in range(n_layers):
        out_names += [f"delta_K_{i}", f"delta_V_{i}"]

    def _step(step_tokens: np.ndarray) -> np.ndarray:
        nonlocal past_len
        feeds = {
            "token_ids": step_tokens,
            "past_len": np.array(past_len, dtype=np.int64),
        }
        for i in range(n_layers):
            feeds[f"past_K_{i}"] = past_K[i]
            feeds[f"past_V_{i}"] = past_V[i]
        outputs = session.run(out_names, feeds)
        logits = outputs[0]
        # Outputs are per-layer KV deltas (new rows only); grow the prefix.
        for i in range(n_layers):
            past_K[i] = np.concatenate([past_K[i], outputs[1 + 2 * i]], axis=0)
            past_V[i] = np.concatenate([past_V[i], outputs[1 + 2 * i + 1]], axis=0)
        past_len += int(step_tokens.shape[0])
        return logits

    # Prefill
    logits = _step(token_ids)

    # Decode loop — one token per session.run()
    for _ in range(max_new_tokens):
        next_id = int(logits[-1].argmax())
        next_token = vocab.id_to_token(next_id)
        if next_token == eos_token:
            break
        yield next_token
        logits = _step(np.array([next_id], dtype=np.int64))


def _load(onnx_path: str) -> _Model:
    """Load an ONNX model, its meta sidecar, and discover KV-cache shape metadata."""
    import onnxruntime  # type: ignore[import-untyped]

    from torchwright.compiler.export import TOKEN_META_FORMAT, meta_path_for

    meta_path = meta_path_for(onnx_path)
    with open(meta_path) as f:
        meta = json.load(f)

    fmt = meta.get("format")
    if fmt != TOKEN_META_FORMAT:
        raise ValueError(
            f"{meta_path}: unexpected format {fmt!r}, "
            f"expected {TOKEN_META_FORMAT!r}"
        )
    vocab = _Vocab(meta["vocab"])
    session = onnxruntime.InferenceSession(onnx_path)

    inputs = {inp.name: inp for inp in session.get_inputs()}
    n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
    assert n_layers > 0, "ONNX model has no past_K_* inputs — expected cached export"
    per_layer_n_heads = [int(inputs[f"past_K_{i}"].shape[1]) for i in range(n_layers)]
    d_head = int(inputs["past_K_0"].shape[2])

    return _Model(
        session=session,
        vocab=vocab,
        n_layers=n_layers,
        per_layer_n_heads=per_layer_n_heads,
        d_head=d_head,
    )


def run_once(
    onnx_path: str,
    prompt: str,
    max_new_tokens: int = 20,
) -> None:
    """Run a single prompt and print the result.

    Args:
        onnx_path: Path to the .onnx model file.
        prompt: The input string (e.g. "12+34").
        max_new_tokens: Maximum tokens to generate.
    """
    model = _load(onnx_path)
    for token in generate(model, prompt + "\n", max_new_tokens):
        sys.stdout.write(token)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()


def run_repl(
    onnx_path: str,
    max_new_tokens: int = 20,
) -> None:
    """Load an ONNX model and run an interactive REPL.

    Expects a meta file at <onnx_path_without_ext>.meta.json (token format).

    Args:
        onnx_path: Path to the .onnx model file.
        max_new_tokens: Maximum tokens to generate per query.
    """
    model = _load(onnx_path)

    print(f"Loaded {onnx_path} ({len(model.vocab.vocab)} tokens). Type 'q' to quit.")
    while True:
        text = input("> ")
        if text.lower() == "q":
            print("Bye")
            break
        for token in generate(model, text + "\n", max_new_tokens):
            sys.stdout.write(token)
            sys.stdout.flush()
        sys.stdout.write("\n")
        sys.stdout.flush()
