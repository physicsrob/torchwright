"""Runtime loaders for torchwright ONNX exports.

:func:`load_onnx` is the front door: it reads the ``<stem>.meta.json``
sidecar, checks its format key, and returns an ``OnnxTokenModule`` — an
``onnxruntime``-backed callable speaking the static-cache prefill/decode
protocol:

- ``OnnxTokenModule`` — token I/O (``torchwright.token.v4``), produced
  by :func:`torchwright.compiler.export.compile_to_onnx`; adds the
  vocab tokenizer and an argmax :meth:`OnnxTokenModule.generate` loop.

Two usage shapes:

- **Independent per-query** (e.g. per-frame DOOM rendering):
  ``module(inputs)`` runs one prefill call and returns outputs. Each
  call is stateless — the KV cache built during the call is discarded.

- **Autoregressive decode**: ``past = module.empty_past()`` for the
  initial state, then ``outputs, past = module.step(inputs, past)``
  repeatedly to extend the cached context one chunk at a time.

``.eval()`` returns ``self`` for PyTorch drop-in symmetry.
"""

import json
import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch

from torchwright.compiler.export import (
    TOKEN_META_FORMAT,
    meta_path_for,
)


def discover_cache_stride(inputs: dict, sidecar_stride, onnx_path) -> int:
    """Resolve the full static slot count ``S`` for a cached-protocol model.

    Shared by both loaders (``OnnxTokenModule`` and torchwright_doom's
    ``OnnxTokenRuntime``).
    Dual path:

    - ``past_K_0`` first dim is a static int — a pre-bucketing export
      (old compile caches): use it.  Such models can only bind the full
      buffer (their mask is full-width); that is automatic and correct.
    - first dim is the symbolic ``cache_slots`` — a stride-bucketing
      export: ``S`` comes from the sidecar meta (``sidecar_stride``,
      resolved by the caller from its meta format).

    A ``past_len`` graph input identifies the ancient pre-static-cache
    (Concat) contract and is always an error naming that cause.
    """
    if "past_len" in inputs:
        raise ValueError(
            f"{onnx_path}: model has a past_len input — this is a "
            f"pre-static-cache (past_len/Concat) artifact; bust the "
            f"compile-cache entry and recompile with the current exporter"
        )
    stride_dim = inputs["past_K_0"].shape[0]
    if isinstance(stride_dim, int):
        return int(stride_dim)
    if sidecar_stride is not None:
        return int(sidecar_stride)
    raise ValueError(
        f"{onnx_path}: past_K_0 first dim is symbolic ({stride_dim!r}) but "
        f"the sidecar meta carries no cache_stride — re-export with the "
        f"current exporter (it writes the key) or restore the sidecar"
    )


@dataclass
class OnnxPast:
    """Static-S sequence-major KV cache plus the committed length.

    ``k[i]`` / ``v[i]`` are the FULL ``(S, n_heads_i, d_head)`` cache
    buffers (S = cache_stride from the export sidecar).  This module
    always feeds the full buffers — it is the contract-correctness
    harness, not a perf path; prefix-window (stride-bucket) feeds are
    the GPU runtime's affordance.  Slots at
    positions >= ``length`` are zeros — never garbage: hidden slots get
    softmax weight exactly 0.0 via the in-graph mask, and ``0 * NaN``
    would still be NaN.  ``step`` writes the returned deltas into
    ``[length : length+n_new)`` in place and returns a new ``OnnxPast``
    with the advanced length (the buffers are shared, the length is
    functional state).
    """

    k: Tuple[torch.Tensor, ...]
    v: Tuple[torch.Tensor, ...]
    length: int


class OnnxTokenModule:
    """Loads a token-I/O cached ONNX model and exposes it as a callable.

    Speaks the static-cache prefill/decode protocol with ``token_ids``
    (int64) as the sequence input and ``logits`` as the sequence output;
    the sidecar carries the vocab, so the module can tokenize/detokenize
    and run an argmax :meth:`generate` loop directly.

    Args:
        onnx_path: Path to the ``.onnx`` file.  A sidecar
            ``<stem>.meta.json`` with format ``torchwright.token.v4``
            must exist alongside it.
        providers: ``onnxruntime`` execution providers list.  Defaults
            to CPU.
    """

    def __init__(self, onnx_path: str, providers=None) -> None:
        import onnxruntime as ort

        meta_path = meta_path_for(onnx_path)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Missing sidecar {meta_path}. Re-export with "
                f"compile_to_onnx to produce it."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        fmt = meta.get("format")
        if fmt != TOKEN_META_FORMAT:
            raise ValueError(
                f"{meta_path}: unexpected format {fmt!r}, "
                f"expected {TOKEN_META_FORMAT!r}"
            )
        self.vocab: List[str] = list(meta["vocab"])
        self._token_to_id = {t: i for i, t in enumerate(self.vocab)}
        self.metadata: dict = dict(meta.get("extra") or {})

        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )

        # KV cache topology discovery from the ONNX graph's input spec.
        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
        assert (
            self._n_layers > 0
        ), f"{onnx_path}: no past_K_* inputs — is this a cached-protocol model?"
        self._cache_stride = discover_cache_stride(
            inputs, meta.get("cache_stride"), onnx_path
        )
        self._per_layer_n_heads = [
            int(inputs[f"past_K_{i}"].shape[1]) for i in range(self._n_layers)
        ]
        self._d_head = int(inputs["past_K_0"].shape[2])

        self._out_names = ["logits"]
        for i in range(self._n_layers):
            self._out_names += [f"delta_K_{i}", f"delta_V_{i}"]

    @property
    def cache_stride(self) -> int:
        """The static slot count ``S`` baked into the loaded model."""
        return self._cache_stride

    @property
    def n_layers(self) -> int:
        return self._n_layers

    @property
    def per_layer_n_heads(self) -> List[int]:
        return list(self._per_layer_n_heads)

    @property
    def d_head(self) -> int:
        return self._d_head

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, 0)  # 0 = <unk>

    def id_to_token(self, token_id: int) -> str:
        if 0 <= token_id < len(self.vocab):
            return self.vocab[token_id]
        return self.vocab[0]

    def empty_past(self) -> OnnxPast:
        """Full-S zero-filled sequence-major cache buffers, length 0.

        Zero (not garbage) is load-bearing: slots beyond the committed
        length are read by the attention with weight exactly 0.0, and
        ``0 * NaN = NaN``.
        """
        S = self._cache_stride
        past_K = tuple(
            torch.zeros(S, nh, self._d_head) for nh in self._per_layer_n_heads
        )
        past_V = tuple(
            torch.zeros(S, nh, self._d_head) for nh in self._per_layer_n_heads
        )
        return OnnxPast(k=past_K, v=past_V, length=0)

    def step(
        self,
        token_ids: torch.Tensor,
        past: OnnxPast,
        past_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, OnnxPast]:
        """Run one cached-protocol call and return (logits, new_past).

        Args:
            token_ids: ``(n_new,)`` int64 tensor of token ids.
            past: :class:`OnnxPast` from a prior step or
                :meth:`empty_past`.
            past_len: Optional explicit base position for the new rows.
                When ``None`` (default), uses ``past.length``.  Must equal
                the committed cache length: the static-cache graph derives
                mask AND pos from ``cache_position``, so a trimmed cache
                with a larger absolute position is not expressible.

        Returns:
            ``(logits, new_past)`` where ``logits`` is a
            ``(n_new, vocab_size)`` torch tensor and ``new_past`` shares
            the (in-place updated) buffers with ``past`` at the advanced
            committed length.
        """
        if not isinstance(past, OnnxPast):
            raise TypeError(
                "OnnxTokenModule.step requires an OnnxPast from empty_past() "
                f"(got {type(past).__name__}) — the static-cache protocol has "
                "no growable-tuple representation"
            )
        assert len(past.k) == self._n_layers
        assert len(past.v) == self._n_layers

        base = past.length if past_len is None else int(past_len)
        assert base == past.length, (
            f"past_len {base} != committed length {past.length}: the static "
            f"cache derives mask AND pos from cache_position; a trimmed cache "
            f"with a larger absolute position is not expressible"
        )
        n_new = int(token_ids.shape[0])
        if base + n_new > self._cache_stride:
            raise RuntimeError(
                f"static cache overrun: length {base} + n_new {n_new} exceeds "
                f"cache_stride {self._cache_stride}; re-export with a larger "
                f"cache_stride"
            )

        ids_np = token_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        feeds: dict = {
            "token_ids": ids_np,
            "cache_position": np.arange(base, base + n_new, dtype=np.int64),
        }
        for i in range(self._n_layers):
            feeds[f"past_K_{i}"] = (
                past.k[i].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            feeds[f"past_V_{i}"] = (
                past.v[i].detach().cpu().numpy().astype(np.float32, copy=False)
            )

        results = self._session.run(self._out_names, feeds)
        logits = torch.from_numpy(results[0])

        # Persist the per-layer deltas (new rows only) into the owned slots.
        for i in range(self._n_layers):
            past.k[i][base : base + n_new] = torch.from_numpy(results[1 + 2 * i])
            past.v[i][base : base + n_new] = torch.from_numpy(results[1 + 2 * i + 1])

        return logits, OnnxPast(k=past.k, v=past.v, length=base + n_new)

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convenience: stateless prefill that discards the cache.

        Equivalent to ``self.step(token_ids, self.empty_past())[0]``.
        """
        logits, _ = self.step(token_ids, self.empty_past())
        return logits

    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 10,
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ) -> Iterator[str]:
        """Autoregressive argmax generation over the cached protocol.

        Prefills ``[bos_token] + list(input_text)``, then decodes one token per
        :meth:`step`, yielding each generated token string as it is produced.
        Stops at ``eos_token`` or ``max_new_tokens``.

        (RoPE recency is now the intrinsic local rotary lobe — there is no
        injected ``<ref>`` reference token; ``docs/rope_port_plan.md`` Phase 6.)
        """
        tokens = [bos_token] + list(input_text)
        ids = torch.tensor([self.token_to_id(t) for t in tokens], dtype=torch.int64)

        logits, past = self.step(ids, self.empty_past())
        for _ in range(max_new_tokens):
            next_id = int(logits[-1].argmax())
            next_token = self.id_to_token(next_id)
            if next_token == eos_token:
                break
            yield next_token
            logits, past = self.step(torch.tensor([next_id], dtype=torch.int64), past)

    def eval(self) -> "OnnxTokenModule":
        return self


def load_onnx(onnx_path: str, providers=None) -> OnnxTokenModule:
    """Load a torchwright token-I/O ONNX export.

    Reads ``<stem>.meta.json`` next to ``onnx_path`` and returns an
    :class:`OnnxTokenModule` (``torchwright.token.v4``).
    """
    meta_path = meta_path_for(onnx_path)
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing sidecar {meta_path}. Re-export with compile_to_onnx "
            f"to produce it."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    fmt = meta.get("format")
    if fmt == TOKEN_META_FORMAT:
        return OnnxTokenModule(onnx_path, providers=providers)
    raise ValueError(
        f"{meta_path}: unknown format {fmt!r}; expected {TOKEN_META_FORMAT!r}"
    )
