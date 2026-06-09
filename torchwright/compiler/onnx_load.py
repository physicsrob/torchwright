"""Runtime loader for headless ONNX models.

Provides ``OnnxHeadlessModule`` — an ``onnxruntime``-backed callable
that speaks the static-cache prefill/decode protocol produced by
:func:`torchwright.compiler.export.compile_headless_to_onnx` — plus a
:class:`HeadlessRuntime` :class:`typing.Protocol` that describes the
shared interface with the in-memory
:class:`torchwright.compiler.export.CompiledHeadless`.

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
from typing import List, Optional, Protocol, Tuple, Union

import numpy as np
import torch

from torchwright.compiler.export import HEADLESS_META_FORMAT, meta_path_for


@dataclass
class OnnxPast:
    """Static-S sequence-major KV cache plus the committed length.

    ``k[i]`` / ``v[i]`` are the FULL ``(S, n_heads_i, d_head)`` cache
    buffers the ONNX graph's static ``past_K_i`` inputs require (S =
    cache_stride baked at export; ORT rejects shorter feeds).  Slots at
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


# CompiledHeadless (the in-process reference) keeps its own head-major
# tuple representation; the two runtimes share only the abstract
# "rows in, outputs + advanced past out" step contract.
PastKV = Union[OnnxPast, Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]]


class HeadlessRuntime(Protocol):
    """Structural type for any headless runtime (in-memory or ONNX).

    Lets callers type-hint "either :class:`CompiledHeadless` or
    :class:`OnnxHeadlessModule`" without importing both — a function
    that renders a frame or runs a decode step only needs to know that
    it has ``input_names``, ``metadata``, and the three callables below.
    """

    input_names: List[str]
    metadata: dict

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def step(
        self,
        inputs: torch.Tensor,
        past: PastKV,
        past_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, PastKV]: ...

    def empty_past(self) -> PastKV: ...

    def eval(self) -> "HeadlessRuntime": ...


class OnnxHeadlessModule:
    """Loads a headless cached ONNX model and exposes it as a callable.

    Args:
        onnx_path: Path to the ``.onnx`` file.  A sidecar
            ``<stem>.meta.json`` with format ``torchwright.headless.v1``
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
                f"compile_headless_to_onnx to produce it."
            )
        with open(meta_path) as f:
            meta = json.load(f)

        fmt = meta.get("format")
        if fmt != HEADLESS_META_FORMAT:
            raise ValueError(
                f"{meta_path}: unexpected format {fmt!r}, "
                f"expected {HEADLESS_META_FORMAT!r}"
            )
        self.input_names: List[str] = list(meta["input_names"])
        self.metadata: dict = dict(meta.get("extra") or {})

        self._session = ort.InferenceSession(
            onnx_path,
            providers=providers or ["CPUExecutionProvider"],
        )

        # Discover KV cache topology from the ONNX graph's input spec.
        # past_K_i inputs are sequence-major (S, n_heads_i, d_head) with a
        # STATIC slot count S = cache_stride; after head trimming each layer
        # may have a different head count.
        inputs = {inp.name: inp for inp in self._session.get_inputs()}
        self._n_layers = sum(1 for name in inputs if name.startswith("past_K_"))
        assert (
            self._n_layers > 0
        ), f"{onnx_path}: no past_K_* inputs — is this a cached-protocol model?"
        stride_dim = inputs["past_K_0"].shape[0]
        if not isinstance(stride_dim, int):
            raise ValueError(
                f"{onnx_path}: past_K_0 first dim is {stride_dim!r}, expected a "
                f"static int — this looks like a pre-static-cache (past_len/"
                f"Concat) export; recompile with the current exporter"
            )
        self._cache_stride = int(stride_dim)
        self._per_layer_n_heads = [
            int(inputs[f"past_K_{i}"].shape[1]) for i in range(self._n_layers)
        ]
        self._d_head = int(inputs["past_K_0"].shape[2])

        # Cache the list of output names in the protocol order so we can
        # unpack session.run() results without another dict lookup.  Outputs
        # are the per-layer KV *deltas* (the new rows only).
        self._out_names = ["outputs"]
        for i in range(self._n_layers):
            self._out_names += [f"delta_K_{i}", f"delta_V_{i}"]

    @property
    def cache_stride(self) -> int:
        """The static slot count ``S`` baked into the loaded model."""
        return self._cache_stride

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
        inputs: torch.Tensor,
        past: OnnxPast,
        past_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, OnnxPast]:
        """Run one cached-protocol call and return (outputs, new_past).

        Args:
            inputs: ``(n_new, d_input)`` float tensor.
            past: :class:`OnnxPast` from a prior step or
                :meth:`empty_past`.
            past_len: Optional explicit base position for the new rows.
                When ``None`` (default), uses ``past.length``.  Must equal
                the committed cache length — the static-cache graph derives
                both the mask and the positional encoding from
                ``cache_position``, so there is no trimmed-cache /
                sliding-window affordance (the old dynamic-concat graph's
                shape-derived mask is gone).

        Returns:
            ``(outputs, new_past)`` where ``outputs`` is a
            ``(n_new, d_output)`` torch tensor and ``new_past`` shares the
            (in-place updated) buffers with ``past`` at the advanced
            committed length.
        """
        if not isinstance(past, OnnxPast):
            raise TypeError(
                "OnnxHeadlessModule.step requires an OnnxPast from empty_past() "
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
        n_new = int(inputs.shape[0])
        if base + n_new > self._cache_stride:
            raise RuntimeError(
                f"static cache overrun: length {base} + n_new {n_new} exceeds "
                f"cache_stride {self._cache_stride}; re-export with a larger "
                f"cache_stride"
            )

        inputs_np = inputs.detach().cpu().numpy().astype(np.float32, copy=False)
        feeds: dict = {
            "inputs": inputs_np,
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
        outputs = torch.from_numpy(results[0])

        # Persist the per-layer deltas (new rows only) into the owned slots.
        for i in range(self._n_layers):
            past.k[i][base : base + n_new] = torch.from_numpy(results[1 + 2 * i])
            past.v[i][base : base + n_new] = torch.from_numpy(results[1 + 2 * i + 1])

        return outputs, OnnxPast(k=past.k, v=past.v, length=base + n_new)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience: stateless prefill that discards the cache.

        Equivalent to ``self.step(inputs, self.empty_past())[0]``. Use
        this for independent per-query inference (e.g. per-frame DOOM
        rendering) where no state is carried between calls.
        """
        outputs, _ = self.step(inputs, self.empty_past())
        return outputs

    def eval(self) -> "OnnxHeadlessModule":
        return self
