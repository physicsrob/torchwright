"""Shared helpers for the HF-parity tests (leading underscore → not collected).

Everything here keeps the stock Phi-3 model and the ONNX oracle
(:class:`OnnxTokenModule`) on the **CPU**: both backends are deterministic
there, so the parity assertion can be exact (``max|Δlogit| == 0``). On a GPU,
cuBLAS picks matmul algorithms run-to-run, so the same comparison would only
hold at token (argmax) granularity — the calculator is a CPU artifact, so we
take the stronger bit-exact bar.
"""

from __future__ import annotations

import importlib
import os
import tempfile
from typing import List, Tuple

import torch


def compile_example(name: str) -> str:
    """Compile ``examples/<name>.py`` to a fresh ONNX artifact; return its path.

    In-process via ``compile_to_onnx`` (no committed artifact — ``*.onnx`` is
    gitignored). Small examples compile in a few seconds on CPU.
    """
    from torchwright.compiler.export import compile_to_onnx

    module = importlib.import_module(f"examples.{name}")
    output_node, embedding = module.create_network_parts()
    out_dir = tempfile.mkdtemp(prefix=f"tw_hf_{name}_")
    artifact = compile_to_onnx(
        output_node,
        embedding,
        os.path.join(out_dir, f"{name}.onnx"),
        d=module.D_MODEL,
        d_head=getattr(module, "D_HEAD", 16),
        bias=False,
    )
    return artifact.path


def oracle_decode(
    oracle, prefill_ids: List[int], n_steps: int
) -> Tuple[List[int], List[torch.Tensor]]:
    """Greedy argmax decode through the ONNX oracle's static-cache step loop.

    Returns (generated_ids, per_step_last_logits) — the last-position logits at
    the prefill and after each decode step (so callers can compare logits, not
    just argmax).
    """
    past = oracle.empty_past()
    logits, past = oracle.step(torch.tensor(prefill_ids, dtype=torch.int64), past)
    out_ids: List[int] = []
    out_logits: List[torch.Tensor] = [logits[-1].clone()]
    for _ in range(n_steps):
        nid = int(logits[-1].argmax())
        out_ids.append(nid)
        logits, past = oracle.step(torch.tensor([nid], dtype=torch.int64), past)
        out_logits.append(logits[-1].clone())
    return out_ids, out_logits


def hf_decode(
    model, prefill_ids: List[int], n_steps: int
) -> Tuple[List[int], List[torch.Tensor]]:
    """Greedy argmax decode through the HF model with a stock DynamicCache.

    Mirrors :func:`oracle_decode` step-for-step so the two are directly
    comparable. Explicit ``cache_position`` keeps the position bookkeeping
    identical to the oracle's static-cache contract.
    """
    from transformers.cache_utils import DynamicCache

    past = DynamicCache()
    n = len(prefill_ids)
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([prefill_ids], dtype=torch.int64),
            past_key_values=past,
            use_cache=True,
            cache_position=torch.arange(n),
        )
    logits = out.logits[0]
    past = out.past_key_values
    out_ids: List[int] = []
    out_logits: List[torch.Tensor] = [logits[-1].clone()]
    for _ in range(n_steps):
        nid = int(logits[-1].argmax())
        out_ids.append(nid)
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[nid]], dtype=torch.int64),
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([n]),
            )
        logits = out.logits[0]
        past = out.past_key_values
        n += 1
        out_logits.append(logits[-1].clone())
    return out_ids, out_logits


def hf_teacher_forced(
    model, prefill_ids: List[int], forced_ids: List[int]
) -> List[torch.Tensor]:
    """Decode the HF model along a FIXED token stream (teacher forcing).

    Mirrors :func:`hf_decode` step-for-step but feeds ``forced_ids`` (the
    oracle's tokens) instead of the model's own argmax, so every per-step logit
    row is computed on inputs identical to the oracle's. That is the only way a
    cross-backend logit comparison stays apples-to-apples once the compiled
    cancel-head rows cancel to denormal magnitude: there the argmax is arbitrary
    noise the two backends round differently, so two free-running loops would
    pick different garbage tokens and diverge. Returns ``len(forced_ids) + 1``
    last-position logit rows (prefill + one per forced step), aligned with
    :func:`oracle_decode`'s output.
    """
    from transformers.cache_utils import DynamicCache

    past = DynamicCache()
    n = len(prefill_ids)
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([prefill_ids], dtype=torch.int64),
            past_key_values=past,
            use_cache=True,
            cache_position=torch.arange(n),
        )
    past = out.past_key_values
    out_logits: List[torch.Tensor] = [out.logits[0][-1].clone()]
    for nid in forced_ids:
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([[nid]], dtype=torch.int64),
                past_key_values=past,
                use_cache=True,
                cache_position=torch.tensor([n]),
            )
        past = out.past_key_values
        n += 1
        out_logits.append(out.logits[0][-1].clone())
    return out_logits


def max_logit_diff(a: List[torch.Tensor], b: List[torch.Tensor]) -> float:
    """Max abs logit difference across all compared (prefill + per-step) rows."""
    return max((x - y).abs().max().item() for x, y in zip(a, b))
