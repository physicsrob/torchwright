from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torchwright.compiler.components.component import Component
from torchwright.graph import PosEncoding
from torchwright.graph.attn import CAUSAL_MASK_SENTINEL  # kept for compat

# F.scaled_dot_product_attention's default backend on A100 with fp32
# inputs is EFFICIENT_ATTENTION, which on some inputs perturbs V by
# 1 fp32 mantissa-LSB in the away-from-zero direction.  Trigger
# condition is mantissa-pattern-dependent (not a clean magnitude or
# bit-position rule) and not explained by TF32 / fp16 / bf16 rounding.
# The MATH backend matches manual softmax+matmul exactly.
#
# Why it matters here: cancel heads (V=identity, O=-identity) rely on
# attn_out + skip = 0 algebraically.  A 1-LSB perturbation at q≈22855
# leaves a ~1/512 leak in the residual column, which propagates through
# the encoder and flips Gray-code bits at boundary q values.
# Same shape applies to compute_linear (Linear emulated as self-attn).
_SDPA_BACKEND = [SDPBackend.MATH]


class AttnLayerComponent(Component):
    """Multi-head attention component.

    Weight matrices:
        query_matrix:  (n_heads, d, d_head)
        key_matrix:    (n_heads, d, d_head)
        value_matrix:  (n_heads, d, d_head)
        output_matrix: (n_heads, d_head, d)
    """

    def __init__(
        self, d: int, d_head: int, pos_encoding: Optional[PosEncoding], name: str = ""
    ):
        super().__init__(d, name)
        assert (d % d_head) == 0, "Invalid combination of d and d_head"
        self.d_head = d_head
        self.n_heads = d // d_head
        self.used_heads = 0
        self.pos_encoding = pos_encoding

        self.query_matrix = torch.zeros(self.n_heads, d, d_head)
        self.key_matrix = torch.zeros(self.n_heads, d, d_head)
        self.value_matrix = torch.zeros(self.n_heads, d, d_head)
        self.output_matrix = torch.zeros(self.n_heads, d_head, d)

    def __repr__(self):
        return f"AttnLayerComponent(name='{self.name}')"

    def forward(self, inp: torch.Tensor):
        # inp shape (n_pos, d)
        assert inp.shape[1] == self.d

        # All heads in parallel via batched ops
        Q = torch.einsum("pd,hdk->hpk", inp, self.query_matrix)
        K = torch.einsum("pd,hdk->hpk", inp, self.key_matrix)
        V = torch.einsum("pd,hdk->hpk", inp, self.value_matrix)

        # Fused attention kernel.  scale=1.0 preserves the raw dot-product
        # magnitude that all attention weights were compiled against (no
        # 1/sqrt(d_head) rescaling).  is_causal=True applies the standard
        # upper-triangular mask for causal prefill.
        # Shape: (n_heads, n_pos, d_head) → unsqueeze batch → squeeze back.
        with sdpa_kernel(_SDPA_BACKEND):
            weighted = F.scaled_dot_product_attention(
                Q.unsqueeze(0),
                K.unsqueeze(0),
                V.unsqueeze(0),
                is_causal=True,
                scale=1.0,
            ).squeeze(
                0
            )  # (n_heads, n_pos, d_head)

        output = torch.einsum("hpk,hkd->pd", weighted, self.output_matrix)
        return output

    def forward_cached(
        self,
        inp: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with KV cache.

        Args:
            inp: (n_new, d) — new positions only (full seq on prefill, 1 on generation)
            past_kv: None or (K, V) each (n_heads, n_past, d_head)

        Returns:
            output: (n_new, d)
            new_kv: (K, V) each (n_heads, n_past+n_new, d_head)
        """
        Q = torch.einsum("pd,hdk->hpk", inp, self.query_matrix)
        K_new = torch.einsum("pd,hdk->hpk", inp, self.key_matrix)
        V_new = torch.einsum("pd,hdk->hpk", inp, self.value_matrix)

        if past_kv is not None:
            K = torch.cat([past_kv[0], K_new], dim=1)
            V = torch.cat([past_kv[1], V_new], dim=1)
        else:
            K, V = K_new, V_new

        n_new = inp.shape[0]
        n_total = K.shape[1]
        n_past = n_total - n_new

        # Three regimes:
        #   pure prefill (no past, n_new == n_total): is_causal=True gives
        #     the standard upper-triangular mask.
        #   single-step decode (n_new == 1, n_past > 0): every K entry is in
        #     the past or is the lone new row, so no mask is needed.
        #   batched decode with past (n_new > 1 and n_past > 0): every new
        #     row sees the full past unconditionally, but among new rows the
        #     mask is lower-triangular (row i sees new rows 0..i). This is
        #     the speculative-decoding shape — is_causal=True would mask out
        #     past columns for early rows; is_causal=False would let row 0
        #     attend to future new rows.
        # scale=1.0 preserves the raw dot-product magnitudes that all attention
        # weights were compiled against (no 1/sqrt(d_head) rescaling).
        if n_new > 1 and n_past > 0:
            past_visible = torch.ones(n_new, n_past, dtype=torch.bool, device=Q.device)
            new_block = torch.ones(
                n_new, n_new, dtype=torch.bool, device=Q.device
            ).tril()
            attn_mask = torch.cat([past_visible, new_block], dim=1)
            with sdpa_kernel(_SDPA_BACKEND):
                weighted = F.scaled_dot_product_attention(
                    Q.unsqueeze(0),
                    K.unsqueeze(0),
                    V.unsqueeze(0),
                    attn_mask=attn_mask,
                    scale=1.0,
                ).squeeze(0)
        else:
            with sdpa_kernel(_SDPA_BACKEND):
                weighted = F.scaled_dot_product_attention(
                    Q.unsqueeze(0),
                    K.unsqueeze(0),
                    V.unsqueeze(0),
                    is_causal=(n_new == n_total),
                    scale=1.0,
                ).squeeze(0)  # (n_heads, n_new, d_head)

        output = torch.einsum("hpk,hkd->pd", weighted, self.output_matrix)

        return output, (K, V)

    def trim_unused_heads(self):
        """Remove trailing unused (all-zero) heads after compilation.

        Heads are allocated contiguously from index 0, so slicing
        [:used_heads] is safe.  Keeps at least 1 head to avoid
        degenerate empty-tensor shapes in downstream ops.
        """
        n = max(self.used_heads, 1)
        if n < self.n_heads:
            self.query_matrix = self.query_matrix[:n].contiguous()
            self.key_matrix = self.key_matrix[:n].contiguous()
            self.value_matrix = self.value_matrix[:n].contiguous()
            self.output_matrix = self.output_matrix[:n].contiguous()
            self.n_heads = n

    def num_params(self) -> int:
        return (
            self.query_matrix.numel()
            + self.key_matrix.numel()
            + self.value_matrix.numel()
            + self.output_matrix.numel()
        )
