import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_attn_mask(
    attn_mask: Optional[torch.Tensor],
    batch_size: int,
    q_len: int,
    k_len: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Normalize attention mask to shape (B, 1, Q, K), bool dtype.

    Rule: True means visible, False means masked.
    """
    if attn_mask is None:
        return None

    if attn_mask.dim() == 2:
        # (Q, K)
        attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
    elif attn_mask.dim() == 3:
        # (B, Q, K)
        attn_mask = attn_mask.unsqueeze(1)
    elif attn_mask.dim() == 4:
        # (B, H or 1, Q, K)
        pass
    else:
        raise ValueError("attn_mask must have 2, 3, or 4 dims.")

    attn_mask = attn_mask.to(device=device, dtype=torch.bool)

    # Basic shape sanity check for Q/K dims.
    if attn_mask.size(-2) != q_len or attn_mask.size(-1) != k_len:
        raise ValueError("attn_mask last two dims must be (q_len, k_len).")

    # If batch dim is 1 it will broadcast automatically.
    if attn_mask.size(0) not in (1, batch_size):
        raise ValueError("attn_mask batch dim must be 1 or batch_size.")

    return attn_mask


def _build_causal_mask(q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
    """Create lower-triangular mask with shape (1, 1, Q, K)."""
    return torch.tril(torch.ones((q_len, k_len), device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    dropout_layer: nn.Dropout,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Manual scaled dot-product attention.

    q/k/v: (B, H, Q/K, D)
    returns:
    - out: (B, H, Q, D)
    - attn_weights: (B, H, Q, K)
    """
    scale = math.sqrt(q.size(-1))
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale

    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(~attn_mask, -1e9)

    attn_weights = F.softmax(attn_scores, dim=-1)
    attn_weights = dropout_layer(attn_weights)
    out = torch.matmul(attn_weights, v)
    return out, attn_weights


class MultiHeadAttentionScratch(nn.Module):
    """Handwritten MHA without using nn.MultiheadAttention."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, D) -> (B, H, L, Dh)
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dh) -> (B, L, D)
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ):
        if key is None:
            key = query
        if value is None:
            value = key

        bsz, q_len, _ = query.shape
        k_len = key.size(1)

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        mask = _normalize_attn_mask(attn_mask, bsz, q_len, k_len, query.device)
        if is_causal:
            causal_mask = _build_causal_mask(q_len, k_len, query.device)
            mask = causal_mask if mask is None else (mask & causal_mask)

        out, attn_weights = _scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        if need_weights:
            return out, attn_weights
        return out


class GroupedQueryAttentionScratch(nn.Module):
    """Handwritten GQA.

    Query heads are more than key/value heads.
    K/V heads are repeated to match Q heads.
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads.")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads.")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.repeat_factor = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(d_model, num_query_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(num_query_heads * self.head_dim, d_model, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # (B, L, H*Dh) -> (B, H, L, Dh)
        bsz, seq_len, _ = x.shape
        x = x.view(bsz, seq_len, num_heads, self.head_dim)
        return x.transpose(1, 2).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        # (B, H, L, Dh) -> (B, L, H*Dh)
        bsz, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, num_heads * head_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ):
        if key is None:
            key = query
        if value is None:
            value = key

        bsz, q_len, _ = query.shape
        k_len = key.size(1)

        q = self._split_heads(self.q_proj(query), self.num_query_heads)
        k = self._split_heads(self.k_proj(key), self.num_kv_heads)
        v = self._split_heads(self.v_proj(value), self.num_kv_heads)

        # Repeat KV heads to match Q heads.
        if self.repeat_factor > 1:
            k = k.repeat_interleave(self.repeat_factor, dim=1)
            v = v.repeat_interleave(self.repeat_factor, dim=1)

        mask = _normalize_attn_mask(attn_mask, bsz, q_len, k_len, query.device)
        if is_causal:
            causal_mask = _build_causal_mask(q_len, k_len, query.device)
            mask = causal_mask if mask is None else (mask & causal_mask)

        out, attn_weights = _scaled_dot_product_attention(q, k, v, mask, self.attn_dropout)
        out = self._merge_heads(out)
        out = self.out_proj(out)

        if need_weights:
            return out, attn_weights
        return out


if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    seq_len = 6
    d_model = 32
    x = torch.randn(batch_size, seq_len, d_model)

    print("===== MHA Demo =====")
    mha = MultiHeadAttentionScratch(d_model=d_model, num_heads=4, dropout=0.1)
    mha_out, mha_attn = mha(x, is_causal=True, need_weights=True)
    print("mha_out shape:", tuple(mha_out.shape))
    print("mha_attn shape:", tuple(mha_attn.shape))

    print("\n===== GQA Demo =====")
    gqa = GroupedQueryAttentionScratch(
        d_model=d_model,
        num_query_heads=8,
        num_kv_heads=2,
        dropout=0.1,
    )
    gqa_out, gqa_attn = gqa(x, is_causal=True, need_weights=True)
    print("gqa_out shape:", tuple(gqa_out.shape))
    print("gqa_attn shape:", tuple(gqa_attn.shape))
