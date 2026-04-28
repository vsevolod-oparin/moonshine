import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import apply_rotary_pos_emb


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

        self.dropout = dropout

        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_kv_groups == 1:
            return x
        bsz, n_kv, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bsz, n_kv, self.num_kv_groups, seq_len, head_dim)
            .reshape(bsz, self.num_heads, seq_len, head_dim)
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        return q, k

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        key_value_mask: torch.Tensor | None = None,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        bsz, q_len, _ = x.shape

        is_cross_attn = key_value_states is not None
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if is_cross_attn and past_key is not None:
            k = past_key
            v = past_value
        else:
            kv_input = key_value_states if is_cross_attn else x
            k = self.k_proj(kv_input).view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(kv_input).view(bsz, -1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if not is_cross_attn and rope_cos is not None and rope_sin is not None:
            q, k = self._apply_qk_norm(q, k)
            q, k = apply_rotary_pos_emb(q, k, rope_cos, rope_sin)

        if not is_cross_attn and past_key is not None:
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        new_key = k if use_cache else None
        new_value = v if use_cache else None

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.embed_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, new_key, new_value
