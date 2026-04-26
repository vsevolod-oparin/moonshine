from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention
from models.config import ModelConfig
from models.masks import make_causal_mask, make_padding_mask, combine_masks
from models.rope import RotaryEmbedding


class DecoderFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim * 2)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        x = self.fc2(x)
        return self.dropout(x)


class DecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.self_attn = MultiHeadAttention(
            embed_dim=config.dec_dim,
            num_heads=config.dec_num_heads,
            num_kv_heads=config.dec_kv_heads(),
            dropout=config.attention_dropout,
            qk_norm=config.qk_norm,
        )
        self.self_attn_norm = nn.LayerNorm(config.dec_dim, bias=False)

        self.cross_attn = MultiHeadAttention(
            embed_dim=config.dec_dim,
            num_heads=config.dec_num_heads,
            num_kv_heads=config.dec_kv_heads(),
            dropout=config.attention_dropout,
            qk_norm=config.qk_norm,
        )
        self.cross_attn_norm = nn.LayerNorm(config.dec_dim, bias=False)

        self.ffn = DecoderFFN(config.dec_dim, config.dec_ffn_dim, config.ffn_dropout)
        self.ffn_norm = nn.LayerNorm(config.dec_dim, bias=False)
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor | None = None,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.self_attn_norm(x)
        x_self, _, _ = self.self_attn(
            x_norm,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=self_attn_mask,
        )
        x = residual + self.dropout(x_self)

        if enc_output is not None:
            residual = x
            x_norm = self.cross_attn_norm(x)
            x_cross, _, _ = self.cross_attn(
                x_norm,
                key_value_states=enc_output,
                attention_mask=cross_attn_mask,
            )
            x = residual + self.dropout(x_cross)

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)
        return x


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(
            config.vocab_size, config.dec_dim, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config, idx) for idx in range(config.dec_num_layers)]
        )
        self.norm = nn.LayerNorm(config.dec_dim, bias=False)

        rope_dim = config.rope_dim(config.dec_head_dim)
        self.rope = RotaryEmbedding(
            dim=rope_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        enc_output: torch.Tensor,
        enc_lengths: torch.Tensor | None = None,
        dec_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(tokens)
        seq_len = x.size(1)

        rope_cos, rope_sin = self.rope(seq_len)

        self_attn_mask = make_causal_mask(seq_len, x.device, x.dtype)
        if dec_lengths is not None:
            pad_mask = make_padding_mask(dec_lengths, seq_len).unsqueeze(1).unsqueeze(2)
            pad_mask = (~pad_mask).float().masked_fill(~pad_mask, float("-inf"))
            self_attn_mask = combine_masks(self_attn_mask, pad_mask)

        enc_seq_len = enc_output.size(1)
        cross_attn_mask = None
        if enc_lengths is not None:
            enc_pad = make_padding_mask(enc_lengths, enc_seq_len).unsqueeze(1).unsqueeze(2)
            cross_attn_mask = (~enc_pad).float().masked_fill(~enc_pad, float("-inf"))

        for layer in self.layers:
            x = layer(
                x,
                enc_output=enc_output,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
            )

        return self.norm(x)
