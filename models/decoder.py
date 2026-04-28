from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention
from models.config import ModelConfig
from models.masks import _MASK_NEG, make_causal_mask, make_padding_mask, combine_masks
from models.rope import RotaryEmbedding


class DecoderCache:
    def __init__(self, num_layers: int):
        self.self_keys: list[torch.Tensor | None] = [None] * num_layers
        self.self_values: list[torch.Tensor | None] = [None] * num_layers
        self.cross_keys: list[torch.Tensor | None] = [None] * num_layers
        self.cross_values: list[torch.Tensor | None] = [None] * num_layers

    @property
    def num_tokens(self) -> int:
        if self.self_keys[0] is not None:
            return self.self_keys[0].size(2)
        return 0

    def reset(self):
        num = len(self.self_keys)
        self.self_keys = [None] * num
        self.self_values = [None] * num
        self.cross_keys = [None] * num
        self.cross_values = [None] * num


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
        past_self_key: torch.Tensor | None = None,
        past_self_value: torch.Tensor | None = None,
        past_cross_key: torch.Tensor | None = None,
        past_cross_value: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        residual = x
        x_norm = self.self_attn_norm(x)
        x_self, new_self_key, new_self_value = self.self_attn(
            x_norm,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=self_attn_mask,
            past_key=past_self_key,
            past_value=past_self_value,
            use_cache=use_cache,
        )
        x = residual + self.dropout(x_self)

        if enc_output is not None or past_cross_key is not None:
            residual = x
            x_norm = self.cross_attn_norm(x)
            x_cross, new_cross_key, new_cross_value = self.cross_attn(
                x_norm,
                key_value_states=enc_output,
                attention_mask=cross_attn_mask,
                past_key=past_cross_key,
                past_value=past_cross_value,
                use_cache=use_cache,
            )
            x = residual + self.dropout(x_cross)
        else:
            new_cross_key = new_cross_value = None

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)

        if use_cache:
            return x, new_self_key, new_self_value, new_cross_key, new_cross_value
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
        cache: DecoderCache | None = None,
    ):
        x = self.embed(tokens)
        seq_len = x.size(1)

        if cache is not None and cache.self_keys[0] is not None:
            offset = cache.self_keys[0].size(2)
            rope_cos, rope_sin = self.rope(seq_len, offset=offset)
            self_attn_mask = None
        else:
            rope_cos, rope_sin = self.rope(seq_len)
            self_attn_mask = make_causal_mask(seq_len, x.device, x.dtype)
            if dec_lengths is not None:
                pad_mask = make_padding_mask(dec_lengths, seq_len).unsqueeze(1).unsqueeze(2)
                pad_mask = (~pad_mask).float().masked_fill(~pad_mask, _MASK_NEG)
                self_attn_mask = combine_masks(self_attn_mask, pad_mask)

        enc_seq_len = enc_output.size(1)
        cross_attn_mask = None
        if enc_lengths is not None:
            enc_pad = make_padding_mask(enc_lengths, enc_seq_len).unsqueeze(1).unsqueeze(2)
            cross_attn_mask = (~enc_pad).float().masked_fill(~enc_pad, _MASK_NEG)

        use_cache = cache is not None
        new_cache = DecoderCache(len(self.layers)) if use_cache else None

        for i, layer in enumerate(self.layers):
            psk = cache.self_keys[i] if cache is not None else None
            psv = cache.self_values[i] if cache is not None else None
            pck = cache.cross_keys[i] if cache is not None else None
            pcv = cache.cross_values[i] if cache is not None else None

            enc_for_layer = enc_output if pck is None else None

            result = layer(
                x,
                enc_output=enc_for_layer,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                past_self_key=psk,
                past_self_value=psv,
                past_cross_key=pck,
                past_cross_value=pcv,
                use_cache=use_cache,
            )

            if use_cache:
                x, sk, sv, ck, cv = result
                new_cache.self_keys[i] = sk
                new_cache.self_values[i] = sv
                new_cache.cross_keys[i] = ck
                new_cache.cross_values[i] = cv
            else:
                x = result

        x = self.norm(x)

        if use_cache:
            return x, new_cache
        return x
