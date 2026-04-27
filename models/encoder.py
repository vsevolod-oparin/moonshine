import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention
from models.config import ModelConfig
from models.masks import _MASK_NEG, make_sliding_window_mask, combine_masks, make_padding_mask
from models.rope import RotaryEmbedding


class EncoderFFN(nn.Module):
    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.window_left = config.window_left
        self.window_right = config.window_right(layer_idx)

        self.self_attn = MultiHeadAttention(
            embed_dim=config.enc_dim,
            num_heads=config.enc_num_heads,
            num_kv_heads=config.enc_kv_heads(),
            dropout=config.attention_dropout,
            qk_norm=config.qk_norm,
        )
        self.attn_norm = nn.LayerNorm(config.enc_dim, bias=False)
        self.ffn = EncoderFFN(config.enc_dim, config.enc_ffn_dim, config.ffn_dropout)
        self.ffn_norm = nn.LayerNorm(config.enc_dim, bias=False)
        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = x
        x_norm = self.attn_norm(x)
        x_attn, _, _ = self.self_attn(
            x_norm,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=attention_mask,
        )
        x = residual + self.dropout(x_attn)

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)
        return x


class EncoderV2(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [EncoderLayer(config, idx) for idx in range(config.enc_num_layers)]
        )
        self.norm = nn.LayerNorm(config.enc_dim, bias=False)
        rope_dim = config.rope_dim(config.enc_head_dim)
        self.rope = RotaryEmbedding(
            dim=rope_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def _build_attention_mask(
        self, seq_len: int, lengths: torch.Tensor | None, device: torch.device
    ) -> dict[int, torch.Tensor]:
        seen = {}
        masks = {}
        for idx, layer in enumerate(self.layers):
            key = (layer.window_left, layer.window_right)
            if key not in seen:
                sw = make_sliding_window_mask(
                    seq_len, layer.window_left, layer.window_right, device
                )
                if lengths is not None:
                    pad_mask = make_padding_mask(lengths, seq_len).unsqueeze(1).unsqueeze(2)
                    pad_mask = (~pad_mask).float().masked_fill(~pad_mask, _MASK_NEG)
                    sw = combine_masks(sw, pad_mask)
                seen[key] = sw
            masks[idx] = seen[key]
        return masks

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cos, rope_sin = self.rope(seq_len)

        attn_masks = self._build_attention_mask(seq_len, lengths, x.device)

        for idx, layer in enumerate(self.layers):
            x = layer(
                x,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attention_mask=attn_masks.get(idx),
            )

        return self.norm(x)
