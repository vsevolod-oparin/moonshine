import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention
from models.config import ModelConfig
from models.encoder import EncoderFFN
from models.masks import (
    _MASK_NEG,
    make_cross_window_mask,
    make_sliding_window_mask,
    combine_masks,
    make_padding_mask,
)
from models.rope import RotaryEmbedding


class CausalDepthwiseConv(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            dim, dim * 2, kernel_size=kernel_size, padding=self.padding, groups=dim
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        seq_len = x.size(1)
        x = x.transpose(1, 2)
        x = self.conv(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        x, gate = x.chunk(2, dim=-1)
        x = x * F.silu(gate)
        return x + residual


class CausalDownsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.stride_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stride_proj(x[:, ::2, :])
        return x


class CausalUpsample(nn.Module):
    def __init__(self, dim: int, target_dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, target_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=1)
        x = self.proj(x)
        return x


class EncoderLayerV21(nn.Module):
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

        self.depthwise_conv = CausalDepthwiseConv(
            config.enc_dim, kernel_size=config.v21_depthwise_kernel
        )
        self.conv_norm = nn.LayerNorm(config.enc_dim, bias=False)

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

        x = self.dropout(self.depthwise_conv(self.conv_norm(x)))

        residual = x
        x_norm = self.ffn_norm(x)
        x = residual + self.ffn(x_norm)
        return x


class EncoderV21(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        stages = config.v21_unet_stages or [config.enc_num_layers // 3] * 3
        ratios = config.v21_unet_downsample_ratios or [1, 2, 1]
        assert len(stages) == 3
        assert sum(stages) == config.enc_num_layers

        self.stage_specs = stages
        self.stage_ratios = ratios

        layer_idx = 0
        self.stages = nn.ModuleList()
        for s, ratio in zip(stages, ratios):
            layers = nn.ModuleList()
            for _ in range(s):
                layers.append(EncoderLayerV21(config, layer_idx))
                layer_idx += 1
            self.stages.append(layers)

        self.downsample = CausalDownsample(config.enc_dim)
        self.upsample = CausalUpsample(config.enc_dim, config.enc_dim)
        self.skip_proj = nn.Linear(config.enc_dim, config.enc_dim, bias=False)

        self.norm = nn.LayerNorm(config.enc_dim, bias=False)
        rope_dim = config.rope_dim(config.enc_head_dim)
        self.rope = RotaryEmbedding(
            dim=rope_dim,
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )

        self.stochastic_depth = config.stochastic_depth
        self.max_drop_rate = config.max_drop_rate
        total_layers = sum(stages)
        self._drop_rates = (
            [(i / total_layers) * self.max_drop_rate for i in range(total_layers)]
            if self.stochastic_depth
            else [0.0] * total_layers
        )

    def _build_stage_mask(
        self,
        seq_len: int,
        layer: EncoderLayerV21,
        lengths: torch.Tensor | None,
        device: torch.device,
        cross_window_frames: int = 0,
    ) -> torch.Tensor:
        if cross_window_frames > 0 and cross_window_frames < layer.window_left:
            mask = make_cross_window_mask(
                seq_len,
                layer.window_left,
                layer.window_right,
                cross_window_frames,
                chunk_size=0,
                device=device,
            )
        else:
            mask = make_sliding_window_mask(
                seq_len, layer.window_left, layer.window_right, device
            )
        if lengths is not None:
            pad_mask = make_padding_mask(lengths, seq_len).unsqueeze(1).unsqueeze(2)
            pad_mask = (~pad_mask).float().masked_fill(~pad_mask, _MASK_NEG)
            mask = combine_masks(mask, pad_mask)
        return mask

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        global_layer = 0
        skip = None

        for stage_idx, (stage_layers, ratio) in enumerate(
            zip(self.stages, self.stage_ratios)
        ):
            seq_len = x.size(1)
            rope_cos, rope_sin = self.rope(seq_len)
            stage_lengths = lengths

            if stage_idx == 1 and ratio == 2:
                x = self.downsample(x)
                if lengths is not None:
                    stage_lengths = (lengths.float() / 2).ceil().long()
                rope_cos, rope_sin = self.rope(x.size(1))
                seq_len = x.size(1)

            if stage_idx == 2:
                prev_ratio = self.stage_ratios[1] if len(self.stage_ratios) > 1 else 1
                if prev_ratio == 2:
                    x = self.upsample(x)
                    if lengths is not None:
                        stage_lengths = lengths
                    rope_cos, rope_sin = self.rope(x.size(1))
                    seq_len = x.size(1)

            if stage_idx == 0:
                skip = self.skip_proj(x.clone())

            for layer in stage_layers:
                drop_p = self._drop_rates[global_layer]
                if self.training and drop_p > 0 and random.random() < drop_p:
                    global_layer += 1
                    continue

                attn_mask = self._build_stage_mask(
                    seq_len,
                    layer,
                    stage_lengths,
                    x.device,
                    cross_window_frames=self.config.v21_cross_window_frames,
                )
                x = layer(
                    x,
                    rope_cos=rope_cos,
                    rope_sin=rope_sin,
                    attention_mask=attn_mask,
                )
                if self.training and drop_p > 0:
                    x = x / (1.0 - drop_p)
                global_layer += 1

            if stage_idx == 2 and skip is not None:
                if skip.size(1) != x.size(1):
                    if skip.size(1) > x.size(1):
                        skip = skip[:, : x.size(1), :]
                    else:
                        skip = F.pad(skip, (0, 0, 0, x.size(1) - skip.size(1)))
                x = x + skip

        return self.norm(x)
