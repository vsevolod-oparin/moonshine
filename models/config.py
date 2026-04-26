import math
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    version: str = "v2"

    vocab_size: int = 256
    enc_dim: int = 320
    dec_dim: int = 320
    enc_num_layers: int = 6
    dec_num_layers: int = 6
    enc_num_heads: int = 8
    dec_num_heads: int = 8
    enc_ffn_dim: int = 1280
    dec_ffn_dim: int = 1280
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    partial_rotary_factor: float = 0.9
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.1

    window_left: int = 16
    window_right_first_last: int = 4
    window_right_middle: int = 0

    ctc_weight: float = 0.3
    label_smoothing: float = 0.1
    blank_token_id: int = 3
    sos_eos_token_id: int = 4
    pad_token_id: int = 5

    qk_norm: bool = False
    stochastic_depth: bool = False
    max_drop_rate: float = 0.1

    v21_depthwise_kernel: int = 7
    v21_unet_stages: Optional[list[int]] = None
    v21_unet_downsample_ratios: Optional[list[int]] = None
    v21_cross_window_frames: int = 2

    min_audio_samples: int = 895

    @property
    def enc_head_dim(self) -> int:
        return self.enc_dim // self.enc_num_heads

    @property
    def dec_head_dim(self) -> int:
        return self.dec_dim // self.dec_num_heads

    def rope_dim(self, head_dim: int) -> int:
        dim = int(head_dim * self.partial_rotary_factor)
        return dim // 2 * 2

    def window_right(self, layer_idx: int) -> int:
        if layer_idx < 2 or layer_idx >= self.enc_num_layers - 2:
            return self.window_right_first_last
        return self.window_right_middle

    def enc_kv_heads(self) -> int:
        return self.enc_num_heads

    def dec_kv_heads(self) -> int:
        return self.dec_num_heads


def load_config(path: str) -> ModelConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    model_data = data.get("model", data)
    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in model_data.items() if k in known}
    return ModelConfig(**filtered)


PRESETS = {
    "v2_tiny": ModelConfig(
        version="v2",
        vocab_size=256,
        enc_dim=320, dec_dim=320,
        enc_num_layers=6, dec_num_layers=6,
        enc_num_heads=8, dec_num_heads=8,
        enc_ffn_dim=1280, dec_ffn_dim=1280,
    ),
    "v2_small": ModelConfig(
        version="v2",
        vocab_size=512,
        enc_dim=620, dec_dim=512,
        enc_num_layers=10, dec_num_layers=10,
        enc_num_heads=10, dec_num_heads=8,
        enc_ffn_dim=2480, dec_ffn_dim=2048,
    ),
    "v2_medium": ModelConfig(
        version="v2",
        vocab_size=512,
        enc_dim=768, dec_dim=640,
        enc_num_layers=14, dec_num_layers=14,
        enc_num_heads=12, dec_num_heads=10,
        enc_ffn_dim=3072, dec_ffn_dim=2560,
    ),
    "v21_tiny": ModelConfig(
        version="v21",
        vocab_size=256,
        enc_dim=320, dec_dim=320,
        enc_num_layers=6, dec_num_layers=6,
        enc_num_heads=8, dec_num_heads=8,
        enc_ffn_dim=1280, dec_ffn_dim=1280,
        stochastic_depth=True,
        v21_unet_stages=[2, 2, 2],
        v21_unet_downsample_ratios=[1, 2, 1],
    ),
    "v21_small": ModelConfig(
        version="v21",
        vocab_size=512,
        enc_dim=620, dec_dim=512,
        enc_num_layers=10, dec_num_layers=10,
        enc_num_heads=10, dec_num_heads=8,
        enc_ffn_dim=2480, dec_ffn_dim=2048,
        stochastic_depth=True,
        v21_unet_stages=[3, 4, 3],
        v21_unet_downsample_ratios=[1, 2, 1],
    ),
}


def get_config(name: str) -> ModelConfig:
    if name in PRESETS:
        import copy
        return copy.deepcopy(PRESETS[name])
    return load_config(name)
