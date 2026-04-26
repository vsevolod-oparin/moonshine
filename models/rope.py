import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq_len:
            self._set_cos_sin_cache(offset + seq_len)
        cos = self.cos_cached[offset : offset + seq_len].unsqueeze(0)
        sin = self.sin_cached[offset : offset + seq_len].unsqueeze(0)
        return cos, sin


class StreamingRotaryEmbedding(RotaryEmbedding):
    def forward(self, seq_len: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(offset, offset + seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)
