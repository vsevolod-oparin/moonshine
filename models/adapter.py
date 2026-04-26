import torch
import torch.nn as nn

from models.config import ModelConfig


class Adapter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_position_embeddings, config.enc_dim))
        self.proj = nn.Linear(config.enc_dim, config.dec_dim, bias=False)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, enc_output: torch.Tensor) -> torch.Tensor:
        seq_len = enc_output.size(1)
        pos = self.pos_embed[:, :seq_len, :]
        return self.proj(enc_output + pos)
