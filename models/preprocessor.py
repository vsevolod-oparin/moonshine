import torch
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig


class Preprocessor(nn.Module):
    MIN_INPUT_SAMPLES = 895

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        embed_dim = config.enc_dim

        self.conv1 = nn.Conv1d(1, embed_dim, kernel_size=127, stride=64, bias=False)
        self.groupnorm = nn.GroupNorm(num_groups=1, num_channels=embed_dim, eps=1e-5)
        self.conv2 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=7, stride=3)
        self.conv3 = nn.Conv1d(2 * embed_dim, embed_dim, kernel_size=3, stride=2)

    def output_length(self, input_length: int) -> int:
        if input_length < self.MIN_INPUT_SAMPLES:
            return 0
        out = (input_length - 127) // 64 + 1
        out = (out - 7) // 3 + 1
        out = (out - 3) // 2 + 1
        return out

    def forward(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        min_len = self.MIN_INPUT_SAMPLES
        needs_pad = audio.size(-1) < min_len
        if needs_pad:
            pad_size = min_len - audio.size(-1)
            audio = F.pad(audio, (0, pad_size))
            if audio_lengths is not None:
                short_mask = audio_lengths < min_len
                audio_lengths = torch.where(short_mask, torch.zeros_like(audio_lengths), audio_lengths)

        with torch.amp.autocast("cuda", enabled=False):
            x = audio.float().unsqueeze(1)
            x = torch.tanh(self.conv1(x))
            x = self.groupnorm(x)
            x = F.gelu(self.conv2(x))
            x = F.gelu(self.conv3(x))
        x = x.permute(0, 2, 1)

        out_lengths = None
        if audio_lengths is not None:
            out_lengths = torch.tensor(
                [self.output_length(l.item()) for l in audio_lengths],
                dtype=torch.long, device=audio.device,
            )
            if x.size(1) < out_lengths.max().item():
                out_lengths = torch.clamp(out_lengths, max=x.size(1))

        return x, out_lengths
