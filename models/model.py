from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.adapter import Adapter
from models.config import ModelConfig
from models.decoder import Decoder
from models.encoder import EncoderV2
from models.encoder_v21 import EncoderV21
from models.preprocessor import Preprocessor


def shift_tokens_right(
    tokens: torch.Tensor, pad_token_id: int, sos_token_id: int
) -> torch.Tensor:
    shifted = torch.zeros_like(tokens)
    shifted[:, 1:] = tokens[:, :-1].clone()
    shifted[:, 0] = sos_token_id
    shifted.masked_fill_(shifted == -100, pad_token_id)
    return shifted


def init_weights(module: nn.Module):
    if isinstance(module, nn.Linear):
        if module.weight.dim() > 1:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.padding_idx is not None:
            nn.init.zeros_(module.weight[module.padding_idx])
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()


class RuMoonshine(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.preprocessor = Preprocessor(config)

        if config.version == "v21":
            self.encoder = EncoderV21(config)
        else:
            self.encoder = EncoderV2(config)

        self.adapter = Adapter(config)
        self.decoder = Decoder(config)

        self.ctc_head = nn.Linear(config.enc_dim, config.vocab_size)
        self.lm_head = nn.Linear(config.dec_dim, config.vocab_size, bias=False)

        self.apply(init_weights)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def encode(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, out_lengths = self.preprocessor(audio, audio_lengths)
        with torch.amp.autocast("cuda", enabled=False):
            enc_output = self.encoder(x, out_lengths)
        return enc_output, out_lengths

    def decode(
        self,
        tokens: torch.Tensor,
        enc_output: torch.Tensor,
        enc_lengths: torch.Tensor | None = None,
        dec_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        adapted = self.adapter(enc_output)
        dec_output = self.decoder(tokens, adapted, enc_lengths, dec_lengths)
        return self.lm_head(dec_output)

    def forward(
        self,
        audio: torch.Tensor,
        tokens: torch.Tensor,
        audio_lengths: torch.Tensor | None = None,
        token_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        enc_output, enc_lengths = self.encode(audio, audio_lengths)

        ctc_logits = self.ctc_head(enc_output)

        if self.training:
            dec_input = shift_tokens_right(
                tokens, self.config.pad_token_id, self.config.sos_eos_token_id
            )
        else:
            dec_input = tokens

        aed_logits = self.decode(dec_input, enc_output, enc_lengths, token_lengths)

        batch_size = audio.size(0)
        stats = {}

        if self.training:
            loss_aed = F.cross_entropy(
                aed_logits.view(-1, self.config.vocab_size),
                tokens.view(-1),
                ignore_index=-100,
                label_smoothing=self.config.label_smoothing,
            )

            ctc_log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            if token_lengths is not None:
                target_lengths = token_lengths
            else:
                target_lengths = torch.full(
                    (batch_size,), tokens.size(1), dtype=torch.long, device=tokens.device
                )
            input_lengths = (
                enc_lengths
                if enc_lengths is not None
                else torch.full(
                    (batch_size,), ctc_logits.size(1), dtype=torch.long, device=ctc_logits.device
                )
            )
            loss_ctc = F.ctc_loss(
                ctc_log_probs,
                tokens,
                input_lengths,
                target_lengths,
                blank=self.config.blank_token_id,
                reduction="mean",
                zero_infinity=True,
            )

            loss = loss_aed + self.config.ctc_weight * loss_ctc

            stats = {
                "loss": loss.detach(),
                "loss_aed": loss_aed.detach(),
                "loss_ctc": loss_ctc.detach(),
            }

            with torch.no_grad():
                preds = aed_logits.argmax(dim=-1)
                mask = tokens != -100
                correct = (preds == tokens) & mask
                acc = correct.sum().float() / mask.sum().float().clamp(min=1)
                stats["acc"] = acc.detach()

            return loss, stats, torch.tensor(batch_size, device=audio.device)

        return aed_logits, stats, torch.tensor(batch_size, device=audio.device)

    def get_output_logits(self, audio: torch.Tensor) -> torch.Tensor:
        enc_output, _ = self.encode(audio)
        return self.ctc_head(enc_output)
