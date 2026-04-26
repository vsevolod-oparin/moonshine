from typing import Optional

import torch
import torch.nn as nn

from models.attention import MultiHeadAttention
from models.config import ModelConfig
from models.encoder import EncoderLayer, EncoderV2
from models.encoder_v21 import EncoderLayerV21, EncoderV21
from models.preprocessor import Preprocessor
from models.rope import StreamingRotaryEmbedding


class CircularKVCache:
    def __init__(self, num_layers: int, num_heads: int, head_dim: int, window_size: int):
        self.num_layers = num_layers
        self.window_size = window_size
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self._offset = 0

    @property
    def offset(self) -> int:
        return self._offset

    def update(self, layer_idx: int, new_key: torch.Tensor, new_value: torch.Tensor):
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = new_key
            self.values[layer_idx] = new_value
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], new_key], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], new_value], dim=2)

        if self.keys[layer_idx].size(2) > self.window_size:
            excess = self.keys[layer_idx].size(2) - self.window_size
            self.keys[layer_idx] = self.keys[layer_idx][:, :, excess:]
            self.values[layer_idx] = self.values[layer_idx][:, excess:]

        self._offset += new_key.size(2)

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.keys[layer_idx], self.values[layer_idx]

    def reset(self):
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self._offset = 0


class StreamingEncoderV2:
    def __init__(self, encoder: EncoderV2, config: ModelConfig):
        self.encoder = encoder
        self.config = config
        self.rope = StreamingRotaryEmbedding(
            dim=config.rope_dim(config.enc_head_dim),
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.cache = CircularKVCache(
            num_layers=config.enc_num_layers,
            num_heads=config.enc_num_heads,
            head_dim=config.enc_head_dim,
            window_size=config.window_left + config.window_right_first_last + 1,
        )

    @torch.no_grad()
    def process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        x = frame
        offset = self.cache.offset
        rope_cos, rope_sin = self.rope(1, offset=offset)

        for idx, layer in enumerate(self.encoder.layers):
            residual = x
            x_norm = layer.attn_norm(x)

            past_key, past_value = self.cache.get(idx)
            x_attn, new_key, new_value = layer.self_attn(
                x_norm,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attention_mask=None,
                past_key=past_key,
                past_value=past_value,
                use_cache=True,
            )

            if new_key is not None:
                self.cache.update(idx, new_key, new_value)

            x = residual + x_attn

            residual = x
            x_norm = layer.ffn_norm(x)
            x = residual + layer.ffn(x_norm)

        return self.encoder.norm(x)

    def reset(self):
        self.cache.reset()


class RepetitionDetector:
    def __init__(self, max_repeat: int = 4):
        self.max_repeat = max_repeat
        self.history: list[int] = []

    def check(self, token_id: int) -> bool:
        self.history.append(token_id)
        if len(self.history) < self.max_repeat:
            return False
        last_n = self.history[-self.max_repeat:]
        return len(set(last_n)) == 1

    def reset(self):
        self.history = []


class HallucinationDetector:
    def __init__(self, identical_threshold: int = 4):
        self.identical_threshold = identical_threshold
        self.history: list[int] = []

    def check(self, token_id: int) -> bool:
        self.history.append(token_id)

        if len(self.history) >= self.identical_threshold:
            last = self.history[-self.identical_threshold :]
            if len(set(last)) == 1:
                return True

        if len(self.history) >= 4:
            last_4 = self.history[-4:]
            if last_4[0] == last_4[2] and last_4[1] == last_4[3] and last_4[0] != last_4[1]:
                return True

        if len(self.history) >= 6:
            last_6 = self.history[-6:]
            if (last_6[0] == last_6[2] == last_6[4] and
                last_6[1] == last_6[3] == last_6[5] and
                last_6[0] != last_6[1]):
                return True

        return False

    def reset(self):
        self.history = []


class StreamingASR:
    def __init__(self, model: nn.Module, config: ModelConfig):
        self.model = model
        self.config = config
        self.model.eval()

        if config.version == "v21":
            raise NotImplementedError("Streaming for v2.1 encoder not yet implemented")

        assert isinstance(model.encoder, EncoderV2)
        self.streaming_encoder = StreamingEncoderV2(model.encoder, config)

        self.repetition_detector = RepetitionDetector(max_repeat=4)
        self.hallucination_detector = HallucinationDetector()

        self.encoder_buffer: list[torch.Tensor] = []
        self.prev_text: str = ""
        self.hold_buffer: list[int] = []
        self.hold_n = 0

    @torch.no_grad()
    def add_audio_chunk(self, audio_chunk: torch.Tensor):
        frame, _ = self.model.preprocessor(audio_chunk.unsqueeze(0), None)
        for t in range(frame.size(1)):
            enc_out = self.streaming_encoder.process_frame(frame[:, t : t + 1, :])
            self.encoder_buffer.append(enc_out.squeeze(0).squeeze(0))

    @torch.no_grad()
    def decode_buffer(self, max_tokens: int = 100) -> list[int]:
        if not self.encoder_buffer:
            return []

        enc_output = torch.stack(self.encoder_buffer, dim=1)
        enc_output = self.model.adapter(enc_output)

        sos = torch.tensor(
            [[self.config.sos_eos_token_id]], device=enc_output.device
        )
        dec_output = self.model.decoder(sos, enc_output)
        logits = self.model.lm_head(dec_output)

        token = logits[:, -1, :].argmax(dim=-1).item()
        if token == self.config.sos_eos_token_id or token == self.config.blank_token_id:
            return []

        tokens = [token]
        generated = sos.clone()
        for _ in range(max_tokens - 1):
            generated = torch.cat(
                [generated, torch.tensor([[token]], device=enc_output.device)], dim=1
            )
            dec_output = self.model.decoder(generated, enc_output)
            logits = self.model.lm_head(dec_output)
            token = logits[:, -1, :].argmax(dim=-1).item()

            if token == self.config.sos_eos_token_id:
                break
            if self.repetition_detector.check(token):
                break
            if self.hallucination_detector.check(token):
                break

            tokens.append(token)

        return tokens

    def reset(self):
        self.streaming_encoder.reset()
        self.encoder_buffer = []
        self.prev_text = ""
        self.hold_buffer = []
        self.repetition_detector.reset()
        self.hallucination_detector.reset()
