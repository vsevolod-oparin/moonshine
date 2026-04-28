import torch
import torch.nn as nn

from models.config import ModelConfig
from models.decoder import DecoderCache
from models.encoder import EncoderV2
from models.masks import _MASK_NEG
from models.rope import StreamingRotaryEmbedding


def _make_streaming_chunk_mask(
    kv_len: int,
    chunk_len: int,
    window_left: int,
    window_right: int,
    kv_start: int,
    device: torch.device,
) -> torch.Tensor:
    abs_positions = torch.arange(kv_start, kv_start + kv_len, device=device, dtype=torch.long)
    query_positions = abs_positions[-chunk_len:]
    rows = query_positions.unsqueeze(1)
    cols = abs_positions.unsqueeze(0)
    mask = torch.where(
        (cols >= rows - window_left) & (cols <= rows + window_right),
        torch.tensor(0.0, device=device, dtype=torch.float32),
        torch.tensor(_MASK_NEG, device=device, dtype=torch.float32),
    )
    return mask.unsqueeze(0).unsqueeze(0)


class CircularKVCache:
    def __init__(self, num_layers: int, window_size: int):
        self.num_layers = num_layers
        self.window_size = window_size
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

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
            self.values[layer_idx] = self.values[layer_idx][:, :, excess:]

        if layer_idx == 0:
            self._frame_count += new_key.size(2)

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.keys[layer_idx], self.values[layer_idx]

    def reset(self):
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self._frame_count = 0


class StreamingEncoderV2:
    def __init__(self, encoder: EncoderV2, config: ModelConfig):
        self.encoder = encoder
        self.config = config
        self.rope = StreamingRotaryEmbedding(
            dim=config.rope_dim(config.enc_head_dim),
            max_seq_len=config.max_position_embeddings,
            theta=config.rope_theta,
        )
        max_window = config.window_left + config.window_right_first_last + 1
        self.cache = CircularKVCache(
            num_layers=config.enc_num_layers,
            window_size=max_window,
        )

    @torch.no_grad()
    def process_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        was_training = self.encoder.training
        if was_training:
            self.encoder.eval()

        chunk_len = chunk.size(1)
        frame_offset = self.cache.frame_count
        rope_cos, rope_sin = self.rope(chunk_len, offset=frame_offset)

        for idx, layer in enumerate(self.encoder.layers):
            residual = chunk
            x_norm = layer.attn_norm(chunk)

            past_key, past_value = self.cache.get(idx)
            if past_key is not None:
                kv_len = past_key.size(2) + chunk_len
                kv_start = frame_offset - past_key.size(2)
            else:
                kv_len = chunk_len
                kv_start = frame_offset

            mask = _make_streaming_chunk_mask(
                kv_len,
                chunk_len,
                layer.window_left,
                layer.window_right,
                kv_start,
                chunk.device,
            )

            x_attn, new_key, new_value = layer.self_attn(
                x_norm,
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attention_mask=mask,
                past_key=past_key,
                past_value=past_value,
                use_cache=True,
            )

            if new_key is not None:
                self.cache.update(idx, new_key, new_value)

            chunk = residual + x_attn

            residual = chunk
            x_norm = layer.ffn_norm(chunk)
            chunk = residual + layer.ffn(x_norm)

        result = self.encoder.norm(chunk)
        if was_training:
            self.encoder.train()
        return result

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
    def __init__(self, model: nn.Module, config: ModelConfig, chunk_size: int = 32):
        self.model = model
        self.config = config
        self.chunk_size = chunk_size
        self.model.eval()

        if config.version == "v21":
            raise NotImplementedError("Streaming for v2.1 encoder not yet implemented")

        assert isinstance(model.encoder, EncoderV2)
        self.streaming_encoder = StreamingEncoderV2(model.encoder, config)

        self.repetition_detector = RepetitionDetector(max_repeat=4)
        self.hallucination_detector = HallucinationDetector()

        self._pending_frames: torch.Tensor | None = None
        self.encoder_buffer: list[torch.Tensor] = []
        self.prev_text: str = ""  # Planned: segment deduplication across chunks
        self.hold_buffer: list[int] = []  # Planned: hold-N mechanism for token revision
        self.hold_n = 0  # Planned: number of tokens to hold back between chunks

    @torch.no_grad()
    def add_audio_chunk(self, audio_chunk: torch.Tensor):
        with torch.amp.autocast("cuda", enabled=False):
            frames, _ = self.model.preprocessor(audio_chunk.unsqueeze(0), None)

        if self._pending_frames is None:
            self._pending_frames = frames
        else:
            self._pending_frames = torch.cat([self._pending_frames, frames], dim=1)

        while self._pending_frames.size(1) >= self.chunk_size:
            chunk_frames = self._pending_frames[:, : self.chunk_size, :]
            self._pending_frames = self._pending_frames[:, self.chunk_size :]
            enc_out = self.streaming_encoder.process_chunk(chunk_frames)
            self.encoder_buffer.append(enc_out.squeeze(0))

    @torch.no_grad()
    def flush_remaining(self):
        if self._pending_frames is None or self._pending_frames.size(1) == 0:
            return
        enc_out = self.streaming_encoder.process_chunk(self._pending_frames)
        self._pending_frames = None
        self.encoder_buffer.append(enc_out.squeeze(0))

    @torch.no_grad()
    def decode_buffer(self, max_tokens: int = 100) -> list[int]:
        if not self.encoder_buffer:
            return []

        enc_output = torch.cat(self.encoder_buffer, dim=0).unsqueeze(0)
        enc_output = self.model.adapter(enc_output)

        cache = DecoderCache(len(self.model.decoder.layers))

        sos = torch.tensor(
            [[self.config.sos_eos_token_id]], device=enc_output.device
        )
        dec_output, cache = self.model.decoder(sos, enc_output, cache=cache)
        logits = self.model.lm_head(dec_output)

        token = logits[:, -1, :].argmax(dim=-1).item()
        if token == self.config.sos_eos_token_id or token == self.config.blank_token_id:
            return []

        tokens = [token]
        for _ in range(max_tokens - 1):
            new_token = torch.tensor([[token]], device=enc_output.device)
            dec_output, cache = self.model.decoder(new_token, enc_output, cache=cache)
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
        self._pending_frames = None
        self.encoder_buffer = []
        self.prev_text = ""
        self.hold_buffer = []
        self.repetition_detector.reset()
        self.hallucination_detector.reset()
