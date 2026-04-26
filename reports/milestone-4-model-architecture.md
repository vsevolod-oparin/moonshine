# M4: Model Architecture (v2 + v2.1)

**Status**: Complete
**Date**: 2026-04-26
**Machine**: Local workstation (CPU for architecture, GPU for forward/backward)
**Time**: ~1 day
**Cost**: $0
**Prerequisites**: M1, M2

## Objective

Implement the full ru-Moonshine model architecture for both v2 (vanilla Moonshine) and v2.1 (improved encoder), with streaming encoder inference, configurable via YAML, and verified by T1-T4 gate tests.

## Actions Completed

### 1. Configuration System

`models/config.py` — `ModelConfig` dataclass with all hyperparameters:

- Model dimensions: `enc_dim`, `dec_dim`, `enc_ffn_dim`, `dec_ffn_dim`, `enc_num_heads`, `dec_num_heads`, `enc_num_layers`, `dec_num_layers`
- Sliding window: `window_left=16`, `window_right_first_last=4`, `window_right_middle=0`
- Training: `ctc_weight=0.3`, `label_smoothing=0.1`, `attention_dropout=0.1`, `ffn_dropout=0.1`
- v2.1-specific: `v21_depthwise_kernel=7`, `v21_unet_stages`, `v21_cross_window_frames=2`, `stochastic_depth`
- RoPE: `rope_theta=10000.0`, `partial_rotary_factor=0.9`

5 presets: `v2_tiny`, `v2_small`, `v2_medium`, `v21_tiny`, `v21_small`. YAML config loader via `load_config()`.

### 2. Preprocessor

`models/preprocessor.py` — Conv1d subsampling matching the actual Moonshine v2 architecture (from HuggingFace transformers):

```
Audio (16kHz) → Conv1d(1, enc_dim, k=127, s=64) + tanh → GroupNorm → Conv1d(enc_dim, 2*enc_dim, k=7, s=3) + gelu → Conv1d(2*enc_dim, enc_dim, k=3, s=2) + gelu → Output
```

- Total subsampling: stride 64×3×2 = 384x
- 5s audio (80,000 samples) → 207 frames (~41.4Hz)
- 30s audio (480,000 samples) → 1,248 frames
- **Forced FP32**: STFT/conv operations wrapped in `torch.amp.autocast(enabled=False)` to prevent NaN under AMP. Source: NeMo, SpeechBrain.

### 3. Encoder v2

`models/encoder.py` — Vanilla Moonshine v2 encoder:

- 6 layers (Tiny), each: LayerNorm → Sliding-Window Self-Attention → Residual → LayerNorm → GELU FFN → Residual
- Layers 0-1 and N-2 to N-1: window (16, 4) — 80ms lookahead
- Middle layers: window (16, 0) — strictly causal
- RoPE positional encoding with `partial_rotary_factor=0.9` (36 of 40 head dims get rotary)
- SDPA attention via `torch.nn.functional.scaled_dot_product_attention` (auto-selects Flash/memory-efficient kernel)
- Optional QK normalization via config

### 4. Encoder v2.1

`models/encoder_v21.py` — Improved encoder with three changes from vanilla:

**Change 1: CausalDepthwiseConv** (kernel=7, GLU gating)
- Added after self-attention in each block: LayerNorm → DepthwiseConv(k=7) + SiLU GLU → Residual
- Captures 140ms local context for Russian consonant clusters, vowel reduction, palatalization

**Change 2: Multi-Scale U-Net**
- 3 stages with configurable layer distribution (Tiny: [2, 2, 2], Small: [3, 4, 3])
- Stage 1 (fine): 50Hz, layers 0-1
- Stage 2 (coarse): 25Hz (causal stride-2 downsample), layers 2-3
- Stage 3 (fine): 50Hz (nearest-neighbor upsample + 1x1 conv), layers 4-5 + skip connection from Stage 1
- Causal downsample: left-pad + Conv1d(k=2, s=2, groups=dim)
- Causal upsample: repeat_interleave(2) + linear projection

**Change 3: SSC-Style Cross-Window Context**
- Configurable `v21_cross_window_frames=2` — boundary frames attend to adjacent-window frames
- Implemented via `make_cross_window_mask` in `masks.py`

**Stochastic depth**: Per-layer drop probability linearly increasing from 0 to `max_drop_rate=0.1`. Surviving layers scaled by `1/(1-p)`.

### 5. Decoder

`models/decoder.py` — Causal Transformer decoder:

- Self-attention (causal, with RoPE) → Cross-attention (bidirectional, to encoder) → SwiGLU FFN
- SwiGLU: `fc1` outputs `2*ffn_dim`, split into value and gate, `value * SiLU(gate)`, then `fc2`
- Embedding layer with `pad_token_id` padding
- Causal mask + padding mask for self-attention
- Encoder padding mask for cross-attention

### 6. Adapter

`models/adapter.py`:

- Learned positional embedding (`max_position_embeddings` × `enc_dim`) added to encoder output
- Linear projection `enc_dim → dec_dim` (identity when enc_dim == dec_dim)

### 7. Full Model

`models/model.py` — `RuMoonshine` with ESPnet-style forward contract:

```python
loss, stats, weight = model(audio, tokens, audio_lengths, token_lengths)
# stats = {"loss": ..., "loss_aed": ..., "loss_ctc": ..., "acc": ...}
```

- `encode()`: preprocessor → encoder → (enc_output, enc_lengths)
- `decode()`: adapter → decoder → lm_head → logits
- `forward()`: encode + CTC head + decode + joint loss (AED + α·CTC)
- CTC loss: `zero_infinity=True`, `blank=0`, `reduction="mean"` (ESPnet pattern)
- Label smoothing: ε=0.1 on AED loss only (NOT on CTC — incompatible with CTC's blank/non-blank structure)
- Token accuracy tracked in stats
- Weight initialization: Xavier uniform for weight matrices, zero bias, normal(0, 0.02) for embeddings and LM head

### 8. Streaming Encoder Inference

`inference/streaming_encoder.py`:

- `CircularKVCache`: per-layer circular buffer of size `window_left + window_right + 1`, auto-evicts oldest KV
- `StreamingRotaryEmbedding`: computes RoPE for chunk offsets (not from 0) — `start_idx` for correct absolute positions
- `StreamingEncoderV2`: processes single preprocessor frames through encoder layers with cached KV
- `RepetitionDetector`: suppresses if last N emitted tokens are identical (N=4). Source: ESPnet.
- `HallucinationDetector`: 3 pattern detectors — 4+ identical, alternating pair, repeating triple. Source: NeMo.
- `StreamingASR`: full streaming pipeline — add audio chunks, accumulate encoder outputs, greedy decode on trigger

### 9. Attention Mask System

`models/masks.py`:

- `make_sliding_window_mask`: (window_left, window_right) → 4D additive mask
- `make_causal_mask`: standard upper-triangular causal mask
- `make_padding_mask`: from variable-length sequences
- `make_cross_window_mask`: SSC-style boundary frame cross-attention
- `combine_masks`: additive mask merging

### 10. YAML Configs

`configs/v2_tiny.yaml`, `configs/v21_tiny.yaml` — complete model + training + logging + data config.

## Parameter Counts

| Variant | Params | enc_dim / dec_dim | Enc Layers | Dec Layers | Enc Heads |
|---------|--------|-------------------|-----------|-----------|-----------|
| v2 Tiny | 22.3M | 320 / 320 | 6 | 6 | 8 |
| v2 Small | 107.9M | 620 / 512 | 10 | 10 | 10 |
| v2 Medium | 227.8M | 768 / 640 | 14 | 14 | 12 |
| v2.1 Tiny | 22.7M | 320 / 320 | 6 | 6 | 8 |
| v2.1 Small | 109.6M | 620 / 512 | 10 | 10 | 10 |

v2.1 adds ~2% params over v2 (depthwise conv + upsample projection).

## Gate Check

| Criterion | Status |
|-----------|--------|
| T1: Forward pass — v2 Tiny, logits shape correct, no NaN | Pass |
| T1: Forward pass — v2.1 Tiny, logits shape correct, no NaN | Pass |
| T2: Backward pass — v2 Tiny, all grads exist, finite norms | Pass |
| T2: Backward pass — v2.1 Tiny, all grads exist, finite norms | Pass |
| T3: Sliding-window mask — (16, 0) correct | Pass |
| T3: Sliding-window mask — (16, 4) correct | Pass |
| T4: Preprocessor output shape (207, 320), no NaN/Inf | Pass |

All 8/8 tests pass.

## Decisions

- **Followed actual Moonshine v2 Conv1d preprocessor** (not plan's mel-based description) — Conv1d subsampling is what the pretrained weights use, ensuring compatibility for transfer learning (M6.13)
- **Partial RoPE (0.9)** — matches Moonshine v2; only 90% of head_dim gets rotary embedding. Remaining 10% passes through unmodified
- **Stochastic depth disabled in backward test** — randomly dropping layers means some params get no gradient on any single pass. Disabled for deterministic test
- **CausalDepthwiseConv uses SiLU GLU** (not just conv) — gating adds expressiveness for phonetic pattern modeling
- **U-Net odd-length handling** — if downsampled sequence length is odd, upsample produces seq_len+1. Skip connection truncates to match
- **Streaming inference for v2 only** — v2.1 streaming requires per-stage caches (deferred to M9)

## Deliverables

- `models/config.py` — ModelConfig, presets, YAML loader
- `models/rope.py` — RotaryEmbedding, StreamingRotaryEmbedding, apply_rotary_pos_emb
- `models/attention.py` — MultiHeadAttention with SDPA, GQA, QK norm, KV cache
- `models/masks.py` — sliding-window, causal, padding, cross-window masks
- `models/preprocessor.py` — Conv1d subsampling (forced FP32)
- `models/encoder.py` — EncoderV2
- `models/encoder_v21.py` — EncoderV21 (depthwise conv + U-Net + SSC)
- `models/decoder.py` — Decoder (causal, RoPE, cross-attention, SwiGLU)
- `models/adapter.py` — positional embedding + linear projection
- `models/model.py` — RuMoonshine (full model, forward contract, CTC, weight init)
- `inference/streaming_encoder.py` — StreamingASR with KV cache + hallucination detection
- `configs/v2_tiny.yaml`, `configs/v21_tiny.yaml`
- `tests/test_m4_model.py` — T1-T4 tests (8/8 pass)
