# M4 Code Review

**Date**: 2026-04-26
**Scope**: All model architecture code written in M4
**Status**: Fixes applied, see "Fix Status" column in tables below

## Dimensions Analyzed

| Dimension | Issues Found |
|-----------|-------------|
| Bugs | 5 |
| Performance | 3 |
| Interface / Integration | 3 (1 blocker) |
| Architecture / Design | 2 |
| Numerical / Export | 3 |
| Code Quality | 6 |
| Weak Spots | 3 |
| Config Correctness (2nd pass) | 3 |
| Token ID Alignment (2nd pass) | 1 (blocker) |
| **Total** | **29** |

---

## Bugs

### B1. `min_audio_samples=400` is wrong — preprocessor crashes on audio < 895 samples
**Files**: `models/config.py:46`, `models/model.py:65-69`
**Severity**: High

The preprocessor conv stack requires >= 895 samples to produce at least 1 output frame:
- conv1 (k=127, s=64) needs >= 511 samples → 7 frames
- conv2 (k=7, s=3) needs >= 13 conv1 frames → needs >= 895 total input samples
- conv3 (k=3, s=2) needs >= 3 conv2 frames

The check in `model.py:67-69` sets `out_lengths[i] = 0` for short audio but does NOT prevent the preprocessor from running — it crashes before reaching that check. Any audio with 400-894 samples (25-56ms) will throw `RuntimeError: Kernel size can't be greater than actual input size`.

### B2. `CausalDownsample` is NOT causal — leaks 1 future frame
**File**: `models/encoder_v21.py:45-49`
**Severity**: Medium

Left-pads by 1 frame, then Conv1d(k=2, s=2). Input `[pad, f0, f1, f2, ...]` produces pairs `(pad,f0), (f1,f2), (f3,f4)`. The second output `(f1,f2)` depends on f2, which is future relative to f1. This breaks streaming causality for the v2.1 encoder.

### B3. SSC cross-window context never activates (`chunk_size=0`)
**File**: `models/encoder_v21.py:171`
**Severity**: Low

`make_cross_window_mask(..., chunk_size=0)` hits the early return `if chunk_size <= 0: return mask`, falling back to a plain sliding window mask. The entire cross-window attention feature is dead code. Needs a meaningful chunk_size to activate.

### B4. Token padding mismatch — dataset pads with `0`, model expects `-100`
**Files**: `training/dataset.py:collate_fn`, `models/model.py:108-113`
**Severity**: High (M5 blocker)

`collate_fn` pads token sequences with `value=0` (blank token). `F.cross_entropy` uses `ignore_index=-100`. Padded positions will have loss computed on them as if they were real blank-token predictions. CTC loss also receives blank-padded labels, producing incorrect gradients. The training loop needs to convert padding from 0 → -100 before calling `model.forward()`, or the collate function needs to pad with -100.

### B5. Dataset returns mel spectrograms, model expects raw audio
**Files**: `training/dataset.py:ASRDataset`, `models/model.py`, `models/preprocessor.py`
**Severity**: High (M5 blocker)

`ASRDataset.__getitem__` returns `(mel_spectrogram, token_ids)`. The model's `forward()` passes audio through `self.preprocessor` which applies Conv1d to raw waveforms. The preprocessor does NOT expect mel features — it expects 1D raw audio `(B, audio_samples)`. The dataset produces 2D mel `(n_mels, T)`. These are incompatible inputs.

The dataset's `AudioProcessor` was written for a mel-based pipeline (M3, before the architecture was designed). The model uses Conv1d subsampling directly on raw audio (matching Moonshine v2). Either the dataset needs a raw-audio mode, or a new dataset class is needed.

---

## Performance

### P1. Sliding window mask uses Python for-loop — O(T²) sequential
**File**: `models/masks.py:11-15`
**Severity**: Medium

Iterates `seq_len` in Python. For T=207 (5s) this is fine. For T=1248 (30s) or with many layers, this becomes noticeable. Should be vectorized:
```python
rows = torch.arange(seq_len, device=device).unsqueeze(1)
cols = torch.arange(seq_len, device=device).unsqueeze(0)
mask = torch.where((cols >= rows - w_left) & (cols <= rows + w_right), 0.0, float("-inf"))
```
Same issue in `make_cross_window_mask`.

### P2. Redundant attention masks — builds one per layer despite shared window configs
**File**: `models/encoder.py:83-96`
**Severity**: Low

Layers 0-1 and N-2 to N-1 share window (16,4), middle layers share (16,0). Only 2 unique masks are needed, but N masks are built. For Tiny (6 layers) this is 6× where 2× would suffice. Minor for Tiny, noticeable for Medium (14 layers).

### P3. `CircularKVCache` allocates new tensors every frame (cat + slice)
**File**: `inference/streaming_encoder.py:26-39`
**Severity**: Low

Each frame does `torch.cat` (alloc) + slice (alloc). At 41Hz × 6 layers = 246 allocs/sec. A pre-allocated ring buffer with index tracking would eliminate allocations entirely.

---

## Interface / Integration

### I1. **[M5 BLOCKER]** Dataset-model interface mismatch (raw audio vs mel)
See B5 above. This blocks M5. Priority: **Fix before M5**.

### I2. **[M5 BLOCKER]** Token padding convention mismatch (0 vs -100)
See B4 above. This blocks M5. Priority: **Fix before M5**.

### I3. Dimension mismatch blocks transfer learning from English Moonshine
**File**: `models/config.py` presets vs `models/configuration_moonshine.py` defaults
**Severity**: Low (optional feature)

Original Moonshine Tiny uses `hidden_size=288, intermediate_size=1152`. Our presets use `enc_dim=320, enc_ffn_dim=1280`. Weight shapes are incompatible — transfer learning from English Moonshine (plan Section 6.13) is impossible without matching dimensions. The plan lists this as optional ("Phase 1b"), so this is acceptable for now. If transfer learning is needed later, add `v2_tiny_tl` presets matching original dims (288/1152).

---

## Architecture / Design

### A1. Encoder FFN uses GELU, but original Moonshine also uses GELU — correct
Confirmed: `MoonshineEncoderMLP` uses `gelu` activation, our `EncoderFFN` uses `F.gelu`. Match is correct.

### A2. Stochastic depth doesn't scale surviving layers
**File**: `models/encoder_v21.py:216-220`
**Severity**: Medium

When stochastic depth drops a layer, the surviving layers should scale their output by `1/(1-p)` to maintain expected activation magnitude. Currently, layers are just skipped with no compensation. With `max_drop_rate=0.1` and 6 layers, the expected magnitude reduction is small (~3%), but it contradicts the plan which specifies scaling.

---

## Numerical / Export

### N1. `torch.amp.autocast("cuda")` in preprocessor — silent on CPU but risks
**File**: `models/preprocessor.py:28`
**Severity**: Low

`torch.amp.autocast("cuda", enabled=False)` works on CPU without error (verified). However, if the model runs on a machine without CUDA at all (CPU-only inference), this still works because PyTorch treats `autocast("cuda")` on CPU as a no-op. No actual issue, but `torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=False)` would be more explicit.

### N2. ONNX export succeeds but with TracerWarnings
**Verified**: Export succeeds. Warnings about:
- RoPE `if offset + seq_len > self.max_seq_len` — Python boolean from tensor comparison (dynamic control flow)
- `torch.tensor(batch_size)` — creates constant in trace
- Mask for-loop — dynamic control flow

For production ONNX export (M12), the mask generation needs to be rewritten with fixed-shape ops, and RoPE needs to avoid the dynamic cache resize.

### N3. CTC loss receives encoder output of potentially 0 length
**File**: `models/model.py:115-137`
**Severity**: Low

If all audio in a batch is shorter than 895 samples and the preprocessor doesn't crash (e.g., after B1 is fixed with padding), `ctc_logits` would have seq_len=0. `F.ctc_loss` with zero-length input would produce NaN/inf. The `zero_infinity=True` handles inf → 0, but `log_softmax` on empty tensor may still error.

---

## Code Quality

### Q1. Stochastic depth doesn't scale surviving layers
See A2 above.

### Q2. `EncoderLayerV21` has `self.dropout` but never calls it on depthwise conv output
**File**: `models/encoder_v21.py:86,105`
**Severity**: Low

`self.dropout = nn.Dropout(config.ffn_dropout)` is defined but the depthwise conv output at line 105 has no dropout applied. The attention output gets dropout (line 103) and the FFN output gets dropout (via `EncoderFFN`), but the conv branch doesn't.

### Q3. Unused imports across 4 files
- `models/rope.py:1-2` — `math`, `Optional` unused
- `models/attention.py:1` — `math` unused
- `models/encoder.py:1-2` — `math`, `Optional` unused
- `models/encoder_v21.py:1,3` — `math`, `Optional` unused

### Q4. `StreamingASR.decode_buffer` is O(T²) — no decoder KV cache
**File**: `inference/streaming_encoder.py:196-211`
**Severity**: Low (known limitation)

Each token step re-runs the full decoder over the entire sequence. Standard autoregressive decoding should maintain a KV cache for decoder self-attention. This is a known initial implementation gap — acceptable for M4 but needed for production.

### Q5. `hold_n` and `hold_buffer` initialized but never used
**File**: `inference/streaming_encoder.py:167`
**Severity**: Low

Plan specifies hold-N mechanism for streaming. Currently dead code.

### Q6. `CausalUpsample.__init__(dim, target_dim)` — `target_dim` name is misleading
**File**: `models/encoder_v21.py:52-53`
**Severity**: Trivial

Parameter name suggests sequence length, but it's actually feature dimension. Currently always called with same dim twice.

---

## Weak Spots

### W1. No gradient checkpointing support
Plan specifies it for Phase 3 Medium (245M params). Not needed for Tiny (22M).

### W2. Streaming for v2.1 not implemented
`inference/streaming_encoder.py:155-156` raises `NotImplementedError`. v2.1 multi-scale encoder needs per-stage KV caches.

### W3. No test for streaming inference
Tests cover forward/backward/masks/preprocessor but not `StreamingASR`, `CircularKVCache`, or `RepetitionDetector`.

---

## Additional Dimensions Considered

| Dimension | Assessed? | Notes |
|-----------|-----------|-------|
| Security | Skipped | Not applicable — model architecture code, no user input, no network |
| Test coverage | Partial | T1-T4 cover happy paths. No edge case tests (short audio, empty batch, streaming) |
| API surface | OK | Forward contract `(loss, stats, weight)` is clean. `encode()` and `decode()` are useful. |
| DDP readiness | OK | All tensor creation uses correct devices. No shared mutable state. No in-place ops. |
| ONNX export | Partial | Succeeds with warnings. Dynamic control flow in masks and RoPE needs fixing for production export. |
| Numerical stability | Partial | Forced FP32 in preprocessor. RoPE forces float32. CTC `zero_infinity=True`. AMP handling present. |
| Config correctness | Issue | Our dims (320) differ from original Moonshine (288). Intentional for from-scratch training, but blocks transfer learning. |

---

## 2nd-Pass Findings (Config Correctness + Token ID Alignment)

Three new issues found by analyzing dimensions that go beyond per-file code review.

### C1. **[BLOCKER]** Token IDs don't match tokenizer model — blank/sos/pad all wrong
**Files**: `models/config.py:31-35`, `data/tokenizer_256.vocab`
**Severity**: High (training produces garbage)

Config says `blank_token_id=0, sos_eos_token_id=1, pad_token_id=2`. But the trained SentencePiece tokenizer has:

| Config value | Config label | Actual tokenizer piece at that index |
|-------------|-------------|--------------------------------------|
| blank=0 | `<blank>` | `<unk>` (unknown) |
| sos_eos=1 | `<sos/eos>` | `<s>` (sentence start) |
| pad=2 | `<pad>` | `</s>` (sentence end) |

The actual special tokens in the tokenizer are at different indices:

| Index | Tokenizer piece | Should be config field |
|-------|----------------|----------------------|
| 0 | `<unk>` | — |
| 1 | `<s>` | — |
| 2 | `</s>` | — |
| 3 | `<blank>` | `blank_token_id` |
| 4 | `<sos/eos>` | `sos_eos_token_id` |
| 5 | `<pad>` | `pad_token_id` |

**Impact**: CTC loss uses `blank=0` which is actually `<unk>`. Decoder starts with `sos_eos=1` which is actually `<s>`. Loss masks with `pad=2` which is actually `</s>`. Training will produce completely wrong gradients — the model will learn to predict `<unk>` where it should predict blanks, and the decoder will never see the correct start/end tokens.

**Fix**: Change config to `blank_token_id=3, sos_eos_token_id=4, pad_token_id=5`.

### C2. RoPE dim is odd for Small and Medium — crash at model construction
**Files**: `models/config.py` presets
**Severity**: High (crash for Small/Medium)

`partial_rotary_factor=0.9` with head_dim=62 (Small enc) gives `int(62*0.9) = int(55.8) = 55`. RoPE creates `inv_freq = 1.0 / theta^(arange(0, 55, 2) / 55)` — arange with step 2 on an odd dim produces 28 elements, then `cat((freqs, freqs))` gives 56, not 55. The `cos_cached` shape becomes `(T, 56)` but q has `[..., :55]` rotary dims. Shape mismatch crash.

Same for Medium enc (head_dim=64, rope=57) and Small/Medium dec (head_dim=64, rope=57).

**Fix**: Round `rope_dim` down to nearest even number: `rope_dim = int(head_dim * partial_rotary_factor) // 2 * 2`.

### C3. `max_position_embeddings=512` — adapter crashes on audio > 10 seconds
**Files**: `models/adapter.py:10`, `models/config.py:21`
**Severity**: High (crash during training on 7,470 clips)

The adapter has a fixed learned `pos_embed` of shape `(1, 512, enc_dim)`. For audio > 10s, the encoder produces > 415 frames. At 20s → 832 frames, the adapter tries `self.pos_embed[:, :832, :]` which crashes.

Training data has 7,470 clips > 10s (max 20s). All of these will crash.

The RoPE auto-extends (it rebuilds cache when seq_len exceeds max), but the adapter's `nn.Parameter` cannot auto-extend.

**Fix options**:
- (A) Set `max_position_embeddings=2048` (covers 50s audio, ~2.3MB param overhead for Tiny)
- (B) Use RoPE-style relative positions in adapter instead of learned absolute embeddings
- (C) Truncate encoder output to `max_position_embeddings` frames before adapter

---

## Priorities

### Fix Now (before M5)

| ID | What | Effort | Status |
|----|------|--------|--------|
| B5+I1 | Dataset returns raw audio, not mel spectrograms | Medium | **Fixed** — `ASRDataset(raw_audio=True)` mode, collate_fn handles both |
| B4+I2 | Token padding: convert 0 → -100 in training loop or collate_fn | Small | **Fixed** — `collate_fn` pads with `value=-100` |
| B1 | `min_audio_samples` = 895, or pad short audio in preprocessor | Small | **Fixed** — preprocessor pads to MIN_INPUT_SAMPLES=895, output_length returns 0 for short |
| C1 | Token IDs don't match tokenizer (blank=3, sos_eos=4, pad=5) | Small | **Fixed** — config/dataset/yaml updated, verified against tokenizer |
| C2 | RoPE dim odd for Small/Medium — crash at construction | Small | **Fixed** — `rope_dim()` helper rounds down to even |
| C3 | max_position_embeddings=512 crashes adapter on audio > 10s | Small | **Fixed** — increased to 2048 |

### Fix Soon (during M5, before training)

| ID | What | Effort | Status |
|----|------|--------|--------|
| B2 | Make CausalDownsample actually causal | Small | **Fixed** — replaced Conv1d(k=2,s=2) with stride-2 slice + learned linear projection |
| P1 | Vectorize mask generation | Small | **Fixed** — `torch.where` with `arange` broadcasting, no Python loops |
| A2 | Add stochastic depth output scaling | Small | **Fixed** — `x = x / (1.0 - drop_p)` after surviving layers |
| Q2 | Add dropout after depthwise conv | Trivial | **Fixed** — `self.dropout(self.depthwise_conv(...))` |
| Q3 | Remove unused imports | Trivial | **Fixed** — removed `math`, `Optional` from rope/attention/encoder/encoder_v21 |
| P2 | Deduplicate masks by window config | Small | **Fixed** — encoder caches masks by `(window_left, window_right)` key |

### Defer to Later Milestones

| ID | What | Milestone | Effort |
|----|------|-----------|--------|
| B3 | SSC cross-window chunk_size | M9 (streaming) | Small |
| N2 | ONNX export clean (no TracerWarnings) | M12 (export) | Medium |
| P3 | Pre-allocated ring buffer KV cache | M9 (streaming) | Medium |
| Q4 | Decoder KV cache for autoregressive decoding | M9 (streaming) | Medium |
| Q5 | Implement hold-N mechanism | M9 (streaming) | Small |
| W1 | Gradient checkpointing | M7 (Medium training) | Small |
| W2 | Streaming v2.1 | M9 (streaming) | Medium |
| W3 | Streaming inference tests | M9 (streaming) | Small |
| I3 | Match original Moonshine dims for transfer learning | M6.5 or later | Medium |

### Accept As-Is

| ID | What | Reason |
|----|------|--------|
| N1 | `autocast("cuda")` on CPU | Works fine, no-op on CPU |
| Q6 | Misleading `target_dim` name | Trivial, works correctly |
| N3 | CTC on zero-length input | Handled by `zero_infinity=True`, edge case after B1 fix |
