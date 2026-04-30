# M9 Optimization Bugfix + Performance Review

**Date:** 2026-04-30
**Scope:** Bugfixes applied to training pipeline + assessment of remaining optimization opportunities
**Hardware:** RTX 3090 Ti 24GB (Ampere, SM_86)

---

## 1. Bugs Fixed

### 1.1 CRITICAL: Training loss always logged as 0.0

**File:** `training/train.py:446` (before fix)

**Root cause:** `accum_stats_buf` was reset to all-zeros **before** being read for loss computation and logging.

```python
# BEFORE (broken):
accum_stats_buf = {"loss": 0.0, ...}   # ← reset HERE
global_step += 1
step_loss = accum_stats_buf["loss"] / accum_steps   # ← reads 0.0
loss_val = accum_stats_buf["loss"] / accum_steps     # ← 0.0 into TensorBoard
```

**Impact:** Training ran correctly (model parameters updated fine), but all logged metrics were zero:
- TensorBoard: flat `train/loss = 0.0`, `train/loss_aed = 0.0`, `train/loss_ctc = 0.0`, `train/acc = 0.0`
- Console: `loss=0.0000 aed=0.0000 ctc=0.0000 acc=0.000`
- `epoch_loss` / `accum_loss_sum`: always 0.0
- Only validation WER (every 4000 steps) provided any training signal

**Fix:** Moved the reset to after all reads from the buffer (line 522, after the `if global_step % log_every == 0:` block completes).

```python
# AFTER (correct):
step_loss = accum_stats_buf["loss"] / accum_steps    # reads accumulated value
epoch_loss += step_loss
if global_step % log_every == 0:
    loss_val = accum_stats_buf["loss"] / accum_steps  # reads accumulated value
    # ... all logging ...
accum_stats_buf = {"loss": 0.0, ...}  # ← reset AFTER all reads
```

### 1.2 MEDIUM: SpecAugment not applied despite config enabled

**Files:** `models/model.py`, `training/train.py`

**Root cause:** Two issues compounded:
1. `train.py:230` hardcoded `spec_augment=False` in the dataset constructor, ignoring the config
2. Even if the config were read, `raw_audio=True` mode returns raw waveforms from the dataset, skipping SpecAugment (which only runs on mel spectrograms)

The dataset-level SpecAugment can never work with `raw_audio=True` because the preprocessor (Conv1d subsampling) is inside the model, not the dataset.

**Fix:** Moved SpecAugment to the model's `encode()` method, applied after the preprocessor converts raw audio to learned features but before the encoder:

```python
# model.py encode():
x, out_lengths = self.preprocessor(audio, audio_lengths)
if self.training and self.spec_augment:
    x = self._apply_spec_augment(x)    # mask on [batch, n_frames, enc_dim]
enc_output = self.encoder(x, out_lengths)
```

- `spec_augment` flag is wired from config: `model.spec_augment = aug_cfg.get("spec_augment", False)`
- Controlled by `self.training` — automatically disabled during `model.eval()` (validation)
- Operates on learned features `[batch, n_frames, enc_dim]` — masks contiguous blocks along both feature (freq) and time dimensions
- Parameters: freq_mask=15, time_mask=50, n_freq_masks=2, n_time_masks=2 (standard SpecAugment)
- Placed **before** `torch.compile` boundary (between preprocessor and compiled encoder) — no interference with compilation

### 1.3 LOW: prefetch_factor default inconsistency

**File:** `training/train.py`

The default `prefetch_factor` varied across DataLoader paths:
- Dynamic batching: default 2
- Static batching: default 4
- Validation: default 4

**Fix:** Unified all defaults to 2. Since the config explicitly sets `prefetch_factor: 2`, this had no runtime effect but eliminates a silent divergence if the key is missing.

---

## 2. Test Verification

All 27/27 tests pass after fixes. Smoke test confirms:
- SpecAugment produces different losses on identical inputs (random masking active)
- Eval mode correctly skips SpecAugment
- Checkpoint save/load roundtrip works with compiled model

---

## 3. Performance Review: What's Already Optimized

| Optimization | Status | Impact |
|---|---|---|
| `torch.compile(encoder, "reduce-overhead")` | Done | 1.2x training, 5.1x inference, 41% less VRAM |
| `set_float32_matmul_precision("high")` (TF32) | Done | ~2x on fp32 matmuls |
| `max_tokens=24000` dynamic batching | Done | 1.85x more tokens/step |
| Fused AdamW (`fused=True`) | Done | Faster optimizer step |
| `set_to_none=True` in `zero_grad` | Done | Less memory allocation |
| `pin_memory=True` + `non_blocking=True` | Done | Overlapped H2D transfer |
| `persistent_workers=True` | Done | No worker restart per epoch |
| `prefetch_factor=2` | Done | Pre-fetches 2 batches per worker |
| Tokenized manifests (442K entries) | Done | Eliminates SentencePiece per-epoch |
| Speed perturbation on raw audio | Done | Applied before mel conversion |
| SpecAugment in model encode() | Done (this fix) | Regularization now active |
| `cudnn.benchmark = True` | Done | Auto-tunes conv algorithms |
| `cudnn_sdp(False)` | Done | Prevents segfault |
| bf16 autocast | Done | Faster decoder/loss computation |
| FP32 encoder + preprocessor | Done | Stability (prevents NaN) |
| Token-based batch sizing | Done | Efficient VRAM utilization |
| Checkpoint `clean_state_dict()` | Done | Compiled model portability |
| RNG state in checkpoint | Done | Reproducible resume |

**Estimated throughput with all optimizations: ~257M tokens/hour, ~9 hours for 100K steps.**

---

## 4. Performance Review: Remaining Opportunities

### 4.1 Worth Implementing (easy, safe)

#### Preprocessor output_length vectorization

`preprocessor.py:51-53` uses a Python list comprehension over batch items:

```python
out_lengths = torch.tensor(
    [self.output_length(l.item()) for l in audio_lengths],
    dtype=torch.long, device=audio.device,
)
```

This can be vectorized trivially since `output_length` is pure integer arithmetic:

```python
out_lengths = (audio_lengths.float() - 127) // 64 + 1
out_lengths = (out_lengths - 7) // 3 + 1
out_lengths = (out_lengths - 3) // 2 + 1
out_lengths = out_lengths.clamp(min=0).long()
```

**Expected gain:** ~0.1-0.2ms per batch (saves ~60 Python function calls for bs=60). Negligible per-step but free.

**Risk:** None — same arithmetic, just vectorized.

#### `torch.inference_mode()` in validation

`validate.py` uses `@torch.no_grad()`. Switching to `@torch.inference_mode()` disables more autograd tracking machinery (version counting on tensors, etc.).

```python
@torch.inference_mode()   # instead of @torch.no_grad()
def validate(...):
```

**Expected gain:** ~1-3% faster validation. Validation runs every 4000 steps (~25 times total), so the total savings are ~30-60 seconds.

**Risk:** None — `inference_mode` is a stricter version of `no_grad`. All operations in validate are read-only (no backward needed).

### 4.2 Worth Considering (moderate effort, meaningful gain)

#### Try bf16 encoder attention for Flash Attention

The encoder runs entirely in fp32, which means `F.scaled_dot_product_attention` uses the math implementation, **not** Flash Attention (which requires fp16/bf16).

On the 3090 Ti (Ampere), Flash Attention is available for bf16. If the encoder's QKV projections and FFN stay in fp32 but the attention scores use bf16, Flash Attention could significantly speed up the encoder backward pass.

```python
# In MultiHeadAttention.forward(), before SDPA:
q = q.bfloat16()
k = k.bfloat16()
v = v.bfloat16()
attn_output = F.scaled_dot_product_attention(q, k, v, ...)
attn_output = attn_output.float()
```

**Expected gain:** Potentially 1.5-2x on the encoder attention portion. The encoder is ~30% of training time, and attention is ~50% of encoder time, so ~15-30% overall training speedup.

**Risk:** HIGH. The original reason for fp32 encoder was NaN in attention operations. Switching attention to bf16 could reintroduce instability, especially during early training when activations are large. Would need careful testing with the actual training data.

**Recommendation:** Test after M9 completes. Run a short training run (1000 steps) with bf16 attention. If stable, adopt for Phase 2.

#### Compile the preprocessor

The preprocessor (3 Conv1d layers + GroupNorm + GELU) runs in fp32 and is relatively small. Compilation would fuse these operations.

```python
model.preprocessor = torch.compile(model.preprocessor, mode="reduce-overhead")
```

**Expected gain:** ~5-10% on preprocessor time. The preprocessor is ~10% of forward time, so ~0.5-1% overall.

**Risk:** Low. The preprocessor has fixed-size kernels and no dynamic control flow. Should compile cleanly.

**Recommendation:** Try it. If it doesn't compile (graph breaks on the padding logic), skip it.

### 4.3 Not Worth Implementing

| Idea | Why Not |
|---|---|
| `pynvml` instead of `nvidia-smi` subprocess | ~10ms overhead per call, only 1000 calls total = ~10s. Not worth adding a dependency. |
| Encoder mask caching | Mask depends on `seq_len` (varies every batch with dynamic batching). Cache hit rate would be near zero. |
| CTC greedy decode vectorization | Pure Python loop, but only runs during validation. Total validation decode time ~5-10s out of 9 hours. |
| `torch.cuda.empty_cache()` after validation | Usually hurts performance by forcing reallocation. PyTorch's allocator reuses freed memory. |
| `torch.compile` on decoder | Decoder uses causal masks + cross-attention with dynamic shapes. Limited compile benefit (1-2% per benchmarks). High risk of graph breaks. |
| Gradient checkpointing | 20% recompute overhead for ~25% memory savings → net negative throughput. Only useful if VRAM-constrained, which we're not (60-70% utilization). |
| CUDA graphs | Requires static tensor shapes. Incompatible with dynamic batching by design. |
| Compile full model | Backward pass fails with graph breaks on `grad_fn`. Encoder-only is the safe boundary. |

---

## 5. Summary

### Bugs fixed: 3 (1 critical, 1 medium, 1 low)

| Bug | Impact | Status |
|---|---|---|
| Loss logged as 0.0 | Zero training visibility for 9h | **Fixed** |
| SpecAugment not applied | Missing regularization | **Fixed** |
| prefetch_factor defaults inconsistent | Silent divergence risk | **Fixed** |

### Performance assessment

The training pipeline is **well-optimized for the 3090 Ti**. The three highest-impact optimizations are already in place (compile, TF32, dynamic batch scaling). The remaining easy wins are marginal (~0.5% total). The single meaningful opportunity is **bf16 encoder attention for Flash Attention**, which carries stability risk and should be tested separately after M9.

**Recommendation:** Proceed to M9 with current optimizations. The only pre-M9 action worth taking is the two easy fixes in section 4.1 (preprocessor vectorization + inference_mode), which together save ~1 minute over the full run and are trivially safe.

---

## 6. Files Modified

| File | Change |
|---|---|
| `training/train.py` | Loss logging fix (accum_stats_buf reset moved after reads), SpecAugment wiring from config, prefetch_factor unified |
| `models/model.py` | Added `_apply_spec_augment()` method, `spec_augment` flag, applied in `encode()` between preprocessor and encoder |
| `training/validate.py` | No changes (SpecAugment correctly skipped via `model.eval()` setting `self.training = False`) |
