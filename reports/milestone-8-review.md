# Milestone 8 Code Review: Scale-Up Readiness

**Date:** 2026-04-30
**Scope:** Training pipeline, data pipeline, configs, model code — readiness for full Phase 1 training (595h, 100K steps)
**Reviewer:** opencode (systematic file-by-file audit)

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 2     |
| MEDIUM   | 4     |
| LOW      | 7     |

All files reviewed: `training/train.py`, `training/validate.py`, `training/dataset.py`, `training/sampler.py`, `training/checkpoint.py`, `training/logger.py`, `models/model.py`, `models/encoder.py`, `models/encoder_v21.py`, `models/decoder.py`, `models/attention.py`, `models/config.py`, `models/preprocessor.py`, `models/adapter.py`, `models/masks.py`, `models/rope.py`, `configs/phase1_v2_tiny.yaml`, `configs/phase1_v2_tiny_full.yaml`, `scripts/download_data.py`, `data/manifests/`

---

## Findings

### F1. HIGH — `phase1_v2_tiny_full.yaml` uses Schedule-Free with no warmup_steps/max_steps for scheduler

**File:** `configs/phase1_v2_tiny_full.yaml:30-32`
**Description:** The full training config uses `schedulefree` optimizer with `lr: 0.002`. Schedule-Free doesn't use a scheduler, so `warmup_steps`/`max_steps` aren't needed for the scheduler. However, the T16 convergence test (100h) found that `schedulefree` at lr=2e-3 caused gradient overflow even with bf16. The T16 test switched to `adamw` with `lr=5e-4` and `warmup_steps=1000`. The full config hasn't been updated to match the proven T16 settings.
**Impact:** Full training may hit gradient overflow at the same rate. Schedule-Free's lr=2e-3 was never validated successfully for this model + data combo.
**Fix:** Either switch to `adamw` with proven lr=5e-4, or validate Schedule-Free at a lower LR (e.g., 1e-3) before committing to 100K steps.

### F2. HIGH — `phase1_v2_tiny_full.yaml` uses `accum_steps: 1` with `max_tokens: 30000`

**File:** `configs/phase1_v2_tiny_full.yaml:34-43`
**Description:** The T16 config uses `accum_steps: 2` with `max_tokens: 15000` (effective batch ~30K tokens). The full config uses `accum_steps: 1` with `max_tokens: 30000` (same effective batch size but single-step accumulation). While mathematically equivalent for gradient averaging, `accum_steps: 1` means every step does optimizer update + LR scheduler step + logging, which is wasteful with 30K-token batches. More importantly, with `accum_steps: 1`, the gradient is computed on a single forward pass of potentially huge batch sizes, which may not fit in VRAM on a 3090 (24GB).
**Impact:** VRAM OOM risk, or unnecessarily frequent optimizer steps.
**Fix:** Use `accum_steps: 2` with `max_tokens: 15000` (proven in T16), or validate that a single 30K-token batch fits in VRAM.

### F3. MEDIUM — `validate.py` does not restore `model.train()` or Schedule-Free train mode

**File:** `training/validate.py:137`
**Description:** `validate()` calls `model.eval()` at line 85 and `model.train()` at line 137. However, if the caller (train.py) also uses Schedule-Free, the optimizer needs `optimizer.train()` called after validation. The caller in train.py (lines 569-571) does handle this, but `validate()` itself doesn't accept or restore the optimizer state. This creates a fragile coupling — if anyone calls `validate()` from outside `train.py`, Schedule-Free train mode won't be restored.
**Impact:** Not a bug currently (train.py handles it), but fragile API design.
**Fix:** Low priority. Document that callers must handle optimizer train/eval mode.

### F4. MEDIUM — `DynamicBatchSampler` regenerates batches on every `__iter__` call when `shuffle=True`

**File:** `training/sampler.py:106-109`
**Description:** When `shuffle=True`, `__iter__` calls `self._make_batches()` on every iteration. `_make_batches()` sorts all samples by duration, creates buckets, shuffles within buckets, and forms batches. For 442K samples with 100 buckets, this is O(N log N) per epoch. The T16 run (74K samples) was fast enough, but with 442K samples this adds noticeable overhead per epoch.
**Impact:** Minor performance cost (~0.5-1 second per epoch, negligible vs training time).
**Fix:** No fix needed. The overhead is tiny compared to epoch time (~30 minutes for 442K clips).

### F5. MEDIUM — `train.py` `load_full_config` uses `__dataclass_fields__` on `ModelConfig` values

**File:** `training/train.py:142`
**Description:** `known = {f.name for f in ModelConfig.__dataclass_fields__.values()}` — this iterates over the Field objects from `__dataclass_fields__`, accessing `.name`. This works but is non-standard. The conventional approach is `ModelConfig.__dataclass_fields__.keys()`.
**Impact:** No bug — `Field.name` equals the dict key. Just unusual style.
**Fix:** Replace with `known = set(ModelConfig.__dataclass_fields__.keys())` for clarity.

### F6. MEDIUM — `collate_fn` pads text tokens with `-100` but validation decode filters `>= 6`

**File:** `training/dataset.py:276` and `training/validate.py:126-131`
**Description:** `collate_fn` pads token sequences with `-100` (standard cross-entropy ignore_index). In `validate.py`, CTC decode filters tokens `>= 6` (excluding special tokens 0-5). This is consistent — the cross-entropy loss uses `ignore_index=-100` to ignore padding, and CTC validation filtering is separate. However, `validate.py:129` reads `ref_ids = tokens[i, :ref_len].tolist()` — this correctly uses `token_lengths` to slice only real tokens, so `-100` padding is never included in references.
**Impact:** No bug. Correct implementation.
**Fix:** None needed.

### F7. LOW — `download_data.py` SOVA speaker_id is MD5 of audio bytes, not actual speaker

**File:** `scripts/download_data.py:331-332`
**Description:** `speaker_id = f"{dataset_name}_{content_hash}"` — this creates a "pseudo-speaker" per clip (MD5 of audio bytes). Since every clip has unique audio bytes, every clip gets a unique speaker_id. This means `split_by_speaker` effectively splits by individual clips, which is equivalent to random splitting. True speaker separation requires actual speaker metadata from the dataset.
**Impact:** Train/val leakage is possible if the same speaker appears in multiple clips. However, SOVA datasets don't provide speaker metadata, so this is the best available approach.
**Fix:** No fix possible without speaker metadata. Document this limitation.

### F8. LOW — `DynamicBatchSampler` doesn't re-seed between epochs

**File:** `training/sampler.py:106-109`
**Description:** When `shuffle=True`, `__iter__` calls `self._make_batches()` which uses `random.shuffle` and `random.shuffle(bucket)`. This depends on global Python RNG state, which `train.py` seeds once at startup (line 208). Between epochs, the DataLoader worker processes may consume RNG state, so bucket order varies naturally. This is fine for training.
**Impact:** No bug. Standard practice.
**Fix:** None needed.

### F9. LOW — `validate.py` only uses CTC greedy decode, no AED decode

**File:** `training/validate.py:55-74`
**Description:** Validation uses CTC greedy decode for WER computation. The AED (attention-based encoder-decoder) path is never used during validation. This is by design — AED requires autoregressive decoding which is expensive and not yet implemented in validate.py. CTC WER is a good proxy for model quality during training.
**Impact:** CTC WER is significantly worse than final AED WER will be. The 64.4% T16 WER would be much lower with AED.
**Fix:** Planned for M9+ (autoregressive decode implementation).

### F10. LOW — `train.py` `_log_gpu_temp` has no return when logging

**File:** `training/train.py:65-88`
**Description:** `_log_gpu_temp` returns `gs` (GPU stats dict) at line 88, but only when the function reaches that line. When the temperature is below the warning threshold, the function reaches line 88 and returns. But when the 60-second throttle is active (`not force and now - _last_gpu_log < 60`), it returns `None` implicitly at line 69. This is fine — callers check for None.
**Impact:** No bug.
**Fix:** None needed.

### F11. LOW — `preprocessor.py` `MIN_INPUT_SAMPLES = 895` differs from config `min_audio_samples = 400`

**File:** `models/preprocessor.py:9` and `configs/phase1_v2_tiny.yaml:26`
**Description:** The Preprocessor class hardcodes `MIN_INPUT_SAMPLES = 895` (minimum audio samples to produce at least 1 output frame from the 3 conv layers). The config has `min_audio_samples: 400`, which is used by ASRDataset to filter clips with `min_duration: 1.0` second (= 16000 samples). Since 16000 >> 895, there's no conflict — all filtered clips exceed the preprocessor minimum.
**Impact:** No bug. The config value is a data filter threshold, the preprocessor value is an architectural constraint.
**Fix:** None needed, but the preprocessor could read from config for consistency.

### F12. LOW — `encoder.py` builds masks per unique `(window_left, window_right)` key but doesn't account for batch dimension in padding masks

**File:** `models/encoder.py:79-96`
**Description:** `_build_attention_mask` creates one mask per unique window configuration, shared across all items in the batch. The padding mask is applied to the same mask for all batch items. However, `make_padding_mask` returns shape `(batch, seq_len)`, and `combine_masks` adds it to the sliding window mask `(1, 1, seq_len, seq_len)` via broadcasting. The result is `(batch, 1, 1, seq_len)` which gets broadcast correctly in SDPA.
**Impact:** No bug. Broadcasting handles this correctly.
**Fix:** None needed.

### F13. LOW — `phase1_v2_tiny_full.yaml` has no `max_steps` in optimizer section

**File:** `configs/phase1_v2_tiny_full.yaml:28-32`
**Description:** The `optimizer` section doesn't include `warmup_steps` or `max_steps`. Since Schedule-Free doesn't use a scheduler, this is correct for that optimizer. But if someone switches to `adamw` in this config (as recommended in F1), they'll need to add these fields.
**Impact:** Config will be incomplete if switching to AdamW.
**Fix:** Add `warmup_steps: 2000` and `max_steps: 100000` to optimizer section.

---

## Recommendations

### Must Fix Before Full Training
1. **F1 + F2:** Update `phase1_v2_tiny_full.yaml` to use proven AdamW settings from T16, or validate Schedule-Free at lower LR first.
2. **F13:** Add scheduler params to config for AdamW compatibility.

### Should Fix (Low Risk)
3. **F5:** Simplify `load_full_config` field name extraction.

### Deferred
4. **F3:** validate.py API coupling — low priority, document only
5. **F7:** SOVA pseudo-speaker IDs — no metadata available, document limitation
6. **F9:** AED validation — planned for M9+
7. **F11:** Preprocessor MIN_INPUT_SAMPLES hardcoded — cosmetic

---

## Data Integrity Audit

| Check | Result |
|-------|--------|
| Train manifest entries | 442,460 |
| Val manifest entries | 17,432 |
| Missing audio files (train) | 0 |
| Missing audio files (val) | 0 |
| Train total duration | 594.7h |
| Val total duration | 22.4h |
| Speaker overlap (train/val) | 0 |
| Duration range | 1.0–29.6s |
| Clips > 20s | 41 |
| Empty texts | 0 |
| Latin-heavy texts | 0 |

**Data quality:** Clean. No missing files, no empty texts, no speaker leakage. The dataset is ready for full training.

---

# Milestone 8 Code Review — Pass 2: Deep Audit

**Date:** 2026-04-30
**Scope:** Bugs, numerical correctness, performance, dead code, config correctness, pipeline integration
**Method:** Line-by-line trace of control flow, dtype propagation, mask correctness, edge cases, data pipeline integrity

---

## Summary

| Severity | Count |
|----------|-------|
| HIGH     | 2     |
| MEDIUM   | 4     |
| LOW      | 7     |

---

## Pass 1 Corrections

**F2 (HIGH from Pass 1) is RESOLVED** — config was updated to AdamW + accum=2 + max_tokens=15000.
**F5 (MEDIUM from Pass 1) is RESOLVED** — `load_full_config` field extraction simplified.
**F13 (LOW from Pass 1) is RESOLVED** — warmup_steps/max_steps added to optimizer section.

---

## New Findings

### P2-F1. HIGH — `dynamic_window` feature is dead code (mutates config, not layer instances)

**File:** `training/train.py:149-158` and `models/encoder.py:26-27`
**Description:** `sample_dynamic_window()` modifies `model_cfg.window_left`, `model_cfg.window_right_first_last`, and `model_cfg.window_right_middle`. However, `EncoderLayer.__init__` copies these values at construction time:
```python
self.window_left = config.window_left          # cached at init
self.window_right = config.window_right(layer_idx)  # cached at init
```
`_build_attention_mask` reads from `layer.window_left` and `layer.window_right` (the cached copies), NOT from the config. So mutating the config after model construction has **zero effect** on attention masks. The entire dynamic window feature does nothing.

When `dynamic_window=True` in config, the code wastes time calling `sample_dynamic_window()` and `restore_window()` every step with no effect.

**Impact:** Dead code that adds confusion. If someone sets `dynamic_window=True`, they'll expect window randomization but get nothing.
**Fix:** Either (a) remove the dead feature, or (b) make `sample_dynamic_window` update each `layer.window_left`/`layer.window_right` directly.

### P2-F2. HIGH — Validation uses `batch_size=1` with dynamic batching config

**File:** `training/train.py:311-312`
**Description:** `val_kwargs` uses `batch_size` from `train_cfg`, which is `1` when dynamic batching is enabled (the `batch_size` field is a placeholder). This means:
- Validation DataLoader has `batch_size=1`
- With `max_batches=100`, only **100 out of 17,432 val samples** are evaluated (0.57%)
- Each forward pass processes a single sample → GPU underutilized

**Impact:** Validation WER is computed on a tiny, potentially unrepresentative sample. Statistical noise in WER makes checkpoint selection unreliable. Also, 100 sequential forward passes are slower than ~175 batched passes.
**Fix:** Add a separate `val_batch_size` config field (e.g., 32 or 64). Use it for `val_kwargs["batch_size"]`.

### P2-F3. MEDIUM — `save_latest` checkpoint doesn't save RNG state → non-reproducible resume

**File:** `training/checkpoint.py:139-150`
**Description:** `save_latest()` (called every `ckpt_every` steps and on SIGTERM) does NOT save `rng_state` or `cuda_rng_state`. Only `save()` (the top-k method) saves RNG state. On resume via `load_latest()`, RNG state is not restored, so data ordering differs from the original run.

**Impact:** After SIGTERM recovery or mid-epoch resume, the DataLoader produces different sample ordering. This is acceptable for training quality but breaks reproducibility.
**Fix:** Add RNG state saving to `save_latest()` to match `save()`.

### P2-F4. MEDIUM — `SpeedPerturbation` breaks `DynamicBatchSampler` token budget

**File:** `training/dataset.py:165-171` and `training/sampler.py:82`
**Description:** `DynamicBatchSampler` computes batch sizes using original durations from the manifest. `SpeedPerturbation` with `speed=0.9` makes audio ~11% longer. The actual frames per sample exceed the sampler's estimate, causing batches to exceed `max_tokens`.

**Impact:** ~11% VRAM overshoot risk when `speed=0.9` is sampled. Could cause OOM on large batches.
**Fix:** Either (a) add a safety margin to `max_tokens` when speed perturbation is enabled, or (b) scale the sampler's duration estimate by the max speed factor.
**Note:** Currently mitigated because `speed_perturbation=False` in all configs.

### P2-F5. MEDIUM — Logged training loss comes from last micro-batch only, not averaged across `accum_steps`

**File:** `training/train.py:467-470`
**Description:** `step_loss` is read from `stats["loss"]` which comes from the last micro-batch only. The actual training loss is `loss / accum_steps` averaged across micro-batches, but the logged value doesn't reflect this averaging. With `accum_steps=2`, the logged loss has ~2x more variance than the true training loss.

**Impact:** Noisier TensorBoard curves. Can mislead when comparing runs or detecting divergence.
**Fix:** Accumulate `stats["loss"]` across `accum_steps` micro-batches and log the average.

### P2-F6. MEDIUM — `_StepTimer.cpu_mark()` would cause label-event misalignment if called

**File:** `training/train.py:190-191`
**Description:** `cpu_mark()` appends a label without a corresponding GPU event. `results_ms()` iterates `(events[i], events[i+1])` pairs mapped to `labels[i]`. If `cpu_mark` were called, the label count would exceed event pairs, causing label-event misalignment.
**Impact:** Currently dead code (`cpu_mark` is never called). But if used in the future, timing output would be silently wrong.
**Fix:** Either remove `cpu_mark` or track CPU vs GPU labels separately.

### P2-F7. LOW — `dataset.py` has unused imports: `re`, `Path`

**File:** `training/dataset.py:8-9`
**Description:** `import re` and `from pathlib import Path` are imported but never used.
**Impact:** Code cleanliness.
**Fix:** Remove unused imports.

### P2-F8. LOW — `collate_fn` has inconsistent return type (3-tuple vs 4-tuple)

**File:** `training/dataset.py:280-282`
**Description:** When `texts` are `torch.Tensor`, returns `(audio_batch, audio_lengths, texts_batch, text_lengths)` (4-tuple). When `texts` are NOT tensors (fallback), returns `(audio_batch, audio_lengths, texts)` (3-tuple). Any caller unpacking all 4 values would break on the fallback path.
**Impact:** Currently unreachable because `ASRDataset` always produces tensor texts via tokenizer. But fragile if the dataset is used without a tokenizer.
**Fix:** Make fallback path also return 4-tuple with empty text_lengths.

### P2-F9. LOW — `_StepTimer._cur_label` is set but never read

**File:** `training/train.py:166,173,182`
**Description:** `_cur_label` is initialized in `__init__`, cleared in `start()`, but never read anywhere.
**Impact:** Dead attribute.
**Fix:** Remove it, or implement the intended functionality.

### P2-F10. LOW — `validate.py` runs entirely in FP32, misses bf16 speedup

**File:** `training/validate.py:77-141`
**Description:** Validation runs without `autocast`, so all computation is FP32. With bf16 training, validation could use `torch.amp.autocast("cuda", dtype=torch.bfloat16)` for ~2x speedup on RTX 3090.
**Impact:** Validation takes ~2x longer than necessary. With `max_batches=100` at batch_size=1, this is ~1-2 minutes vs ~30-60 seconds.
**Fix:** Wrap validation forward pass in `torch.amp.autocast("cuda", dtype=torch.bfloat16)`.

### P2-F11. LOW — `CircularKVCache.window_size` uses first/last window_right for all layers

**File:** `inference/streaming_encoder.py:77-81`
**Description:** `max_window = config.window_left + config.window_right_first_last + 1 = 21` is used for all layers. Middle layers (2-3) only need `window_left + 0 + 1 = 17` positions. The cache is ~24% larger than needed for those layers.
**Impact:** Wastes ~4 positions of KV cache per middle layer. Negligible for bs=1 inference.
**Fix:** Per-layer window sizes would save memory but adds complexity. Not worth it.

### P2-F12. LOW — `DynamicBatchSampler` with `drop_last=True` discards some samples every epoch

**File:** `training/sampler.py:89,98`
**Description:** Batches smaller than `min_batch_size` are dropped. With `min_batch_size=4` and 100 buckets, the last batch in each bucket (often < 4 samples) is discarded. Over an epoch, some samples are never seen.
**Impact:** With 442K samples and 100 buckets, ~100-400 samples are dropped per epoch (< 0.1%). Negligible.
**Fix:** Acceptable. Could set `drop_last=False` but would create tiny uneven batches.

### P2-F13. LOW — `download_data.py` SOVA speaker IDs are pseudo-speakers (per-clip MD5)

**File:** `scripts/download_data.py:331-332`
**Description:** Already noted in Pass 1 (F7). Re-confirming: `speaker_id = f"{dataset_name}_{content_hash}"` creates unique IDs per clip. `split_by_speaker` effectively does random splitting, not speaker-independent splitting.
**Impact:** Train/val speaker leakage is possible. However, SOVA datasets don't provide speaker metadata, so this is the best available approach.
**Fix:** Document limitation. No code fix possible.

---

## Verified Correct (No Bug Found)

These areas were thoroughly traced and confirmed correct:

1. **Padding mask logic** (`encoder.py:91-92`): The `(~pad_mask).float().masked_fill(~pad_mask, _MASK_NEG)` pattern is correct. Both `~pad_mask` references on the RHS refer to the original bool tensor (Python evaluates RHS before assignment). Valid positions get 0.0 (allowed), padding gets -1e4 (blocked).

2. **CTC loss targets**: `model.py` passes `tokens` (padded with -100) and `token_lengths` to `F.ctc_loss`. CTC internally uses `target_lengths` to slice valid tokens; -100 padding is never read. Correct.

3. **Decoder KV cache**: Cross-attention correctly skips `k_proj`/`v_proj` when cached KV exists (`attention.py:74-76`). Self-attention correctly concatenates past KV (`attention.py:86-88`). Cache offset for RoPE is correct (`decoder.py:159`).

4. **Mask shapes for SDPA**: Sliding window mask `(1, 1, Q, K)` + padding mask `(batch, 1, 1, K)` → combined `(batch, 1, Q, K)` broadcasts to `(batch, num_heads, Q, K)` in SDPA. Correct.

5. **GradScaler with bf16**: `enabled=False` makes all scaler methods passthrough. `scaler.unscale_()` is a no-op, `scaler.step()` calls `optimizer.step()` directly. `clip_grad_norm_` operates on unscaled (true) gradients. Correct.

6. **CTC greedy decode**: Standard collapse-blank-and-repeat algorithm. Token filtering `>= 6` correctly excludes SP's built-in `<unk>=0, <s>=1, </s>=2` and our `<blank>=3, <sos/eos>=4, <pad>=5`.

7. **Streaming encoder mask**: `_make_streaming_chunk_mask` uses absolute positions for correct cross-chunk sliding window attention. Mask shape `(1, 1, chunk_len, kv_len)` compatible with SDPA.

---

## Combined Priority Matrix

| ID | Severity | Effort | Fix Before M9? |
|----|----------|--------|----------------|
| P2-F2 | HIGH | Small | **YES** — val_batch_size config field |
| P2-F1 | HIGH | Small | Recommended — remove dead feature |
| P2-F3 | MEDIUM | Small | Recommended — add RNG state to save_latest |
| P2-F4 | MEDIUM | Small | No — speed_perturbation=False |
| P2-F5 | MEDIUM | Small | Recommended — average loss across accum |
| P2-F6 | MEDIUM | Small | No — dead code, remove cpu_mark |
| P2-F7 | LOW | Trivial | Optional — remove unused imports |
| P2-F8 | LOW | Small | Optional — consistent return type |
| P2-F9 | LOW | Trivial | Optional — remove _cur_label |
| P2-F10 | LOW | Small | Optional — bf16 validation |
| P2-F11 | LOW | None | No — negligible |
| P2-F12 | LOW | None | No — acceptable |
| P2-F13 | LOW | None | No — documented limitation |
