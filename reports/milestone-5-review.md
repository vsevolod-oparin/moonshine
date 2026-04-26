# M5 Code Review: Training Loop

**Date**: 2026-04-26
**Scope**: `training/{train.py, validate.py, checkpoint.py, logger.py, sampler.py}` + changes to `models/preprocessor.py`, `training/dataset.py`
**Reviewer**: Self-review

## Dimensions

| Dimension | Issues | Critical | High | Medium | Low |
|-----------|--------|----------|------|--------|-----|
| Bugs | 3 | 2 | 1 | 0 | 0 |
| Performance | 2 | 0 | 1 | 1 | 0 |
| Interface/Integration | 2 | 0 | 1 | 1 | 0 |
| Code Quality | 4 | 0 | 0 | 2 | 2 |
| Config Correctness | 1 | 0 | 1 | 0 | 0 |
| **Total** | **12** | **2** | **4** | **4** | **2** |

---

## Bugs

### B1 [CRITICAL] — `validate()` crashes: model in eval mode returns logits, not loss

**File**: `training/validate.py:105-108`

`model.eval()` is called at line 90, then `model(audio, tokens, ...)` at line 105. In eval mode, `RuMoonshine.forward()` returns `(aed_logits, stats, weight)` — the first element is a `[B, T, V]` logits tensor, not a scalar loss. Then line 108 calls `loss.item()` on this tensor, which crashes with `RuntimeError: a Tensor with 5120 elements cannot be converted to Scalar`.

**Impact**: Every validation step crashes. WER never computed. Training loop fails at first validation checkpoint.

**Fix**: Compute loss manually in eval mode, or call `model.train()` before the loss computation and `model.eval()` after, or restructure to compute loss from the returned logits.

### B2 [CRITICAL] — Preprocessor zeros ALL `out_lengths` when any clip is short

**File**: `models/preprocessor.py:33-36`

```python
if audio.size(-1) < min_len:
    audio = F.pad(audio, (0, min_len - audio.size(-1)))
    if audio_lengths is not None:
        audio_lengths = torch.clamp(audio_lengths, max=0)
```

The padding check is on the batch dimension (`audio.size(-1)` is the max across the batch after collation), not per-item. But `torch.clamp(audio_lengths, max=0)` zeros ALL lengths in the batch, even for clips that were long enough. If ONE clip in a batch is shorter than 895 samples, ALL clips get `out_length=0`, meaning ALL attention masks become fully masked, producing zero/NaN encoder output.

**Impact**: Mixed-length batches with at least one short clip produce garbage. Training data has `min_duration=1.0` filter, so 1s audio = 16000 samples > 895. This may be rare in practice but would produce silent failures when it occurs.

**Fix**: Clamp per-item — only zero out lengths for clips that were actually short:
```python
if audio_lengths is not None:
    short_mask = audio_lengths < min_len
    audio_lengths = torch.where(short_mask, 0, audio_lengths)
```
Also the padding check should be per-item, not batch-level.

### B3 [HIGH] — CTC hypothesis tokens not filtered for special tokens before decoding

**File**: `training/validate.py:129`

`tokenizer.decode(ids)` where `ids` comes from CTC greedy decode. CTC can output any token ID including special tokens (0-5: `<unk>`, `<s>`, `</s>`, `<blank>`, `<sos/eos>`, `<pad>`). Decoding `<unk>` (id=0) produces the literal string `<unk>` in the hypothesis, inflating WER.

**Impact**: Hypothesis contains `<unk>` literal strings. WER is slightly inflated.

**Fix**: Filter `ids` for `t >= 6` before decoding, same as reference filtering at line 132.

---

## Performance

### P1 [HIGH] — `validate()` double-encodes: calls `model()` then `model.encode()` separately

**File**: `training/validate.py:105-112`

Lines 105-107 call `model(audio, tokens, ...)` which runs encode → CTC head → decode → loss. Then lines 111-112 call `model.encode(audio, audio_lengths)` again followed by `model.ctc_head(enc_output)`. The encoder is run twice per validation batch.

**Impact**: ~2x validation time. Encoder is ~60% of compute, so validation is ~1.6x slower than necessary. With `max_batches=50` at every 2K steps, this adds ~20 minutes over a 50K step training run.

**Fix**: Either (a) extract `enc_output` from the first forward pass (requires restructuring model forward), or (b) only compute CTC-based WER without running the full model for loss, or (c) compute val_loss from CTC logits directly.

### P2 [MEDIUM] — `torch.cuda.amp.autocast` and `GradScaler` deprecated

**File**: `training/train.py:14,244`

PyTorch 2.5 emits `FutureWarning` for `torch.cuda.amp.GradScaler` and `torch.cuda.amp.autocast`. Should use `torch.amp.autocast('cuda')` and `torch.amp.GradScaler('cuda')`.

**Impact**: Deprecation warnings in logs. No functional impact yet, will break in future PyTorch.

**Fix**: Replace with `torch.amp.autocast(device_type='cuda')` and `torch.amp.GradScaler('cuda')`.

---

## Interface/Integration

### I1 [HIGH] — `validate()` does not restore `Schedule-Free` optimizer to train mode

**File**: `training/validate.py:139` + `training/train.py:341-343`

`validate()` calls `model.eval()` at line 90 and `model.train()` at line 139. Back in `train.py`, lines 341-343 also call `model.train()` and `optimizer.train()`. But `validate()` doesn't know about the optimizer, so between line 90 (model.eval) and line 139 (model.train) in validate.py, the Schedule-Free optimizer remains in whatever state it was in.

This is actually fine — Schedule-Free's `train()`/`eval()` only affects the optimizer step (weight averaging), not the forward pass. No optimizer step happens during validation. However, the pattern is fragile — if someone adds an optimizer step inside validate(), it would be silently wrong.

**Impact**: No current functional impact. Fragile pattern for future changes.

### I2 [MEDIUM] — `collate_fn` ignores `tokenizer` parameter

**File**: `training/dataset.py:237`

`collate_fn` accepts a `tokenizer` parameter but never uses it. In `create_dataloader`, it's passed as `lambda b: collate_fn(b, tokenizer=tokenizer_model)` — a string path, not a tokenizer object. The parameter is dead code.

**Impact**: Dead code, minor confusion.

---

## Code Quality

### Q1 [MEDIUM] — 7 unused imports in `train.py`

**File**: `training/train.py:2,7-8,13,16`

Unused: `json`, `sys`, `time`, `torch.distributed as dist`, `models.config.load_config`. Only `np` (numpy) is used once in `setup_seed`.

### Q2 [MEDIUM] — `sample_dynamic_window` mutates model config in-place

**File**: `training/train.py:84-91`

After dynamic window training, `model.config` retains the last random window values, not the original config. This means if you save the model after training, the config in the checkpoint has random window values. Any post-training inference or export uses these random values.

**Fix**: Save original values and restore after forward pass, or pass window config as a separate argument.

### Q3 [LOW] — `validate.py` imports `defaultdict` but never uses it

**File**: `training/validate.py:3`

### Q4 [LOW] — `validate.py` imports `F` (torch.nn.functional) but never uses it

**File**: `training/validate.py:2`

---

## Config Correctness

### C1 [HIGH] — `training.optimizer.warmup_steps` in YAML conflicts with Schedule-Free

**File**: `configs/v2_tiny.yaml:37`, `training/train.py:36`

YAML has `warmup_steps: 2000` under `training:`. But `setup_optimizer()` passes `warmup_steps=0` to `AdamWScheduleFree`. The YAML's `warmup_steps` is only read by `setup_scheduler()` which is skipped for Schedule-Free. This is technically correct (Schedule-Free handles warmup internally) but the YAML key `warmup_steps: 2000` is misleading — it looks like it should affect Schedule-Free but doesn't.

**Impact**: User confusion. If someone switches optimizer, warmup_steps suddenly becomes active with different semantics.

**Fix**: Add comment in YAML, or move warmup_steps under an `adamw:` subsection.

---

## Summary

### Fix Now (before M6)

| ID | What | Effort | Status |
|----|------|--------|--------|
| B1 | validate() crashes on eval-mode model returning logits | Small | **Fixed** — validate() encodes+CTC directly, no model() call |
| B2 | Preprocessor zeros ALL out_lengths for mixed-length batch with short clips | Small | **Fixed** — per-item `torch.where` instead of `torch.clamp(max=0)` |
| B3 | Filter special tokens from CTC hypothesis before decoding | Small | **Fixed** — `hyp_ids = [t for t in ids if t >= 6]` |
| P1 | validate() double-encodes — extract enc_output from first forward | Small | **Fixed** — single encode pass, CTC logits reused for loss+decode |

### Fix Soon (before production training)

| ID | What | Effort | Status |
|----|------|--------|--------|
| P2 | Deprecated torch.cuda.amp API | Small | **Fixed** — `torch.amp.{autocast,GradScaler}` |
| Q1 | Remove 7 unused imports in train.py | Small | **Fixed** |
| Q2 | Dynamic window mutates model config persistently | Small | **Fixed** — save/restore original values |
| Q3-Q4 | Remove unused imports in validate.py | Small | **Fixed** — removed `F`, `defaultdict` |
| C1 | YAML warmup_steps misleading for Schedule-Free | Small | **Fixed** — added clarifying comment |
| I2 | Dead `tokenizer` parameter in collate_fn | Small | **Fixed** — removed dead parameter |

### Accept

| ID | What | Reason |
|----|------|--------|
| I1 | validate() doesn't restore SF optimizer.train() | No current impact, SF eval only affects step, not forward |
