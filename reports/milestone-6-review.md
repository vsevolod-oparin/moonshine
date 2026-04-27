# M6: Overfit Sanity Check — Code Review

## Scope

Files changed during M6 (since M5 commit `05bc685`):

**Modified:**
- `models/masks.py` — replaced `float("-inf")` with `_MASK_NEG = -1e4`
- `models/encoder.py` — padding mask uses `-1e4`
- `models/encoder_v21.py` — padding mask uses `-1e4`
- `models/decoder.py` — padding + cross-attn masks use `-1e4`
- `models/model.py` — encoder forced to FP32 under AMP
- `training/train.py` — GPU temp/power monitoring

**New:**
- `configs/overfit_v2_tiny_10.yaml`, `overfit_v2_tiny_100.yaml`
- `configs/overfit_v21_tiny_10.yaml`, `overfit_v21_tiny_100.yaml`
- `scripts/overfit_check.py` — overfit test + memory profiling
- `reports/milestone-6-overfit-check.md`

---

## Findings

### 1. Duplicated GPU monitoring code
**Severity:** LOW
**Files:** `training/train.py`, `scripts/overfit_check.py`

GPU temp monitoring (`_gpu_stats`, `_log_gpu_temp` / `gpu_stats`, `log`) is copy-pasted between `training/train.py` and `scripts/overfit_check.py`. Minor naming inconsistencies: `train.py` uses `_gpu_stats()` (private), `overfit_check.py` uses `gpu_stats()` (public). Same `nvidia-smi` subprocess call duplicated.

**Recommendation:** Extract to a shared utility like `training/gpu_monitor.py`. Low priority — both scripts work correctly, and `overfit_check.py` is a test script not production code.

### 2. `_log_gpu_temp` uses module-level mutable global `_last_gpu_log`
**Severity:** LOW
**File:** `training/train.py:22`

`_last_gpu_log = 0.0` is module-level mutable state. This works fine for single-process training but would be fragile if the module were imported in tests or multi-process contexts. The `global _last_gpu_log` declaration in `_log_gpu_temp` is correct Python, just unusual.

**Recommendation:** Acceptable for now. If we add distributed training (M9), wrap this in a class.

### 3. Encoder FP32 forcing may reduce decoder effectiveness
**Severity:** LOW
**File:** `models/model.py:65`

`torch.amp.autocast("cuda", enabled=False)` forces the entire encoder to FP32. This is correct — it fixes NaN — but it means:
- The encoder's ~22M params don't benefit from FP16 forward/backward speedups
- Memory savings from AMP apply only to decoder (~10M params) + loss computation
- The preprocessor already had its own FP32 forcing (line 41 in preprocessor.py), so there's a double-disable in the preprocessor path — harmless but redundant

**Recommendation:** Acceptable trade-off. FP32 encoder ensures stable training. The Tiny model fits easily in 24GB regardless. For Small/Medium models, this will increase peak VRAM but still fit. Can revisit with Flash Attention / BF16 on H100.

### 4. `overfit_check.py` LR schedule may produce negative LR
**Severity:** LOW
**File:** `scripts/overfit_check.py:70-73`

```python
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: min(1.0, step / warmup) * (1.0 - step / max_steps),
)
```

At `step = max_steps`, `lr_lambda = 0.0` (correct). But `LambdaLR` calls `lr_lambda` with the *epoch* number, and the scheduler is stepped once per outer loop iteration (line 183). If `max_steps` is reached exactly, the last LR is 0. No issue in practice.

However, if someone reused this with more steps than `max_steps`, the LR would go negative: `(1 - step/max_steps)` becomes negative. Should clamp to 0.

**Recommendation:** Add `max(0.0, ...)` clamp:
```python
lr_lambda=lambda step: max(0.0, min(1.0, step / warmup) * (1.0 - step / max_steps))
```

### 5. Inconsistent `-1e4` vs `_MASK_NEG` usage
**Severity:** LOW
**Files:** `models/encoder.py:92`, `models/encoder_v21.py:175`, `models/decoder.py:125,130`

The central constant `_MASK_NEG = -1e4` is defined in `masks.py` but the encoder/encoder_v21/decoder files use the literal `-1e4` instead of importing and using `_MASK_NEG`. If we ever need to tune this value (e.g., for BF16 where `-1e4` might not be sufficient), we'd have to update 5+ locations across 4 files.

**Recommendation:** Import `_MASK_NEG` from `masks.py` and use it everywhere. Low urgency since this value is unlikely to change, but it's a single-source-of-truth issue.

### 6. `overfit_check.py` loads config but ignores training section
**Severity:** LOW
**File:** `scripts/overfit_check.py:276-287`

`load_config_for_overfit()` loads the full YAML config and returns `(model_cfg, full_cfg)`, but the overfit test hardcodes its own optimizer (`AdamW, lr=1e-3`), batch size, and steps — ignoring `full_cfg["training"]`. The YAML configs define `training.max_steps`, `training.batch_size`, etc. that are never read by the script.

This means the overfit configs' training sections (`max_steps: 500`, `batch_size: 10`) are documentation only — the actual values come from `main()`'s hardcoded arguments (`max_steps=800`, `batch_size=10`).

**Recommendation:** Either read training params from the config or remove the training section from overfit configs to avoid confusion. The current state is misleading.

### 7. `overfit_check.py` NaN gradient handling logs loss but continues
**Severity:** LOW
**File:** `scripts/overfit_check.py:133-138`

When NaN gradient is detected, the code increments `epoch_loss` with the (presumably NaN/inf) stats loss and counts it as a batch. This means `avg_loss` displayed could show NaN. The loss value is cosmetic only (not used for optimization), but it makes the output confusing.

**Recommendation:** Skip the loss accumulation for NaN batches:
```python
if torch.isnan(grad_norm) or torch.isinf(grad_norm):
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    continue  # don't count NaN batches
```

### 8. `overfit_check.py` unused import `load_manifest` result
**Severity:** TRIVIAL
**File:** `scripts/overfit_check.py:99`

`records = load_manifest(manifest_path)` is used only for `len(records)` to print the count. The actual dataset loads the manifest independently via `ASRDataset`. Double-loading is wasteful but negligible for 10-100 entries.

### 9. `encode()` hardcodes `"cuda"` in autocast
**Severity:** LOW
**File:** `models/model.py:65`

`torch.amp.autocast("cuda", enabled=False)` hardcodes `"cuda"`. If the model is ever run on CPU (e.g., for testing), this will warn or silently not apply. The preprocessor uses the same pattern, so this is consistent with existing code.

**Recommendation:** Could use `device_type=audio.device.type` for robustness, but this is the existing convention across the codebase.

---

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0 |
| HIGH | 0 |
| MEDIUM | 0 |
| LOW | 8 |
| TRIVIAL | 1 |

**No blocking issues.** All M6 changes are correct and the overfit tests pass. The findings are all low-severity cleanups: code duplication, hardcoded literals, and cosmetic output issues.

## Recommended Actions (Priority Order)

1. **Clamp LR to 0** in `overfit_check.py` lambda (finding #4) — one-line safety fix
2. **Import `_MASK_NEG`** in encoder/decoder files (finding #5) — single source of truth
3. **Skip NaN batches in loss accumulation** (finding #7) — cleaner output
4. **Sync overfit configs with script** (finding #6) — remove confusing dead config sections
5. Others can be deferred to M9+ (distributed training, shared utilities)
