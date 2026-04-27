# M6: Overfit Sanity Check — Report

## Summary

**All tests pass.** Both v2 and v2.1 Tiny architectures can memorize training data, confirming correct gradient flow, loss computation, tokenizer, and data pipeline.

## Bugs Fixed

### B1: SDPA NaN under AMP with `-inf` attention masks

**Root cause:** `F.scaled_dot_product_attention` with FP16 Q/K/V and FP32 `attn_mask` containing `float("-inf")` produces NaN in the backward pass (cuDNN SDPA bug with stride mismatches under mixed precision).

**Fix:** Replace all `float("-inf")` values in attention masks with `-1e4` (large finite negative). Applied to:
- `models/masks.py` — `make_sliding_window_mask()`, `make_causal_mask()`
- `models/encoder.py` — padding mask
- `models/encoder_v21.py` — padding mask
- `models/decoder.py` — padding mask + cross-attention mask

### B2: v2.1 encoder NaN backward under AMP

**Root cause:** The v2.1 U-Net encoder (downsample/upsample + depthwise conv) produces NaN gradients under FP16 AMP due to the same cuDNN SDPA backward issue compounded by the multi-scale architecture.

**Fix:** Force the encoder to run in FP32 under AMP (`models/model.py:encode()` wraps encoder in `torch.amp.autocast("cuda", enabled=False)`). The decoder still benefits from AMP.

### B3: Training divergence during overfit

**Root cause:** CosineAnnealingLR schedule wraps LR back to max after reaching minimum, causing catastrophic divergence when the model has already converged to near-zero loss.

**Fix:** Use linear warmup + linear decay to zero (`LambdaLR`) in the overfit script. For production training, Schedule-Free handles LR internally.

## Test Results

### T5: Overfit 10 samples (WER → 0%)

| Model | Steps to WER=0% | Final Loss | Status |
|-------|-----------------|------------|--------|
| v2 Tiny | ~300 | 0.0035 | **PASS** |
| v2.1 Tiny | ~300 | 0.0003 | **PASS** |

### T6: Overfit 100 samples (WER < 5%)

| Model | Steps to WER<5% | Final WER | Final Loss | Status |
|-------|-----------------|-----------|------------|--------|
| v2 Tiny | ~150 | 0.00% | 0.0000 | **PASS** |
| v2.1 Tiny | ~150 | 0.00% | 0.0001 | **PASS** |

### T17: GPU Memory Profiling (batch=32)

| Model | Parameters | Peak VRAM | Total GPU | Headroom | Status |
|-------|-----------|-----------|-----------|----------|--------|
| v2 Tiny | 22.8M (86.9 MB) | 1,967 MB | 24,111 MB | 22,145 MB | **OK** |
| v2.1 Tiny | 23.1M (88.2 MB) | 2,164 MB | 24,111 MB | 21,948 MB | **OK** |

Peak VRAM at batch=32 is ~2 GB. With 24 GB GPU, max batch size for Tiny is estimated at ~200+ clips. For Small (10 layers, 620 dim), expect ~6-8x more VRAM → batch=32 should still fit.

## GPU Temperature

With improved cooling, GPU temps stayed at **57-62°C** during sustained training at ~330W. Well below the 85°C warning threshold.

## New Features

- **Temperature monitoring** in `scripts/overfit_check.py`: logs GPU temp/power every 30s, warns at 85°C, critical at 90°C
- **Temperature monitoring** in `training/train.py`: logs GPU temp/power every `log_every` steps to W&B/TensorBoard as `sys/gpu_temp` and `sys/gpu_power`

## Files Modified

- `models/masks.py` — replaced `float("-inf")` with `-1e4`
- `models/encoder.py` — replaced `float("-inf")` with `-1e4`
- `models/encoder_v21.py` — replaced `float("-inf")` with `-1e4`
- `models/decoder.py` — replaced `float("-inf")` with `-1e4`
- `models/model.py` — encoder forced to FP32 for v2.1 (and all versions) under AMP
- `training/train.py` — added GPU temperature/power monitoring
- `scripts/overfit_check.py` — created overfit test script with temp monitoring
- `configs/overfit_v2_tiny_10.yaml`, `overfit_v2_tiny_100.yaml` — overfit configs
- `configs/overfit_v21_tiny_10.yaml`, `overfit_v21_tiny_100.yaml` — overfit configs
- `data/manifests/overfit_10.jsonl` — 10-clip subset manifest
- `data/manifests/overfit_100.jsonl` — 100-clip subset manifest

## Gate

**T5 passes = model can learn. T6 passes = tokenizer works. T17 confirms memory is sufficient.** All gates passed. Ready for M6.5.
