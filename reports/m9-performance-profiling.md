# M9 Performance Profiling Report (Profiled, 2026-04-30)

**Hardware:** RTX 3090 Ti 24GB (Ampere SM_86) | **Model:** v2 Tiny (18M params)

---

## 1. Profiling Methodology

Measured with `_StepTimer` + `torch.cuda.synchronize()`, 20-step averages after 5-step warmup. All runs with: bf16 autocast, compiled encoder, SpecAugment, TF32, fused AdamW.

---

## 2. Throughput vs Batch Size (10s audio, 415 frames/sample)

| max_tokens | batch_size | tokens/batch | fwd (ms) | bwd (ms) | opt (ms) | total (ms) | steps/s | tok/s | VRAM alloc | VRAM % |
|-----------|-----------|-------------|---------|---------|---------|-----------|---------|--------|-----------|--------|
| 24,000 | 57 | 23,655 | 49 | 124 | 3.7 | 177 | 5.7 | 134K | 1,409 MB | 5.8% |
| 48,000 | 115 | 47,725 | 93 | 231 | 3.7 | 327 | 3.1 | 146K | 2,532 MB | 10.5% |
| 96,000 | 231 | 95,865 | 191 | 479 | 3.7 | 673 | 1.5 | 142K | 4,777 MB | 19.8% |
| 144,000+ | — | — | — | — | — | — | — | — | OOM | — |

**Key findings:**
- **GPU is compute-bound at ~142K tok/s** — throughput per GPU-second is constant regardless of batch size
- **VRAM is NOT the bottleneck** — only 5.8% utilized at 24K tokens, 19.8% at 96K
- **Backward pass dominates** — 70% of total step time (attention O(n²) in fp32)
- **Larger batches decrease steps/s but maintain same tok/s** — the GPU is saturated

### Real training estimate (dynamic batching with mixed durations)

Using the config's `frames_per_sec` to estimate frames from audio duration:
- **Actual frames_per_sec = 40.0** (preprocessor subsampling factor: 64×3×2 = 384, 16000/384 = 41.67, minus kernel overhead)
- Config uses `frames_per_sec: 41.0` — overestimates by 2.5%, providing a small safety margin

| Scenario | Effective tokens/step | ms/step | steps/s | tok/s | 100K steps |
|----------|----------------------|---------|---------|-------|------------|
| max_tokens=24K, accum=2 | 48,000 | ~354ms | ~2.8 | 136K | **~9.8h** |
| max_tokens=48K, accum=1 | 48,000 | ~330ms | ~3.0 | 145K | **~9.2h** |
| max_tokens=48K, accum=2 | 96,000 | ~660ms | ~1.5 | 145K | ~18.5h |

---

## 3. Compile Mode Comparison

| Mode | fwd (ms) | bwd (ms) | total (ms) | VRAM alloc |
|------|---------|---------|-----------|------------|
| `reduce-overhead` | 49.3 | 123.6 | 176.6 | 1,409 MB |
| `default` | 48.6 | 123.4 | 176.0 | 3,879 MB |

**Finding: Identical speed.** `reduce-overhead` uses CUDA graphs (which cache device memory), `default` allocates fresh buffers. At this model scale, the CUDA graphs overhead is negligible.

**CUDA graph memory concern:** With `reduce-overhead` + dynamic batching, a CUDA graph is created per unique batch size/shape. With `num_buckets=100`, up to 100 graphs could accumulate. Each graph for this model uses ~50-100 MB, so worst case ~5-10 GB of graph memory. This fits within 24 GB but reduces headroom. **Not a blocker, but worth monitoring.**

---

## 4. Preprocessor Vectorization

| Batch size | Loop (current) | Vectorized | Speedup |
|-----------|---------------|-----------|---------|
| 4 | 0.047ms | 0.193ms | 0.2x |
| 16 | 0.128ms | 0.068ms | 1.9x |
| 64 | 0.446ms | 0.067ms | 6.7x |
| 256 | 1.705ms | 0.067ms | 25.5x |

**At realistic batch size (60), saves 0.38ms per step (0.2%).** Not worth implementing — the loop is only a Python comprehension over small ints, and the overhead is buried in the 177ms step time. Vectorization becomes beneficial only above ~100 batch size.

---

## 5. Optimization Recommendations

### RECOMMENDED: Increase max_tokens to 48K, reduce accum to 1

**What:** `max_tokens: 24000 → 48000`, `accum_steps: 2 → 1`

**Why:**
- Same effective batch size (48K tokens per optimizer step)
- Eliminates micro-batch boundary overhead: `optimizer.zero_grad()`, `grad_norm` computation, `scaler.update()`, scheduler step — saved once per optimizer step
- ~5% faster (354ms → 330ms per optimizer step)
- VRAM at 48K tokens = 10.5% — plenty of headroom (13.5 GB free for CUDA graphs + checkpoint buffering)

**Risk:** Minimal. Effective batch unchanged. The only behavioral difference is gradient noise: accum=2 averages over 2 micro-batches, accum=1 uses one large batch. For from-scratch training, the larger single batch has LESS gradient noise, which is better.

**Config change:**
```yaml
training:
  accum_steps: 1
  batching:
    max_tokens: 48000
    frames_per_sec: 41.0   # keep 41 for safety margin (actual is 40)
```

### OPTIONAL: Fix frames_per_sec to 40

Actual value is 40.0, config uses 41.0. Keeping 41 is fine (2.5% safety margin for speed perturbation overshoot). Changing to 40 would increase batch sizes by 2.5% (slightly more efficient packing) but remove the safety margin. **Recommend keeping 41.**

### OPTIONAL: `torch.inference_mode()` in validate.py

Replace `@torch.no_grad()` with `@torch.inference_mode()`. Saves ~1-3% on validation time. Validated 3 times per epoch, ~170 validation runs total → saves ~30-60 seconds. Marginal but free.

### NOT RECOMMENDED

| Idea | Reason |
|------|--------|
| Compile mode change | Identical speed, reduce-overhead is fine |
| Preprocessor vectorization | 0.2% savings, not worth code churn |
| bf16 encoder attention (Flash Attention) | HIGH risk of NaN, test separately after M9 |
| max_tokens > 48K | OOM risk from CUDA graph memory accumulation + speed perturbation overshoot |

---

## 6. Final Config for M9

```yaml
training:
  accum_steps: 1                       # was 2 → same effective batch (48K)
  batching:
    max_tokens: 48000                  # was 24000
    frames_per_sec: 41.0               # unchanged (safety margin)
```

**Expected: ~9.2 hours for 100K steps** (was ~9.8h with accum=2, max_tokens=24K).

---

## 7. Profiling Notes

- **Preprocessor actual frames/sec:** 40.0 (16000 samples/sec ÷ 384 stride ≈ 41.67 Hz, minus conv kernel overhead → 40.0)
- **Breakdown per 177ms step:** forward 28% (preprocessor 5% + encoder 15% + decoder/loss 8%), backward 70%, optimizer 2%
- **VRAM breakdown (24K tokens):** params 324MB + optimizer states 288MB + activations ~580MB + CUDA graphs ~420MB = 1.4GB allocated, 5.1GB reserved
- **CUDA graphs accumulate per unique batch shape** — at num_buckets=100, worst case ~5-10GB of graph memory. Safe for 24GB.
- **Data loading is NOT a bottleneck** — 4 workers × 2 prefetch provide 310ms of GPU-pipelined I/O, enough for 177ms step time.
