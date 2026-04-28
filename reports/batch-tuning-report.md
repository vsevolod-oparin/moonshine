# Batch Size & Dynamic Batching Tuning Report

**Date**: 2026-04-28
**Model**: ru-Moonshine v2 Tiny (22M params, enc_dim=320, 6 layers)
**GPU**: NVIDIA RTX 3090 Ti (24GB)
**Data**: 156K clips (237h train), CV21 ru + RuLS

## Executive Summary

Dynamic token-based batching (`max_tokens` per batch) significantly improves model quality over fixed batch size training, with a modest speed penalty after cuDNN warmup. The best configuration (V5, `max_tokens=30000`) achieved **76.97% WER** — a **6.5% absolute improvement** over fixed BS=96 — while using only 59% VRAM.

## Runs Compared

| | V2 | V3 | V4 | V5 |
|---|---|---|---|---|
| **Batching** | Fixed BS=96 | Fixed BS=96 | Dynamic, max_tokens=20000 | Dynamic, max_tokens=30000 |
| **Optimizations** | None | Fused AdamW, deferred .item(), cudnn.benchmark, pre-tokenized manifests | Same as V3 | Same as V3 |
| **DataLoader workers** | 4 (persistent) | 4 (persistent) | 0 (workers crash with variable batches) | 0 |
| **Epochs in 5000 steps** | 3.1 | 3.1 | 2.8 | 4.1 |
| **Batches/epoch** | 1624 | 1624 | 1799 | 1214 |

## Results

### Quality

| Metric | V2 | V3 | V4 | V5 |
|---|---|---|---|---|
| **Val WER** | 83.45% | 83.66% | **78.98%** | **76.97%** |
| Val SER | 99.69% | 99.56% | 100.00% | 100.00% |
| Final loss | 0.5718 | 0.5879 | 0.7060 | 0.6897 |
| Final accuracy | 93.4% | 93.8% | 91.0% | 92.8% |
| Val loss | 1.7192 | 1.7283 | 1.5541 | 1.4447 |

### Performance

| Metric | V2 | V3 | V4 | V5 |
|---|---|---|---|---|
| **Total time** | 29.6 min | 32.5 min | 54.6 min | 64.7 min |
| ms/step (overall) | 355 | 390 | 656 | 776 |
| ms/step (epoch 3+) | 376 | — | 389 | 495 |
| **Peak VRAM** | 84% (20.6GB) | 84% (20.6GB) | 39% (9.6GB) | 59% (14.5GB) |
| Avg batch size | 96 | 96 | 87 | 129 |
| Batch size range | 96 | 96 | 23–288 | 42–253 |
| `fwd` time (stable) | 28–116ms | 28–116ms | 47–58ms | 69–90ms |

### Per-Epoch Timing

| Epoch | V2 | V3 | V4 | V5 |
|---|---|---|---|---|
| 1 | 4.9 min | 5.4 min | 31.7 min | 27.8 min |
| 2 | 7.6 min | 8.5 min | 13.9 min | 14.4 min |
| 3 | 10.2 min | 10.3 min | 9.1 min | 11.4 min |
| 4 | 7.0 min | 8.2 min | — | 10.0 min |
| 5 | — | — | — | 1.1 min |

## Key Findings

### 1. V3 optimizations had zero quality impact

V3 added fused AdamW, deferred `loss.item()`, `cudnn.benchmark`, pre-tokenized manifests, and persistent workers. Result: **identical loss curves** to V2 (same seed produces same loss at every step), 9.7% slower. The Tiny model on a 3090 is compute-bound — optimizer/data overhead is negligible.

### 2. Dynamic batching dramatically improves WER

Switch from fixed BS=96 to token-budget batching improved WER by 4.5–6.5% absolute. Causes:

- **Uniform token budgets**: Each batch gets consistent encoder computation regardless of clip duration. Fixed BS=96 wastes compute on short clips and starves on long clips.
- **Implicit regularization**: Variable batch sizes introduce gradient noise, acting as regularizer.
- **Better gradient signal**: Large batches of short clips (bs=200+) give low-variance gradients; small batches of long clips (bs=40) focus on harder utterances.

### 3. V5 (30K tokens) > V4 (20K tokens)

Increasing max_tokens from 20K to 30K: WER 78.98% → **76.97%** (-2.0%), VRAM 39% → 59%, late-training speed +27%.

### 4. Epoch 1 cuDNN warmup is the speed bottleneck

| Phase | V2 | V4 | V5 |
|---|---|---|---|
| Epoch 1 | 4.9 min | 31.7 min | 27.8 min |
| Epoch 3+ (ms/step) | 376 | 389 | 495 |

V4's late-training speed is essentially identical to V2 (1.0x). V5 is only 1.3x slower. Bulk of time penalty is one-time cuDNN warmup.

### 5. DataLoader workers crash with variable batches

`batch_sampler` + `num_workers > 0` causes worker crashes. Each batch has 50–200+ indices; with `prefetch_factor=4` and 4 workers, that's 800+ audio files pinned in memory simultaneously. Forces `num_workers=0`.

## Open Issues

1. **DataLoader worker compatibility** — need custom collation or pre-batched manifests
2. **Optimal `max_tokens`** — 40K–50K unexplored; 59% VRAM leaves headroom
3. **Validation SER** — 100% SER expected for CTC greedy on Tiny; AED beam search would help

## Recommendations

**Phase 1 Tiny**: Use V5 config (`max_tokens=30000`, `num_workers=0`). Accept ~2x time for 6.5% WER improvement.

**Phase 2 Small**: **Must** use dynamic batching (will OOM with fixed BS=96). Fix DataLoader workers first (high priority). Start with `max_tokens=30000`.

```yaml
training:
  batch_size: 1
  batching:
    max_tokens: 30000
    frames_per_sec: 41.0
    max_batch_size: 512
    min_batch_size: 4
  num_workers: 0  # TODO: fix worker crash
```

## Appendix: Loss Curves

### V2 (fixed BS=96)
```
step  loss     aed     ctc     acc
 100  6.4136   4.8316  5.2735  0.071
 500  4.3758   3.0037  4.5737  0.296
1000  2.1988   1.3437  2.8506  0.629
2000  1.3640   0.7350  2.1302  0.778
3000  1.0517   0.5459  1.6903  0.838
4000  0.8112   0.3277  1.4515  0.894
5000  0.5718   0.1988  1.2432  0.934  val_WER=83.45%
```

### V5 (dynamic 30K tokens)
```
step  loss     aed     ctc     acc    bs
 100  6.5735   5.0081  5.2180  0.034   143
 500  4.4003   2.9618  4.7949  0.300    75
1000  2.3820   1.4994  2.9421  0.597   135
2000  1.2255   0.6262  1.9977  0.810   153
3000  0.9153   0.4048  1.7018  0.877   157
4000  0.6571   0.2695  1.2920  0.918    79
5000  0.6897   0.2370  1.5093  0.928   112  val_WER=76.97%
```

Note: V5's final loss (0.69) > V2's (0.57), yet V5's WER is 6.5% better. Dynamic batching produces noisier per-step losses but better model quality through more uniform gradient signal across utterance lengths.
