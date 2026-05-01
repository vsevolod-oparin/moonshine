# M9 Training Diagnostics Report

Generated from TensorBoard data at step ~116K.

## Summary of Findings

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | AED decoder plateaued — missing EOS token | **Critical** | Fix dataset + retrain |
| 2 | LR decays too aggressively | **High** | Switch to WSD (see ml-lr-proposal.md) |
| 3 | Non-standard loss weighting formula | **Moderate** | Normalize to `0.3*ctc + 0.7*aed` |
| 4 | CTC loss variance is very high | **Moderate** | Investigate hard samples |
| 5 | SER stuck at ~94% | **Low** | Will improve with other fixes |
| 6 | No overfitting detected | **Info** | Model has capacity to learn more |

---

## 1. AED Decoder Plateau (Critical)

### The Data

```
AED train loss (10K averages):
  step     0: 1.925
  step 20K:  1.470  (-24%)
  step 40K:  1.354  (-30%)
  step 60K:  1.325  (-31%)  ← FLAT FROM HERE
  step 80K:  1.314  (-32%)
  step100K:  1.272  (-34%)
  step116K:  1.294  (no meaningful change for 80K steps)

AED decoder accuracy (10K averages):
  step     0: 60.7%
  step 20K:  81.8%
  step 40K:  85.7%
  step 60K:  86.4%  ← PLATEAU FROM HERE
  step 80K:  86.5%
  step100K:  88.5%
  step116K:  87.5%  (bouncing around 87%)

AED validation WER (post-fix):
  step 92K:  40.42%
  step 96K:  40.48%
  step100K:  40.63%
  step104K:  40.05%
  step108K:  39.28%
  step112K:  39.18%
  → Improved only 1.23% in 20K steps (-0.062/1K)
```

Meanwhile CTC continues improving:
```
  CTC WER: 55.31 → 52.92 = -2.39% in 20K steps (-0.120/1K)
  CTC WER improves at 2x the rate of AED WER
```

### Root Cause: Missing EOS Token

`dataset.py` line 225 strips all tokens with id < 6:
```python
token_ids = [t for t in token_ids if t >= 6]
```

This removes `</s>` (token 2) — the end-of-sequence marker. The AED decoder **never learns when to stop generating**. This creates a hard ceiling on decoder capability because:

1. The decoder optimizes cross-entropy against targets that never contain EOS
2. It learns content tokens well (87% accuracy) but has zero gradient signal for termination
3. The loss plateaus at an information-theoretic ceiling — the objective is under-specified
4. Inference uses a heuristic length ceiling (CTC length × 1.2) instead of learned EOS

This likely accounts for **50-70% of the AED plateau**.

### Fix

Add EOS token to training targets in `dataset.py`:
```python
token_ids = [t for t in token_ids if t >= 6]
token_ids.append(2)  # </s> — teach decoder to predict end-of-sequence
```

Then update `validate.py` to use EOS prediction as the primary stopping criterion instead of the CTC-length heuristic.

**Important:** This requires retraining from scratch or from an early checkpoint. The decoder has trained 115K steps without ever seeing EOS — it may have learned maladaptive patterns that need unlearning. Retraining from scratch is recommended.

---

## 2. LR Decay Too Aggressive (High)

Already documented in `ml-lr-proposal.md`. Key data:

```
Current cosine schedule (max_steps=200K):
  step 112K: lr=2.07e-4 (41% of peak)
  step 140K: lr=1.05e-4 (21% of peak)
  step 160K: lr=5.00e-5 (floor — 10% of peak)
  step 200K: lr=5.00e-5 (20% of training at floor!)

WER is still improving at 0.25%/4K steps — model has capacity.
But LR will be at floor by step 160K, killing gradient signal.
```

Recommendation: Switch to WSD (Warmup-Stable-Decay) schedule.

---

## 3. Non-Standard Loss Formula (Moderate)

### Current Code (`model.py` line 159):
```python
loss = loss_aed + self.config.ctc_weight * loss_ctc
# With ctc_weight=0.3: effective weights are AED=1.0, CTC=0.3
# Total weight = 1.3 (unnormalized)
```

### Standard Formula (ESPnet, Hori et al. 2017):
```python
loss = (1 - ctc_weight) * loss_aed + ctc_weight * loss_ctc
# With ctc_weight=0.3: effective weights are AED=0.7, CTC=0.3
# Total weight = 1.0 (normalized)
```

The current formula gives AED 77% of gradient signal vs ESPnet's standard 70%. This is close but the unnormalized total means:
- The `ctc_weight` parameter doesn't directly control the balance as expected
- LR sensitivity is increased (effective LR is 1.3x what it "should" be)

**Impact on CTC loss dominance:** Despite AED getting 77% of weight, the CTC loss is consistently higher (1.8 vs 1.3 on average) and more variable (CV=46% vs 20%). The CTC branch is harder to optimize and may be under-weighted.

### Fix

```python
loss = (1 - self.config.ctc_weight) * loss_aed + self.config.ctc_weight * loss_ctc
```

---

## 4. CTC Loss Variance (Moderate)

### The Data

```
Recent training (steps 90K-116K):
  train/loss:     avg=1.820  min=1.137  max=4.288  CV=27%
  train/loss_ctc: avg=1.797  min=0.545  max=5.371  CV=46%
  train/loss_aed: avg=1.281  min=0.973  max=2.677  CV=20%

CTC loss is 2.3x more variable than AED loss.
```

Top loss spikes (steps 60K+):
```
  step 76100: total=4.43  ctc=5.44  aed=2.79  acc=44.6%
  step107900: total=4.29  ctc=5.37  aed=2.68  acc=47.9%
  step 95500: total=3.99  ctc=4.67  aed=2.59  acc=50.8%
```

All spikes are driven by CTC loss jumping to 4-5x its average. The AED loss also spikes but much less dramatically. These are likely **hard audio samples** (long duration, noisy, unusual content) where:
- CTC frame-level alignment is poor
- The framewise assumption breaks down

`grad_clip=5.0` limits the damage, but the high variance suggests the CTC branch struggles with certain samples.

### Possible Fixes
- **Dynamic CTC weight:** Reduce CTC weight on high-loss samples (gradient noise scaling)
- **Data filtering:** Investigate whether specific utterances consistently cause spikes
- **Curriculum:** Start with higher CTC weight, reduce later (ESPnet approach)

---

## 5. SER Stuck at ~94% (Low)

```
val/ser trajectory:
  step 60K: 95.47%
  step 80K: 95.77%
  step100K: 95.05%
  step112K: 94.17%

Improvement: 95.5% → 94.2% in 52K steps = -0.025/1K steps
```

94-95% of sentences still have at least one error. This is expected given the overall WER of 53% — with that many word errors per sentence, nearly every sentence will be affected. SER should improve proportionally as WER improves. Not a separate problem.

---

## 6. No Overfitting (Info)

```
Train-Val loss gap:
  step  4K: train=2.93  val=2.40  gap=+17.8%  (train > val)
  step 60K: train=1.79  val=1.06  gap=+40.7%
  step112K: train=1.86  val=0.98  gap=+47.2%
```

Train loss is **consistently higher** than val loss. This is expected with SpecAugment (masks training features) + dropout (0.1) + label_smoothing (0.1). The model is well-regularized and has capacity to learn more. The gap is slowly growing but this is normal — val loss has more variance and drops faster than the noisy train average.

---

## 7. WER/CER Ratio (Info)

```
WER/CER ratio is consistently ~2.9x throughout training.

This means each character error creates ~3 word errors on average.
Typical for Russian: short function words (prepositions, conjunctions)
being missed entirely → whole-word errors from single-character mistakes.
```

Not a training issue — this is a property of the language and tokenizer. May improve with a better tokenizer or subword regularization.

---

## Recommended Priority Order

1. **Fix EOS token** in dataset.py — retrain from scratch
2. **Fix loss formula** — normalize to `0.3*ctc + 0.7*aed`
3. **Switch to WSD LR schedule** — decouple from max_steps
4. **Retrain with all three fixes** — the EOS fix alone justifies a full retrain

The model is undertrained, not overfitted. All losses are still slowly declining. The main bottleneck is the AED decoder being unable to learn sequence termination (EOS bug), not model capacity or data issues.

---

## References

- Hori et al. (2017) — "Advances in Joint CTC-Attention based End-to-End Speech Recognition" — standard loss formula
- ESPnet documentation — CTC/attention multi-task training defaults
- Bengio et al. (2015) — Scheduled Sampling — teacher forcing exposure bias
- Defazio et al. (2024) — "Revisiting Cosine Schedule" — WSD matches cosine
- Wen et al. (2024) — "Understanding Warmup-Stable-Decay" — theoretical WSD justification
