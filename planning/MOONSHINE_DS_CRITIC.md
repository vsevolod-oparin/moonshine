# Critical Evaluation of ru-Moonshine Plan

Weak spots, missing points, and additional recommendations for `MOONSHINE_PLAN.md`.

---

## Weak Spots

### 1. No held-out test set strategy

The plan says "95/5 train/validation split, no speaker overlap" but doesn't define which datasets form the official **test set**. Common Voice has a dedicated test split; Golos has one too. If validation data from CV/Golos leaks into test reporting, results are invalid.

**Fix**: Designate a held-out test set **before training begins** — never used for validation or HP tuning. Suggested: CV-ru test (official) + Golos test (official) = ~150 hours. Lock it. Report only once per model.

---

### 2. No data mixing strategy

5.4K hours across 5+ datasets with wildly different acoustic conditions:
- Common Voice: read speech, clean, web-collected
- Golos: far-field + close-talk, device-recorded
- MLS: audiobooks
- RuLS: read speech
- SOVA: varied

Training will overfit to whatever dataset dominates sampling or is easiest (clean read speech). The plan assumes uniform shuffling works.

**Fix**: Specify a sampling strategy — proportional (weight by hours), temperature-balanced (`p_i ∝ hours_i^α` with α=0.5-0.8), or curriculum (clean → noisy). Whisper uses dataset temperature to balance.

---

### 3. WER targets are optimistic for 5.4K hours

| Model | Data | Params | WER |
|-------|------|--------|-----|
| Moonshine v2 Tiny | 300K hours English | 34M | 12% |
| GigaAM-v3 | 700K hrs SSL pretrain + ~1K hrs fine-tune | 240M | 8.4% |
| ru-Moonshine v2 Small target | **5.4K hours** | 123M | **9-10%** |

The plan targets ~10% WER from 5.4K hours when Moonshine Tiny achieves 12% from **55× more data**. GigaAM relies on 700K hours of SSL pretraining before supervised fine-tuning. T-one uses proprietary telephony data.

**Fix**: Set realistic expectations. 12-15% for v2 Small is more plausible. Frame 10% as aspirational, achieved only with pseudo-labeling (Phase 3). The gap is bridgeable but honesty about difficulty prevents disappointment.

---

### 4. No validation WER during training

Loss is a weak signal for ASR. Training converges to low-loss but poor-WER states (predicting blanks or most-common token). The plan's post-hoc evaluation catches this too late.

**Fix**: Build WER evaluation into the training loop from Phase 0 (even Tiny on 700h). Run CTC greedy decoding on validation set every 1K-5K steps. Log WER alongside loss. Gate checkpoint selection on WER, not loss.

---

### 5. Checkpoint strategy missing

Training for 2-3 days on cloud H100 — zero discussion of checkpoint frequency, retention, or resume logic. A failure at hour 47 could lose everything.

**Fix**: Specify: checkpoint every 5K steps, keep top-3 by validation WER, auto-resume from latest. Include estimated storage (Small BF16: ~250MB per checkpoint, ~5GB for a run with 20 checkpoints).

---

### 6. Pseudo-labeling filter is under-specified

> "Filter low-confidence labels (beam score threshold)"

Beam score is a poor filter — high-confidence wrong transcripts are well-documented. No discussion of how to determine the threshold, validate label quality, or handle the iterative retraining loop.

**Fix**: Add a filtering strategy: confidence threshold (e.g., beam score > top 20th percentile) + transcript length sanity (min 3 chars, max 500) + LM perplexity filter against a Russian KenLM (train on Ru Wikipedia) + manual spot-check of 100 random labels per 10K hours. Document the expected label noise rate (~10-30%).

---

### 7. Mobile deployment section is thin

| Platform | Claim | Issue |
|----------|-------|-------|
| iPhone CoreML | "Tiny ONNX → CoreML" | CoreML tool ONNX→MLPackage conversion often fails on custom ops (conv transpose, custom attention masks). Needs validation. |
| iPhone memory | "Tiny ~50MB, fits" | Fine for iPhone 14+ (6GB RAM). iPhone 12 (4GB): 50MB model + audio buffer + VAD + app = ~200MB — acceptable but not trivial. |
| Android NNAPI | "Small <150ms" | NNAPI latency varies wildly by chip (Snapdragon vs Tensor vs Exynos). No benchmark methodology discussed. |
| Audio pipeline | Not addressed | Which thread? Audio callback → ring buffer → VAD → inference → output. Latency from microphone to inference call is architecture-dependent (iOS AVAudioEngine vs Android AudioRecord). |

**Fix**: Add an inference latency budget breakdown: microphone capture (20-30ms) + VAD (10ms) + feature extraction (5ms) + encoder per frame (2-3ms cached) + decoder (N×5ms). This reveals the actual pipeline latency vs the claimed TTFT.

---

### 8. No ablation plan for v2.1

Training v2 and v2.1 as monolithic variants tells us nothing about which improvement caused any accuracy delta. Causal conv? Multi-scale U-Net? SSC cross-window? Unknown.

**Fix**: Add an ablation pipeline for Phase 1 Tiny:
1. v2 baseline
2. v2 + causal conv only
3. v2 + multi-scale only
4. v2 + causal conv + multi-scale (no SSC)
5. Full v2.1

Compare WER on held-out test. If #2 equals #5, skip the complexity of multi-scale and SSC.

---

### 9. ONNX streaming export complexity understated

The plan says "standard ONNX export" but streaming requires:
- KV cache tensors as explicit graph **inputs** (not model state)
- Updated KV cache tensors as graph **outputs**
- `dynamic_axes` for variable-length inputs — ONNX Runtime supports this but shape inference can be fragile
- TorchScript trace → ONNX path may fail if dynamic control flow (if statements based on cache state)

**Fix**: Dedicate a subsection to ONNX streaming export. Note that it requires `torch.onnx.export` with explicit cache placeholders and `dynamic_axes`. Reference the Moonshine repo's existing export code as a starting point (v2). For v2.1, expect custom debugging.

---

### 10. Gradient accumulation not mentioned

Batch sizes in the plan (16-32 Tiny, 16-24 Small) are per-GPU. Effective batch size determines convergence quality. No mention of gradient accumulation steps.

**Fix**: Add gradient accumulation to reach effective batch size of 128-256 (standard for AED ASR). On 3090 with Small batch=8, accumulation steps=16 → effective batch=128. Already well within budget.

---

### 11. No Russian-specific evaluation nuance

Russian has grammatical cases, verb conjugations, and free word order. WER penalizes inflection errors (wrong case ending) the same as word substitution errors — but these have very different perceptual impact. A model that says "большой дом" instead of "большого дома" scores a 50% WER but is perfectly understandable.

**Fix**: Add CER + Russian G2P-normalized WER (convert to phonemes before comparison) as secondary metrics. CER is less sensitive to morphological variation. G2P WER catches phonetic errors (bad) while ignoring case-ending variation (acceptable).

---

### 12. No thermal/practical guidance for 3090

2-3 day continuous training on a home RTX 3090 generates significant heat (~350W sustained). Ambient temperature rise of 5-10°C in a closed room is realistic. Thermal throttling can silently reduce throughput by 10-20%.

**Fix**: Brief practical note: ensure case airflow, monitor GPU temp (target <80°C), consider undervolting for sustained loads. Not a plan flaw but omission means a newcomer could hit mysterious slowdowns.

---

## Additional Recommendations

| Item | Priority | Rationale |
|------|----------|-----------|
| Russian G2P-normalized WER metric | Medium | Differentiates phonetic errors from morphological variation. |
| Trigger word / voice activity test | Low | For voice assistant use case — measure false accept on 100h of non-speech. |
| Server-side throughput benchmark | Low | If model is also used for batch transcription. |
| Streaming-first user study (5 human raters) | Medium | WER doesn't capture subjective quality. 50 real queries vs Vosk comparison. |
| Validation WER in training loop | **High** | Catch training issues early, gate checkpoint selection. |
| Dataset temperature sampling | Medium | Prevent clean-read-speech overfit, ensure robustness. |

---

## Summary

**Strongest sections**: PoC test suite, shared training tricks, cache-aware inference, tokenizer analysis.

**3 biggest gaps**:
1. **WER targets are too optimistic** — 10% for Small on 5.4K hours is aspirational, not baseline. Realistic: 12-15%. Bridgeable via pseudo-labeling.
2. **No validation WER during training** — must be built into the training loop from Phase 0, not post-hoc.
3. **No data mixing strategy** — uniform shuffling of 5+ diverse datasets risks overfit to the easiest domain.

The plan is well-structured and covers most important topics. These gaps are not fatal — they're refinements for a plan that's already solid.
