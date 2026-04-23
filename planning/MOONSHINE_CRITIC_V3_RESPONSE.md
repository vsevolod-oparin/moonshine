# Response to MOONSHINE_CRITIC_V3.md

Point-by-point evaluation: **ACCEPT** / **PARTIALLY ACCEPT** / **DECLINE** with reasoning and proposed resolution.

---

## 1. No decode strategy (beam search + LM) — **ACCEPT**

Valid and important. The plan already trains a KenLM for pseudo-label filtering (Section 6.4) — reusing it for shallow fusion during decoding gives 10-15% relative WER improvement at zero additional training cost. This is the highest-impact gap in V3.

**Resolution**: Add new Section 7.6 (Decoding Strategy). Add greedy vs. beam+LM WER to evaluation. Specify beam width and LM integration.

---

## 2. No regularization beyond label smoothing — **ACCEPT**

Valid. Training 123M params on 5.4K hours is low-resource. English Moonshine's 300K hours provided implicit regularization; we need explicit dropout. Standard practice in every production ASR system.

**Resolution**: Add new Section 6.10 (Regularization). Add dropout to training config tables in Phase 1 and Phase 2.

---

## 3. v2.1 stride-2 operations undefined — **ACCEPT**

This is a correctness issue. Average pooling (non-causal) would silently break streaming. Must specify explicit causal operations.

**Resolution**: Expand Section 3.3 Change 2 with explicit downsample/upsample operations.

---

## 4. No distributed training strategy — **ACCEPT (minor)**

Valid but operational. Single H100 should work for Small (123M). Medium (245M) may need multi-GPU. Worth a brief note.

**Resolution**: Add to Training Infrastructure (Section 9). Phase 1-2: single GPU with gradient accumulation. Phase 3 Medium: FSDP if needed.

---

## 5. No error analysis framework — **ACCEPT**

Valid. Without error categorization, iterative improvement is blind. After each phase, understanding WHERE errors happen (clusters, vowel reduction, proper nouns) guides next steps.

**Resolution**: Add error analysis subsection to Section 10 (Evaluation). Run on 500+ validation samples after each phase.

---

## 6. No inference benchmarking methodology — **ACCEPT**

Valid. Latency claims like "< 100ms on MacBook CPU" are meaningless without specifying which MacBook, which runtime, which conditions. Reproducibility requires a protocol.

**Resolution**: Add benchmarking protocol to Section 10 (Evaluation).

---

## 7. No curriculum learning — **PARTIALLY ACCEPT**

**ACCEPT**: The idea is sound — clean data first, then noisy. Standard in low-resource ASR.

**DECLINE**: Not worth prescribing in detail. Curriculum effectiveness is dataset-dependent and hard to predict. Better to add as an optional Phase 2 note and test empirically.

**Resolution**: Add optional curriculum note to Phase 2 training config. If validation WER plateaus early, try clean-first curriculum as a tuning knob.

---

## 8. Speaker overlap enforcement — **ACCEPT (minor)**

Valid but the critic's own analysis is correct: cross-dataset speaker deduplication is essentially impossible. Within-dataset splitting by speaker ID (available for CV, MLS) is the best we can do.

**Resolution**: Add note to Section 5 (Data Preprocessing): split by speaker ID within each dataset. Acknowledge minor cross-dataset overlap as accepted noise.

---

## 9. No experiment tracking — **ACCEPT (minor)**

Valid but operational. Any of wandb/TensorBoard/MLflow works. Not an architectural decision.

**Resolution**: Add brief note to Training Infrastructure (Section 9). Log: loss, WER, LR, gradient norms, per-dataset WER, git commit hash.

---

## 10. Text normalization edge cases — **ACCEPT**

Valid. Number normalization is covered but abbreviations, dates, and hyphenated words are real gaps in Russian text processing.

**Resolution**: Expand Section 5 text normalization with abbreviation dictionary, date handling, and hyphenated word rules. Add edge cases to T12 test.

---

## 11. BPE subword regularization — **ACCEPT**

Valid. One-line parameter change (`nbest_size=5, alpha=0.1`) during training. Deterministic at inference. Minor robustness gain for free.

**Resolution**: Add to Section 4 (Tokenizer Considerations).

---

## 12. Attention sink handling — **PARTIALLY ACCEPT**

**ACCEPT**: Worth monitoring. The XLSR-Transducer finding is relevant.

**DECLINE**: Preemptive mitigation (learnable sink token) adds complexity for an unconfirmed problem. Moonshine's ergodic encoder (no positional embeddings) may not exhibit this behavior.

**Resolution**: Add monitoring to Phase 1 evaluation: visualize encoder attention patterns on 10 utterances. If sinks appear, add sink token mitigation in Phase 2. No code change now.

---

## 13. FLOPs budget analysis — **ACCEPT (minor)**

Valid. Params determine size; FLOPs determine speed. The v2.1 multi-scale should be faster than v2 despite more params (Stage 2 at half frame rate). Worth quantifying.

**Resolution**: Add FLOPs estimate to Section 3.3 parameter table. Brief calculation showing v2.1 encoder has ~20% fewer FLOPs than v2 encoder.

---

## 14. Knowledge distillation — **ACCEPT**

Phase 4 already mentions distillation but is under-specified. The critic's suggestion to add KL divergence + CTC + hidden-state matching is concrete.

**Resolution**: Expand Phase 4 distillation details. Already in the plan, just needs more specificity.

---

## 15. BiasNorm for v2.1 — **DECLINED**

BiasNorm is a Zipformer-specific innovation tied to their SwooshR/SwooshL activation functions and ScaledAdam optimizer. Moonshine uses standard LayerNorm + SwiGLU + Schedule-Free. Mixing normalization schemes without the rest of Zipformer's stack provides no proven benefit and adds compatibility risk.

If Phase 1 ablation shows normalization issues specifically (unlikely), reconsider. Otherwise, keep LayerNorm for consistency with v2 and English Moonshine.

---

## 16. CI / reproducibility — **ACCEPT (minor)**

Valid but operational. Dockerfile + dependency pinning + config-per-run is standard practice.

**Resolution**: Add to Phase 0 setup checklist. Each training run gets a unique config file + git commit hash. Dockerfile for cloud GPU runs.

---

## 17. Streaming failure modes — **ACCEPT**

Valid and not discussed anywhere in the plan. VAD failures, truncation, fast/slow speech, and background speech are real production issues.

**Resolution**: Add "Streaming Failure Modes" subsection to Section 11 (Deployment). Specify mitigations for each case.

---

## 18. Dialectal / accented speech evaluation — **PARTIALLY ACCEPT**

**ACCEPT**: The gap is real. Russian dialectal variation and accented speech (Ukrainian-accented, Central Asian) are important for real-world deployment.

**DECLINE**: Dedicated dialectal evaluation is a stretch goal. Collecting accented speech samples is out of scope for the training plan.

**Resolution**: Acknowledge in Section 10 (Evaluation) as a known coverage gap. Add to Phase 4 stretch goals: collect 50-100 diverse-accent samples for coverage testing.

---

## 19. Streaming deduplication strategy — **ACCEPT**

Section 3.4 mentions "deduplication handles overlap" but never specifies how. The critic's recommendation (word-level overlap comparison with last 2 words) is practical.

**Resolution**: Expand Section 3.4 token emission with explicit deduplication strategy.

---

## 20. Early stopping patience — **ACCEPT (minor)**

The plan says "patience = 6K steps" (3 evaluations × 2K step interval). This is already specific. But the critic is right that it should be stated more prominently.

**Resolution**: No change needed — already specified in Section 9 Training Infrastructure. The existing text is adequate.

---

## Summary

| # | Criticism | Verdict | Action |
|---|-----------|---------|--------|
| 1 | No decode strategy (beam search + LM) | **ACCEPT** | Add Section 7.6, update eval and deployment |
| 2 | No regularization (dropout) | **ACCEPT** | Add Section 6.10, update training configs |
| 3 | v2.1 stride-2 ops undefined | **ACCEPT** | Expand Section 3.3 Change 2 |
| 4 | No distributed training plan | **ACCEPT (minor)** | Add to Section 9 Training Infrastructure |
| 5 | No error analysis framework | **ACCEPT** | Add to Section 10 |
| 6 | No benchmarking methodology | **ACCEPT** | Add to Section 10 |
| 7 | No curriculum learning | **PARTIALLY ACCEPT** | Optional note in Phase 2 |
| 8 | Speaker overlap enforcement | **ACCEPT (minor)** | Add note to Section 5 |
| 9 | No experiment tracking | **ACCEPT (minor)** | Add to Section 9 |
| 10 | Text normalization edge cases | **ACCEPT** | Expand Section 5 |
| 11 | BPE subword regularization | **ACCEPT** | Add to Section 4 |
| 12 | Attention sink handling | **PARTIALLY ACCEPT** | Monitor in Phase 1, add to eval |
| 13 | FLOPs budget analysis | **ACCEPT (minor)** | Add to Section 3.3 |
| 14 | Knowledge distillation | **ACCEPT** | Expand Phase 4 |
| 15 | BiasNorm for v2.1 | **DECLINED** | Zipformer-specific, not applicable |
| 16 | CI / reproducibility | **ACCEPT (minor)** | Add to Phase 0 |
| 17 | Streaming failure modes | **ACCEPT** | Add to Section 11 |
| 18 | Dialectal evaluation | **PARTIALLY ACCEPT** | Acknowledge gap in Section 10 |
| 19 | Streaming dedup strategy | **ACCEPT** | Expand Section 3.4 |
| 20 | Early stopping patience | **ACCEPT (minor)** | Already specified, no change needed |

**Score**: 13 accepted, 4 partially accepted, 1 declined, 2 accepted-minor-no-change. 19 of 20 points have some validity.

### Top 3 integrations by impact:

1. **Decode strategy** (beam search + shallow LM fusion) — single biggest free WER improvement
2. **Regularization** (dropout) — prevents overfitting on 5.4K hours
3. **v2.1 stride-2 operations** — correctness fix, prevents silent streaming breakage
