# Response to MOONSHINE_DS_CRITIC.md

Point-by-point evaluation: **ACCEPT** / **PARTIALLY ACCEPT** / **DECLINE** with reasoning and proposed resolution.

---

## 1. No held-out test set strategy — **ACCEPT**

Valid. The plan says "95/5 split" but Common Voice and Golos both ship with official test splits. Not using them as the locked test set would be a methodological error — you'd optimize against validation and report against the same data.

**Resolution**: Add to Section 10 (Evaluation):

> **Test set (locked, never used for validation or HP tuning)**:
> - Common Voice 19 ru official test split
> - Golos official test split
> - Total: ~100-150 hours
>
> **Validation set** (for checkpoint selection and HP tuning):
> - 5% of training data, no speaker overlap with train or test
>
> Report test WER **once** per finalized model. Do not iterate on test.

---

## 2. No data mixing strategy — **ACCEPT**

Valid. Uniform shuffling across domains with different acoustic properties (read speech, far-field, audiobooks) risks the model overfitting to whichever domain is cleanest or most represented. This is a known pitfall in multi-dataset ASR training.

**Resolution**: Add to Section 5 (Data):

> **Data mixing**: Temperature-based sampling with α=0.7:
> ```
> p_i ∝ hours_i^0.7
> ```
> This slightly upweights smaller datasets (SOVA, RuLS) relative to raw proportions, ensuring the model sees enough far-field and telephony audio. Whisper uses this approach. Validate mixing by checking per-dataset validation WER during Phase 1 — if any dataset's WER is >2x the average, adjust temperature.

---

## 3. WER targets are optimistic — **PARTIALLY ACCEPT**

The comparison table is misleading in both directions:

**Where the critic is right**: 10% for v2 Small on 5.4K hours is aggressive. The data-to-accuracy curve in ASR is well-studied — diminishing returns start early but you still need a minimum floor.

**Where the critic overstates**:
- The comparison uses Moonshine **Tiny** (34M) achieving 12% on 300K hours. But the plan targets **Small** (123M, 3.6x more capacity). Moonshine Small English achieves much better than Tiny.
- Russian is more phonetic than English — orthography is closer to pronunciation, reducing acoustic ambiguity.
- GigaAM's 8.4% is the ceiling, not the baseline. GigaAM uses 700K hours SSL pretraining + a 240M parameter model. Our 123M model won't match it.
- Data quality vs quantity: 5.4K hours of supervised Common Voice/Golos is higher signal-per-hour than Moonshine's likely mix of web-scraped audio.

**Resolution**: Adjust targets to tiered expectations:

| Scenario | v2 Small WER | v2.1 Small WER | Condition |
|----------|-------------|----------------|-----------|
| Phase 2 baseline (5.4K hrs) | 12-15% | 10-13% | Realistic |
| Phase 3 + pseudo-labeling (25K hrs) | 10-12% | 8-10% | Aspirational |
| Stretch goal | <10% | <8% | Requires SSL pretrain or 50K+ hrs |

Keep the original targets as "stretch" but set honest baselines. This doesn't change the plan — just expectations.

---

## 4. No validation WER during training — **ACCEPT**

Valid. The PoC tests (T5-T6) use WER as a gate, and Section 10 defines WER as the primary metric. But the training plan (Section 9) never says "evaluate WER every N steps during training." This is a gap — loss alone is insufficient for ASR checkpoint selection.

**Resolution**: Add to each Phase in Section 9:

> **Validation**: Evaluate WER on validation set every 2K steps via CTC greedy decoding. Log WER alongside loss. Select checkpoints by WER, not loss. Abort training if WER stops improving for 3 consecutive evaluations (early stopping with patience).

---

## 5. Checkpoint strategy missing — **ACCEPT (minor)**

Valid but operational detail, not architectural gap. Still worth specifying since cloud GPU failures are common.

**Resolution**: Add brief note to Section 9:

> **Checkpoints**: Save every 5K steps. Retain top-3 by validation WER + latest. Auto-resume from latest on restart. Storage estimate: ~250MB per Small BF16 checkpoint, ~5GB total per run.

---

## 6. Pseudo-labeling filter is under-specified — **PARTIALLY ACCEPT**

The critic is right that "beam score threshold" is thin. However, the detailed filtering pipeline proposed (LM perplexity, manual spot-checks) is premature at the plan stage — you design filtering based on observed noise rates from the actual model.

**Resolution**: Add to Section 6.4:

> **Filtering strategy**: Multi-stage: (1) confidence threshold (top 60th percentile by beam score), (2) length sanity (3-500 chars), (3) language model filter (KenLM trained on Russian Wikipedia, reject if perplexity > 95th percentile of labeled data), (4) spot-check 200 random samples per iteration. Expected noise rate: 10-30%. Retrain only on filtered subset. Iterate if noise > 25%.

Keep it specific enough to implement but acknowledge that thresholds will be calibrated based on Phase 2 model quality.

---

## 7. Mobile deployment section is thin — **PARTIALLY ACCEPT**

**ACCEPT**: The latency budget breakdown is genuinely useful and missing from the plan. End-to-end latency != encoder TTFT.

**DECLINE**: CoreML conversion issues, audio thread architecture, and Android chip variance are implementation details. The plan is an architecture and training document, not a mobile engineering guide. Those concerns belong in a deployment runbook, not the plan.

**Resolution**: Add a latency budget to Section 11 (Deployment):

> **End-to-end streaming latency budget (Small, INT8, phone-class CPU)**:
>
> | Stage | Latency | Notes |
> |-------|---------|-------|
> | Audio capture + buffering | 20-40ms | One frame at 50Hz = 20ms |
> | VAD (Silero) | <5ms | Runs on buffered audio |
> | Feature extraction | <5ms | CMVN + asinh + conv |
> | Encoder (cached, per frame) | 2-5ms | KV cache reuse |
> | Decoder (N tokens) | N × 3-8ms | N=3-8 typical for Russian |
> | **Total TTFT** | **30-60ms** | Within <100ms target |
>
> Note: This is inference latency only. Audio capture pipeline (AVAudioEngine/AudioRecord) adds OS-dependent overhead (10-30ms) outside the model's control.

---

## 8. No ablation plan for v2.1 — **ACCEPT**

Valid. Without ablation, a v2.1 improvement is unattributable. Is it causal conv? Multi-scale? SSC? The proposed 5-step ablation is feasible during Phase 1 (Tiny training is cheap — 2-3 days on 3090, can run variants sequentially).

**Resolution**: Add to Phase 1 in Section 9:

> **Ablation plan (Phase 1 Tiny)**:
>
> | Run | Architecture | Purpose |
> |-----|-------------|---------|
> | A | v2 baseline | Control |
> | B | v2 + causal conv (k=7) | Isolate conv contribution |
> | C | v2 + multi-scale U-Net | Isolate multi-scale contribution |
> | D | v2 + conv + multi-scale | Combined without SSC |
> | E | Full v2.1 (conv + multi-scale + SSC) | Full stack |
>
> Compare WER on held-out test. If B ≈ E, skip multi-scale and SSC — they add complexity without gain. This ablation runs on Tiny (cheap) and determines the v2.1 architecture for Phase 2 Small training.

---

## 9. ONNX streaming export complexity understated — **PARTIALLY ACCEPT**

**ACCEPT**: The technical details are correct — streaming ONNX export requires explicit KV cache inputs/outputs and dynamic axes. Worth a brief note.

**DECLINE**: The concern is already addressed by PoC tests T14 (ONNX export smoke test) and T15 (ONNX streaming test). The plan explicitly gates on these passing. Adding a paragraph about ONNX export mechanics to the architecture section would duplicate what the PoC tests already cover.

**Resolution**: Add a brief note to Section 7.4:

> **Streaming ONNX export**: Requires explicit KV cache tensors as graph inputs/outputs with `dynamic_axes` for variable-length sequences. The Moonshine repo's existing export scripts provide a working starting point for v2. v2.1 multi-scale encoder requires custom cache management (per-stage buffers). PoC tests T14-T15 validate this before committing to training.

---

## 10. Gradient accumulation not mentioned — **ACCEPT (minor)**

Valid but minor. The plan gives per-GPU batch sizes without specifying effective batch size or accumulation steps. This is standard practice to clarify.

**Resolution**: Add to Phase 2 in Section 9:

> **Effective batch size**: Target 128-256 via gradient accumulation. On H100 with Small batch=16 per GPU, accumulation steps = 8-16. On 3090 with Small batch=8, accumulation steps = 16-32.

---

## 11. No Russian-specific evaluation nuance — **ACCEPT**

Valid and well-argued. Russian case endings are phonetically reduced in spoken language — "большой дом" vs "большого дома" sounds nearly identical in connected speech. WER penalizes this as a full word error. CER partially addresses it but a phoneme-normalized metric would be more honest.

**Resolution**: Add to Section 10:

> **Secondary metrics**:
> - CER (already listed) — less sensitive to morphological variation
> - **G2P-normalized WER**: Convert both reference and hypothesis to phonemes using Russian G2P before computing WER. This ignores orthographic case-ending variation (acceptable errors) while catching phonetic errors (real errors). Use `ru_num2words` for number normalization + a rule-based Russian G2P (Russian is nearly phonetic — G2P is straightforward).

---

## 12. No thermal/practical guidance for 3090 — **DECLINED**

This is not a plan gap. It is practical hardware advice. Any ML practitioner running multi-day GPU training is aware of thermals. Including GPU cooling tips in an architecture and training plan would be noise.

If the plan eventually has a "practical notes" appendix, this could go there. But it doesn't belong in the main document.

---

## Additional Recommendations Evaluation

| Recommendation | Verdict | Reasoning |
|----------------|---------|-----------|
| Russian G2P-normalized WER | **ACCEPT** | Covered by point 11. Worth adding. |
| Trigger word / wake word test | **DECLINED** | Wake word detection is a separate system from ASR. Different model, different objective, different evaluation. Out of scope for this plan. |
| Server-side throughput benchmark | **DECLINED** | The plan explicitly targets edge deployment. Adding server-side benchmarks scope-creeps the document. If the model is later used server-side, benchmark it then. |
| Streaming user study (5 raters) | **PARTIALLY ACCEPT** | Valuable for product validation, but belongs in Phase 4 (deployment), not the training plan. Add as a Phase 4 activity. |
| Validation WER in training loop | **ACCEPT** | Covered by point 4. Already accepted. |
| Dataset temperature sampling | **ACCEPT** | Covered by point 2. Already accepted. |

---

## Summary

| # | Criticism | Verdict | Action |
|---|-----------|---------|--------|
| 1 | No held-out test set strategy | **ACCEPT** | Add locked test set definition to Section 10 |
| 2 | No data mixing strategy | **ACCEPT** | Add temperature-based sampling to Section 5 |
| 3 | WER targets optimistic | **PARTIALLY ACCEPT** | Add tiered WER expectations (baseline / aspirational / stretch) |
| 4 | No validation WER during training | **ACCEPT** | Add WER eval every 2K steps to Section 9 phases |
| 5 | Checkpoint strategy missing | **ACCEPT (minor)** | Add checkpoint policy to Section 9 |
| 6 | Pseudo-labeling filter under-specified | **PARTIALLY ACCEPT** | Add multi-stage filtering description to Section 6.4 |
| 7 | Mobile deployment thin | **PARTIALLY ACCEPT** | Add latency budget breakdown to Section 11 |
| 8 | No ablation plan for v2.1 | **ACCEPT** | Add 5-step ablation to Phase 1 |
| 9 | ONNX export complexity understated | **PARTIALLY ACCEPT** | Add brief streaming export note to Section 7.4 |
| 10 | Gradient accumulation not mentioned | **ACCEPT (minor)** | Add effective batch size + accumulation to Section 9 |
| 11 | No Russian-specific evaluation | **ACCEPT** | Add G2P-normalized WER to Section 10 |
| 12 | No thermal guidance for 3090 | **DECLINED** | Out of scope for an architecture/training plan |

**Score**: 8 accepted, 4 partially accepted, 1 declined. The critic raised legitimate concerns — 11 of 12 points have some validity. The plan benefits from incorporating these resolutions.
