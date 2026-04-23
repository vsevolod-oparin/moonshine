# Third-Pass Critical Evaluation of ru-Moonshine Plan

Gaps not covered by MOONSHINE_DS_CRITIC.md, MOONSHINE_DS_COMMENTS.md, MOONSHINE_DS_EVALUATION.md, MOONSHINE_DS_CRITIC_RESPONSE.md, MOONSHINE_CRITIC_V2.md, or MOONSHINE_GLM_COMMENTS.md.

Previous reviews covered: test set strategy, data mixing, WER expectations, validation during training, checkpoints, pseudo-labeling, mobile latency, ablation, ONNX export, gradient accumulation, evaluation metrics, tokenizer vocab size, implementation difficulty, batch sizes, transfer learning, decoder streaming mechanics, punctuation/capitalization, noise robustness, code-switching, v2.1 layer count, weight initialization, label smoothing, data versioning, non-streaming baseline.

---

## Critical Gaps

### 1. No decode strategy — beam search, LM rescoring, or shallow fusion

This is the biggest miss across all reviews. The plan discusses the model producing tokens but never describes **how tokens are decoded at inference**. The implied assumption is greedy decoding (argmax at each step), but this leaves significant accuracy on the table.

Standard AED decoding options:

| Strategy | WER Impact | Latency Impact | Complexity |
|----------|-----------|----------------|------------|
| Greedy (argmax) | Baseline | Fastest | Zero |
| Beam search (width 5-10) | +5-10% relative WER improvement | 5-10x slower decoder | Low |
| Beam search + shallow LM fusion | +10-15% relative WER improvement | 5-10x slower decoder + LM lookup | Medium |
| N-best rescoring with neural LM | +12-20% relative WER improvement | 2-pass (slow) | High |

For Russian specifically, an external language model helps because:
- Russian has rich inflection (multiple valid word forms sound similar)
- Case endings are phonetically reduced — the acoustic signal is ambiguous
- A Russian N-gram or KenLM trained on Wikipedia/news resolves many ambiguities

**The plan already trains a KenLM for pseudo-label filtering (Section 6.4).** The same LM can be used for shallow fusion during decoding — zero additional training cost.

**Fix**: Add a "Decoding Strategy" subsection:
1. Phase 1: Greedy decoding (fast iteration, baseline WER)
2. Phase 2: Beam search (width 5-8) + shallow LM fusion with the same KenLM from pseudo-labeling
3. Report both greedy and beam+LM WER — quantifies the decoding strategy's contribution
4. For streaming: beam search adds decoder latency. Measure whether it fits within the latency budget. If not, use greedy for streaming, beam+LM for non-streaming

---

### 2. No regularization strategy beyond label smoothing

The plan trains on 5.4K hours — 55x less data than English Moonshine (300K hours). English Moonshine likely relied on massive data diversity for regularization. With limited data, explicit regularization becomes critical.

The plan mentions only label smoothing (ε=0.1). Missing:

| Technique | What It Does | Expected Impact |
|-----------|-------------|----------------|
| Dropout (attention + FFN) | Prevents co-adaptation of features | +1-3% relative WER, standard in low-resource ASR |
| Attention dropout | Prevents attention overfitting to specific positions | +1-2% relative WER |
| Stochastic depth (drop path) | Randomly skips encoder layers during training | Regularizes deep encoders, improves generalization |
| Zoneout (for recurrent components) | Preserves partial hidden state across steps | Less relevant for pure Transformer |

GigaAM-v3 uses dropout throughout. Conformer models standardly use attention dropout 0.1 + FFN dropout 0.1 + drop path 0.1. With 5.4K hours, omitting these risks overfitting to training speakers and acoustic conditions.

**Fix**: Add to training config:
- Attention dropout: 0.1
- FFN dropout: 0.1
- Drop path / stochastic depth: 0.1 (v2.1 only, with deeper encoder)
- These are standard hyperparameters. Tune via Phase 1 validation WER.

---

### 3. v2.1 stride-2 downsample/upsample operations are undefined

The plan says "stride-2 downsample" and "upsampling + skip connection" but never specifies **what operation**:

| Option | Causal? | Learnable? | Streaming-safe? |
|--------|---------|------------|-----------------|
| Learned conv stride-2 | Yes (with causal padding) | Yes | Yes |
| Average pooling stride-2 | No (looks at future frames) | No | **No** |
| Learnable pooling (strided linear) | Yes | Yes | Yes |
| Nearest-neighbor upsample + conv | Yes | Yes | Yes |
| Transpose conv (stride-2) | Yes (with causal padding) | Yes | Yes |
| Repeat + learned blend | Yes | Yes | Yes |

This is not a minor detail — choosing a non-causal downsampling operation silently breaks streaming. The plan's streaming latency claims depend on every operation being causal.

**Fix**: Specify explicitly:
- **Downsample**: Causal learned conv stride-2 (kernel=2, stride=2, padding=left-only)
- **Upsample**: Nearest-neighbor repeat + learned conv (1x1 or small kernel)
- **Skip connection**: Element-wise addition of Stage 1 output (at 50Hz) to Stage 3 output (after upsampling to 50Hz)
- All operations must be verified causal in PoC test T9

---

### 4. No distributed training or multi-GPU strategy

Phase 2 assumes a single H100. But:
- Small (123M) with batch 16-24 at sequence length 1500 on a single H100 may hit memory limits with optimizer states (AdamW/Schedule-Free stores 2x model params)
- Medium (245M) in Phase 3 almost certainly requires multi-GPU
- The plan never mentions DDP, FSDP, DeepSpeed, or gradient accumulation implementation

Even if single-H100 works for Small, having a distributed strategy prevents re-engineering when Medium is needed.

**Fix**: Add brief note:
- Phase 1 (3090): Single GPU, gradient accumulation to effective batch 128
- Phase 2 (H100): Start with single GPU. If OOM, use PyTorch FSDP with gradient checkpointing. No code change needed beyond wrapping the model.
- Phase 3 (Medium): FSDP required. Plan for 2x H100 or use gradient checkpointing on 1x H100.

---

### 5. No error analysis framework

The plan measures WER but doesn't categorize **what types of errors the model makes**. For a project with iterative improvement (Phase 1 → 2 → 3), understanding error patterns guides where to invest effort.

For Russian, error categories that matter:

| Error Type | Example | Fix |
|-----------|---------|-----|
| Consonant cluster errors | "вств" → "ст" | More data with clusters, or causal conv (v2.1) |
| Vowel reduction errors | unstressed "о" → "а" predicted as "о" | More natural speech data (less read speech) |
| Proper noun errors | Names, places, brands | More diverse training text, or post-processing with entity list |
| Number formatting | "2024" vs "две тысячи двадцать четыре" | Better text normalization |
| Code-switching | "email" garbled | English loanwords in tokenizer |
| Homophone confusion | "код" vs "кот" (context-dependent) | LM rescoring, more data |
| Boundary errors | Missing/extra words at chunk edges | Fix streaming chunking, SSC context |
| Morphological errors | Wrong case ending | LM rescoring, more data |

**Fix**: After each phase, run error analysis on 500+ validation samples:
1. Compute per-category error rates
2. Identify the top-3 error categories by frequency
3. Target next phase improvements toward those categories
4. This is a one-time script (word-level alignment + rule-based categorization for Russian)

---

### 6. No inference benchmarking methodology

The plan states latency targets ("< 100ms TTFT on MacBook CPU") but doesn't define:
- Which MacBook? M1? M2? M3? M4? Pro? Air? Max?
- Which CPU core type? Performance or efficiency?
- ONNX Runtime version? Which execution provider? (CPU, CoreML, NNAPI)
- Warm or cold start? First run or averaged over 100 runs?
- Audio length for the measurement? (TTFT should be constant for streaming, but verify)
- Concurrent load? (Is anything else running?)

Without this, latency claims are not reproducible and cannot be compared across models or with competitors.

**Fix**: Define a benchmarking protocol:
```
Hardware: MacBook Pro M2 (specific model), 16GB RAM
Runtime: ONNX Runtime 1.x, CPU execution provider
Model: Small, INT8 dynamic quantization
Measurement: Average TTFT over 100 utterances, 5-15s each
Precondition: 10 warmup runs, then measure
Audio: Common Voice test split (reproducible)
Environment: No other CPU-intensive processes
```

Apply the same methodology to each target platform (iPhone, Android, Pi).

---

## Medium-Priority Gaps

### 7. No curriculum learning for low-resource regime

With 5.4K hours (vs 300K for English Moonshine), training convergence is harder. The plan dumps all data at once. A curriculum could help:

**Phase-internal curriculum** (within Phase 2):
1. Epochs 1-2: Train on cleaner subsets (Common Voice validated + MLS) — establish baseline acoustic patterns
2. Epochs 3+: Mix in noisier data (Golos far-field, SOVA) — robustness

This is standard in low-resource ASR (ESPnet recipes do this by default). It costs nothing — just dataset ordering.

**Fix**: Add optional curriculum note to Phase 2 training config. Evaluate: does curriculum training converge faster or to better WER than random shuffling? If no difference, drop it.

---

### 8. Speaker overlap enforcement is vague

The plan says "no speaker overlap" between train/validation/test. But:
- Common Voice provides speaker IDs — easy to split by speaker
- Golos: speaker IDs may or may not be available
- MLS: chapter/book metadata exists but not always speaker IDs
- SOVA, RuLS: speaker metadata varies

Cross-dataset speaker deduplication is essentially impossible (different speaker ID namespaces). If the same person contributed to both Common Voice and Golos, they'll appear in both train and test with different IDs.

**Fix**: Best-effort approach:
1. Within each dataset: split by speaker ID (available for CV, MLS)
2. Across datasets: accept minor overlap as noise — unlikely to significantly affect results
3. Explicitly state this limitation in the evaluation section

---

### 9. No experiment tracking or logging infrastructure

The plan runs multiple phases, multiple model variants (v2, v2.1), ablations (A-E), hyperparameter searches, and pseudo-labeling iterations. Without experiment tracking:

- Cannot compare runs across phases
- Cannot reproduce results ("which checkpoint had the best WER?")
- Cannot debug training failures retroactively

**Fix**: Specify a lightweight experiment tracking tool:
- **Weights & Biases** (wandb): Free for individual use, minimal setup
- **TensorBoard**: Built into PyTorch, zero dependency
- **MLflow**: Self-hosted, good for experiment comparison

Log: loss, WER, learning rate, gradient norms, epoch, step count, per-dataset WER, hyperparameters, Git commit hash. This is ~10 lines of code.

---

### 10. Missing text normalization edge cases

The plan discusses number normalization (Section 5) but Russian has other normalization challenges:

| Category | Example | Challenge |
|----------|---------|-----------|
| Abbreviations | "США", "МГУ", "ФСБ" | Should be spelled out or kept as-is? Depends on how speakers say them |
| Hyphenated words | "какой-то", "из-за" | Treat as one token or two? Affects WER calculation |
| Mixed scripts | "iPhone 15 Pro" | Code-switching, Latin characters in Russian speech |
| URLs/emails | "напиши на info@..." | Rare in read speech, common in dictation |
| Foreign proper nouns | "Шекспир", "Вашингтон" | Transliterated, may have multiple valid spellings |
| Dates | "1 мая", "23 февраля" | Gendered and declined number forms |
| Ordinal vs cardinal | "первый" vs "один" | Context-dependent, both valid depending on speech style |

**Fix**: Add to text normalization pipeline:
- Abbreviation expansion: keep a dictionary of ~200 common Russian abbreviations with their spoken forms
- Hyphenated words: keep as single tokens (standard for Russian NLP)
- Unified date/time normalization rules
- Add edge-case test to T12 (tokenizer roundtrip): include abbreviations, dates, hyphenated words, mixed script

---

### 11. BPE subword regularization

SentencePiece supports sampling multiple valid BPE segmentations for the same text (`nbest_size` parameter). During training, different segmentations are sampled per epoch, making the model robust to tokenization boundary ambiguity.

This is especially valuable for Russian where word-internal morpheme boundaries are ambiguous (e.g., "переписывать" could be segmented as "пере/пис/ывать" or "перепис/ывать").

**Fix**: Enable during training: `sp.SetEncodeExtraOptions("bos:eos")` and use `nbest_size=5` with `alpha=0.1` for stochastic subword sampling. At inference, use deterministic encoding (nbest_size=1). This is a one-line parameter change.

---

### 12. Attention sink handling

The STREAMING_ASR_REVIEW.md (Section 5.3) notes that streaming models exhibit "attention sinks" — disproportionate attention to initial frames regardless of content (XLSR-Transducer discovery). The plan doesn't address this.

Moonshine's ergodic encoder (no positional embeddings) may be less susceptible, but this hasn't been validated for Russian.

**Fix**: Add a low-cost mitigation: during training, occasionally (10% of batches) prepend a learnable "sink token" frame to the encoder input. At inference, this token absorbs excess attention. Alternatively, just monitor attention patterns in Phase 1 — if sinks appear, add the mitigation.

---

### 13. FLOPs budget analysis missing

The plan gives parameter counts (34M, 123M, 245M) but never estimates FLOPs. For edge deployment:

- **Params** determine model size (storage, memory)
- **FLOPs** determine inference speed (computation)

Two models with the same parameter count can have very different FLOPs:
- v2 encoder at 50Hz: attention is O(T × w) where w=16 (small)
- v2.1 encoder with multi-scale: Stage 2 at 25Hz has half the frames → fewer FLOPs

The v2.1 multi-scale should actually be **faster** than v2 despite having slightly more params, because Stage 2 operates on half the frames. But the plan doesn't quantify this.

**Fix**: Estimate FLOPs for each variant:
```
v2 Small encoder: 10 layers × 50Hz × [attention O(T×w) + FFN O(T×d²)] per layer
v2.1 Small encoder: 3 layers @ 50Hz + 4 layers @ 25Hz + 3 layers @ 50Hz
```
This directly informs deployment decisions — v2.1 may be both more accurate AND faster.

---

## Minor Gaps

### 14. Knowledge distillation (Small → Tiny) not planned

The STREAMING_ASR_REVIEW mentions distillation, and the plan has speculative decoding (Tiny drafts for Small). But there's no plan to distill Small knowledge into Tiny. A Tiny model trained via distillation from a strong Small model can approach Small-level accuracy at Tiny-level cost.

This matters because Tiny is the phone-deployment model. Better Tiny = better user experience.

**Fix**: Add as optional Phase 4 activity:
- Teacher: Small model (v2 or v2.1, whichever is better)
- Student: Tiny model (same track)
- Method: KL divergence on logits + CTC loss + hidden-state matching
- Expected outcome: Tiny WER within 1-2% of Small (vs 3-5% gap from training Tiny directly)

---

### 15. No BiasNorm consideration for v2.1

Zipformer replaces LayerNorm with BiasNorm specifically to avoid issues with streaming normalization (batch statistics change between training and streaming inference). The v2.1 multi-scale architecture borrows from Zipformer but doesn't adopt this change.

**Fix**: Low priority — try it in ablation (Phase 1b). If it helps, adopt for v2.1 Small. If not, keep LayerNorm for consistency with v2.

---

### 16. Continuous integration / reproducibility

The plan discusses data versioning but not:
- Code versioning strategy (tagging releases per phase?)
- Environment pinning (requirements.txt, Dockerfile, CUDA version)
- Model card / documentation for released weights
- Reproducible training scripts (`train.py --config phase2_small.yaml`)

For a multi-week project with cloud GPU spend, a bad environment can waste expensive GPU time.

**Fix**: Add brief reproducibility note:
- Pin all dependencies (PyTorch version, CUDA, Python, pip packages)
- Use a Dockerfile for cloud GPU runs
- Each training run gets a unique ID + config file + git commit hash
- Released weights include a model card with training details

---

### 17. No streaming failure mode analysis

What happens when:
- **VAD fails**: Misses speech onset → truncated first word. False positive → spurious decoder invocation → garbage tokens.
- **Audio truncation**: User starts speaking, audio buffer overflows → lost frames → incomplete utterance.
- **Very fast speech**: Speaking rate > 200 WPM → more tokens per segment → decoder latency exceeds budget.
- **Very slow speech**: Long pauses within utterance → VAD splits mid-sentence → loss of cross-utterance context.
- **Background speech**: Other speakers in the room → VAD triggers on wrong speaker → wrong language/model input.

None of these are discussed. The plan assumes clean VAD and clean audio pipeline.

**Fix**: Add a "Streaming Failure Modes" subsection acknowledging these and specifying mitigations:
- VAD failure: tune Silero VAD threshold, add minimum speech duration filter (200ms)
- Fast speech: increase decoder trigger frame count (N=64 → N=96)
- Background speech: Speaker verification before ASR (out of scope, but document as known limitation)

---

### 18. No evaluation on accented speech or dialectal variation

Russian spoken across the post-Soviet space varies significantly:
- Moscow vs. St. Petersburg pronunciation differences
- Southern Russian dialects (fricative "г")
- Ukrainian-accented Russian (common, millions of speakers)
- Central Asian Russian (Kazakhstan, Uzbekistan)
- Caucasus-accented Russian

Common Voice and Golos primarily contain metropolitan Russian. The model may perform poorly on accented/dialectal speech — the exact scenarios where edge ASR is most needed (users in diverse regions).

**Fix**: Add accented/dialectal test samples to evaluation:
1. Record or collect 50-100 samples from diverse Russian speakers
2. Report WER by accent/dialect group
3. If significant gaps exist, target augmentation in Phase 3 (speed perturbation partially helps, but accent-specific data is better)

---

### 19. Missing: Streaming chunk overlap/deduplication

Section 3.4 mentions "Deduplication handles overlap between consecutive segments" but doesn't specify how. In streaming, consecutive decoder invocations may produce overlapping text (e.g., last 2 words of segment 1 overlap with first 2 words of segment 2). The deduplication strategy affects both accuracy and user experience.

Options:
- **Simple overlap**: Keep last N words from previous segment, discard matching prefix of current segment
- **Attention-based**: Decoder cross-attends to last K encoder frames from previous segment
- **No overlap**: Decoder only sees new encoder frames — simpler but less accurate at boundaries

**Fix**: Specify the overlap strategy. Recommendation: simple word-level overlap (keep last 2 words from previous decoder output, compare with current output, discard duplicates). This is O(N) and handles the common case.

---

### 20. No early stopping patience defined

Section 9 mentions "early stopping if WER stops improving" but doesn't define patience. Too short → premature stop. Too long → wasted compute on cloud GPU.

**Fix**: Specify: "Early stopping patience = 3 consecutive WER evaluations with no improvement (6K steps at 2K eval interval). Resume from best checkpoint."

---

## Summary

### Priority Ranking

| # | Gap | Priority | Impact if Unaddressed |
|---|-----|----------|----------------------|
| 1 | No decode strategy (beam search + LM) | **Critical** | Leaving 10-15% WER improvement on the table for free |
| 2 | No regularization (dropout, etc.) | **High** | Overfitting on 5.4K hours, inflated validation WER |
| 3 | v2.1 stride-2 operations undefined | **High** | Silent streaming breakage if non-causal ops chosen |
| 4 | No distributed training plan | Medium | OOM on Medium, re-engineering cost |
| 5 | No error analysis framework | Medium | Cannot guide iterative improvement effectively |
| 6 | No benchmarking methodology | Medium | Irreproducible latency claims |
| 7 | No curriculum learning | Low-Medium | Suboptimal convergence on limited data |
| 8 | Speaker overlap enforcement | Low | Minor data leakage, unlikely to affect results |
| 9 | No experiment tracking | Low | Operational inconvenience, not an ML gap |
| 10 | Text normalization edge cases | Low | Noisy training signal for abbreviations, dates |
| 11 | BPE subword regularization | Low | Minor robustness improvement |
| 12 | Attention sink handling | Low | May not affect Moonshine (ergodic encoder) |
| 13 | FLOPs budget analysis | Low | Deployment decision gap |
| 14 | Knowledge distillation | Low | Optional Phase 4 improvement |
| 15 | BiasNorm for v2.1 | Low | Minor architectural refinement |
| 16 | CI/reproducibility | Low | Engineering best practice |
| 17 | Streaming failure modes | Low | Edge cases in production |
| 18 | Dialectal evaluation | Low | Coverage gap |
| 19 | Streaming deduplication strategy | Low-Medium | Affects streaming output quality |
| 20 | Early stopping patience | Low | Tunable, not a design gap |

### Top 3 Recommendations

1. **Add beam search + LM rescoring** — the plan already trains a KenLM for pseudo-labeling. Reusing it for decoding gives 10-15% relative WER improvement at zero training cost. This is the single highest-impact, lowest-effort improvement not in the plan.

2. **Add dropout regularization** — 5.4K hours is low-resource for a 123M parameter model. Attention dropout 0.1 + FFN dropout 0.1 is standard practice and prevents overfitting. Without it, the gap between validation and test WER may be large.

3. **Specify v2.1 stride-2 operations** — the multi-scale architecture is described at a high level but the downsample/upsample operations are not defined. Choosing the wrong operation (e.g., average pooling) silently breaks streaming. This is a correctness issue, not an optimization.

### What the Plan Gets Right

- **PoC test suite** (T1-T18) is excellent — catches bugs before expensive cloud training
- **Two-track strategy** is pragmatic — v2 as safety net, v2.1 for improvement
- **Tiered WER targets** are honest about difficulty
- **Shared components** across tracks minimize duplicated engineering
- **Cache-aware inference** is the right optimization for streaming encoder
- **Streaming decoder mechanics** (Section 3.4) fill the gap identified in CRITIC_V2
