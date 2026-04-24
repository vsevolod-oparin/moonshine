# Fourth-Pass Critical Evaluation of ru-Moonshine Plan

Gaps not covered by MOONSHINE_DS_CRITIC.md (V1), MOONSHINE_CRITIC_V2.md (V2), MOONSHINE_CRITIC_V3.md (V3), or any associated response/comment documents.

Previous reviews covered 40+ topics including: test set strategy, data mixing, WER targets, validation, checkpoints, pseudo-labeling, mobile latency, ablation, ONNX export, gradient accumulation, evaluation metrics, tokenizer, transfer learning, decoder mechanics, punctuation, noise robustness, code-switching, weight init, label smoothing, data versioning, non-streaming baseline, decode strategy, regularization, v2.1 stride-2 ops, distributed training, error analysis, benchmarking, curriculum learning, experiment tracking, text normalization, BPE subword reg, attention sinks, FLOPs, distillation, streaming failure modes, dialectal coverage, streaming dedup.

---

## Critical Gaps

### 1. v2.1 Tiny multi-scale architecture is underspecified

The plan specifies v2.1 Small as 10 layers split 3+4+3 across three stages. But Phase 1 trains **Tiny (6 encoder layers)** for both v2 and v2.1. How does a 3-stage U-Net split work with only 6 layers?

Options:
- **2+2+2**: 2 layers per stage — each stage has barely any depth. Stage 2 at 25Hz with 2 layers may not learn meaningful coarse features
- **1+3+2**: Asymmetric — stage 2 gets 3 layers, stages 1 and 3 get minimal processing
- **2+3+1**: More weight on coarse features, minimal refinement
- **Skip multi-scale for Tiny**: Use v2.1 conv + SSC but not U-Net. Only add multi-scale for Small (10 layers) and Medium (14 layers)

This is not a minor detail — Phase 1 validates whether v2.1 improves over v2. If the Tiny v2.1 architecture is a bad split, the validation result will be misleading (v2.1 appears worse than it would be at Small scale).

**Fix**: Specify Tiny v2.1 split explicitly. Recommendation: **2+2+2 with reduced downsample effect** — the multi-scale is less impactful at Tiny scale, which is fine. Tiny is for validation, not production. Or: skip multi-scale for Tiny entirely, add only causal conv + SSC. Document which approach is used.

---

### 2. No variable-length batching strategy

Audio clips range from 1s to 30s (50 to 1500 encoder frames). Naive batching pads all sequences to the longest in the batch. If one 30s clip lands in a batch of 5s clips, 80% of the compute is wasted on padding.

This directly affects GPU utilization and training throughput. With 5.4K hours of variable-length audio, this matters significantly.

Standard approaches:

| Strategy | How | Trade-off |
|----------|-----|-----------|
| Bucket by length | Sort clips into buckets (e.g., 1-5s, 5-10s, 10-15s, 15-30s). Batch within buckets | Minimal padding waste. Requires sorting. Standard in production ASR |
| Dynamic batching | Group clips to minimize total padding per batch | Optimal but complex. Used in Seq2Seq training |
| Random + max_len cap | Truncate/pad to a max length per batch (e.g., 15s) | Simple. Wastes some data (truncated clips) |
| Pack sequences | Pack multiple short clips into one sequence up to max_len | Optimal compute. Complex data loading. Used in LLM training |

**Fix**: Add to training config:
- **Phase 1 (Tiny, 700h)**: Bucket by length into 4 bins. No truncation.
- **Phase 2 (Small, 5.4Kh)**: Bucket by length into 6 bins. Max batch duration per GPU = `batch_size × max_clip_in_bucket` must fit in VRAM.
- Report effective padding ratio (should be < 15%).

---

### 3. Data loader performance not addressed

ASR training is often data-loader-bound, not compute-bound. GPU utilization should be >85% but often drops to 50-70% when the CPU can't load and augment audio fast enough.

For 5.4K hours of audio:
- Total raw audio size: ~5,400h × 3600s/h × 16kHz × 2 bytes ≈ 620GB of WAV files
- Loading audio from disk + resampling + augmentation every batch

The plan doesn't discuss:
- How many DataLoader workers? (Typically: 4-8 per GPU)
- Pre-extract features to disk (avoid loading raw audio every epoch)?
- Store audio as optimized format (WebDataset, HDF5, LMDB) instead of individual WAV files?
- Pin GPU memory for faster transfer?
- `prefetch_factor` for DataLoader?

**Fix**: Add brief data loader config to training infrastructure:
- **Workers**: 8 per GPU. If CPU-bound, pre-extract mel features to disk as `.pt` tensors
- **Audio storage**: Individual WAV files are fine for Phase 1 (700h). For Phase 2 (5.4Kh), consider WebDataset (tar archives of audio+transcript pairs) for faster sequential reads
- **Prefetch**: `prefetch_factor=2` in DataLoader
- **Monitor**: GPU utilization should be >85%. If <70%, data loading is the bottleneck — increase workers or pre-extract features

---

### 4. No explicit go/no-go threshold between Phase 1 and Phase 2

Phase 1 produces Tiny WER results. Phase 2 spends $250-430 on cloud H100. But the plan never specifies: **what Tiny WER justifies spending cloud money?**

If Tiny v2 WER is 35%, something is fundamentally wrong — spending money on Small won't fix it. If Tiny WER is 20%, that's promising for Small to hit 12-15%. Where's the line?

**Fix**: Add explicit Phase 1 exit criteria:

| Tiny WER (greedy, validation) | Decision |
|-------------------------------|----------|
| < 20% | Proceed to Phase 2. Architecture and pipeline validated |
| 20-25% | Proceed cautiously. Investigate error patterns. Consider transfer learning before scaling |
| 25-30% | Stop. Debug: check data quality, loss convergence, tokenizer coverage. Try transfer learning on Tiny before spending cloud budget |
| > 30% | Do not proceed. Fundamental issue. Re-examine architecture, data pipeline, or training loop |

This prevents wasting cloud budget on a broken pipeline.

---

### 5. KenLM training and deployment details missing

The plan mentions KenLM twice: for pseudo-label filtering (Section 6.4) and shallow fusion decoding (Section 7.6). But never specifies how to build it.

This matters because:
- LM order (3-gram vs 5-gram) affects both WER improvement and model size
- A 5-gram KenLM on Russian Wikipedia is ~2-5GB on disk — doesn't fit on a phone
- A pruned 3-gram is ~50-200MB — fits on any device
- The LM needs to match the tokenizer's vocabulary (Russian BPE, not characters)

**Fix**: Add KenLM specification:

**Training**:
1. Download Russian Wikipedia dump
2. Strip markup, lowercase, apply same text normalization as training data (numbers, abbreviations)
3. Tokenize with the same SentencePiece model used for ASR
4. Train KenLM: `lmplz -o 4 -S 8G --prune 0 1 2 3 < tokenized_text > model.arpa`
   - 4-gram (good balance of accuracy and size)
   - Prune singleton 2-grams, 3-grams, and 4-grams to control size
5. Binary build: `build_binary model.arpa model.bin`
6. Expected size: ~100-200MB (pruned 4-gram with 512 BPE vocab)
7. For phone deployment: further prune to ~50MB (3-gram, aggressive pruning)

---

### 6. CTC blank token must be suppressed during AED decoding

The vocabulary includes `<blank>` for CTC training. But the AED decoder should never produce it — blanks are a CTC-specific concept (frame-level silence/noise). If the blank token is not suppressed during AED decoding, the decoder may occasionally output it, producing garbage like "при<blank>вет".

This is a one-line fix but a silent correctness bug if missed:

```python
# During AED decoding, suppress blank token
logits[:, blank_token_id] = float('-inf')
```

**Fix**: Add to Section 7.6 (Decoding Strategy): "Suppress CTC `<blank>` token during AED decoding by setting its logit to -inf. The blank token exists only for CTC frame-level alignment, not for autoregressive text generation."

---

### 7. Schedule-Free fallback not discussed

The plan commits fully to Schedule-Free with no alternative. If Schedule-Free produces poor results for Russian (unlikely but not impossible — it's validated on English Moonshine), there's no Plan B documented.

**Fix**: Add brief note: "Fallback: if Schedule-Free shows no convergence after 2K steps in Phase 1 T16, switch to AdamW with cosine LR schedule (LR 1e-3, warmup 1K steps, cosine decay to 1e-5). Standard, well-understood for ASR."

---

## Medium-Priority Gaps

### 8. Mixed precision: FP16 → BF16 switch unexplained

Phase 1 uses FP16 (3090), Phase 2 uses BF16 (H100). No explanation for this switch:

- **FP16** (3090): 10-bit mantissa, 5-bit exponent. Needs loss scaling (GradScaler) to avoid underflow
- **BF16** (H100): 8-bit mantissa, 8-bit exponent. Same dynamic range as FP32, no loss scaling needed
- H100 has native BF16 tensor cores — it's faster than FP16 on H100
- 3090 doesn't have native BF16 — it's emulated and slower than FP16

**Fix**: Add brief note to training config tables:
- Phase 1 (3090): FP16 with `torch.amp.GradScaler` — native on 3090, fastest option
- Phase 2 (H100): BF16 with `torch.amp.autocast('cuda', dtype=torch.bfloat16)` — native on H100, no loss scaling needed, wider dynamic range

---

### 9. Validation should use AED greedy, not just CTC greedy

The plan says "evaluate WER on validation set every 2K steps via CTC greedy decoding." CTC greedy is fast but measures encoder alignment quality, not the actual model output quality. The AED decoder is what produces final text.

CTC WER and AED WER can diverge — especially early in training when the decoder hasn't learned to cross-attend properly. A model with good CTC WER but bad AED WER has a decoder problem, not an encoder problem.

**Fix**: Log both:
- Every 2K steps: CTC greedy WER (fast, ~seconds)
- Every 10K steps (or at checkpoint evaluation): AED greedy WER (slower, ~minutes, but measures actual output quality)
- If CTC WER improves but AED WER plateaus, the issue is in the decoder or adapter, not the encoder

---

### 10. No data quality verification step

The plan has extensive preprocessing but no human verification that the output is correct. Common issues:
- Silence splitting creates mid-word cuts (despite the 300ms threshold)
- Text normalization produces wrong results ("2024" → "две тысячи двадцать четыре" but maybe "двадцать двадцать четыре" depending on context)
- Some Common Voice transcripts are wrong (crowd-sourced)
- Audio-transcript misalignment in SOVA/RuLS
- Corrupt audio files (zero-length, all-silence, clipping)

**Fix**: Add verification step after M3 (Data Pipeline):
- Listen to 50 random samples from processed data, verify transcript matches
- Check for: empty transcripts, transcripts shorter than 3 characters, audio-transcript duration mismatch (too many words for short audio or vice versa)
- Plot duration distribution — look for unexpected spikes or gaps
- Check for duplicate audio files across datasets (hash comparison)

---

### 11. Random seed specification

The plan mentions logging the random seed but doesn't specify what seed to use or set. For reproducibility:

```python
import random, numpy as np, torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

Also: `torch.backends.cudnn.deterministic = True` for full reproducibility (slightly slower, use for debugging only).

**Fix**: Add to training infrastructure: "Set fixed seed (default: 42) for all RNG sources at training start. Log seed in config. For production training, use deterministic=False (faster) but log seed anyway for approximate reproducibility."

---

### 12. Cloud GPU provider specifics for budget estimates

The plan estimates "$200-350" for Phase 2 cloud training but doesn't specify the provider. H100 pricing varies dramatically:

| Provider | H100 Price/hr | Phase 2 (6-8 days) | Notes |
|----------|--------------|-------------------|-------|
| Lambda Labs | ~$1.50-2.50 | ~$220-480 | Community cloud, shared infrastructure |
| RunPod | ~$2.00-3.00 | ~$290-580 | Easy setup, good UX |
| GCP A3 | ~$3.50-4.00 | ~$500-770 | Enterprise, reliable |
| AWS P5 | ~$4.00-5.00 | ~$580-960 | Most expensive |
| Vast.ai | ~$1.00-2.00 | ~$145-385 | Cheapest, least reliable |

The plan's "$200-350" estimate assumes ~$1.50-2.50/hr (Lambda/Vast range). This is achievable but worth specifying.

**Fix**: Add brief cloud cost note: "Budget assumes Lambda Labs or Vast.ai H100 (~$2/hr). GCP/AWS would double the cost. Use spot/preemptible instances when available."

---

## Minor Gaps

### 13. Gradient clipping not in training config tables

Mentioned in MILESTONES.md M5 (norm 5.0) but not in the main plan's Phase 1 or Phase 2 config tables. Should be explicit.

**Fix**: Add `Gradient clip: max_norm 5.0` to both config tables.

---

### 14. No English Moonshine sanity check

Before training Russian from scratch, verify the architecture code works by running English Moonshine inference on English test data. If the code can't produce reasonable English WER, the model definitions are wrong.

**Fix**: Add to Phase 0: "Sanity check: load English Moonshine v2 Tiny weights, run inference on 10 English audio clips, verify output is not garbage. This validates that model definitions match the original architecture."

---

### 15. Warmup for transfer learning underspecified

Section 6.7 says "LR warmup" for transfer learning but doesn't specify duration. Standard practice: warmup for 500-2000 steps (1-5% of total training). Too short → instability. Too long → waste.

**Fix**: Add to Section 6.7: "Transfer learning warmup: linear warmup over 1K steps for new layers (embedding, output projection), 500 steps for transferred layers (they start closer to good values)."

---

### 16. Augmentation during validation should be explicitly disabled

The plan doesn't state this. It's obvious to experienced practitioners but worth being explicit: SpecAugment, speed perturbation, MUSAN noise, and RIR augmentation are training-only. Validation uses clean audio with no augmentation.

**Fix**: Add to training config: `augmentation: {enabled: true, eval_enabled: false}`. One line in the config, prevents a common bug.

---

## Summary

### Priority Ranking

| # | Gap | Priority | Why |
|---|-----|----------|-----|
| 1 | v2.1 Tiny multi-scale underspecified | **High** | Phase 1 validates v2.1 on Tiny — if Tiny split is bad, validation is misleading |
| 2 | Variable-length batching | **High** | Directly affects training throughput and GPU utilization |
| 3 | Data loader performance | **High** | GPU <85% utilization = wasted time/money |
| 4 | Go/no-go threshold Phase 1→2 | **High** | Prevents wasting $250-430 on broken pipeline |
| 5 | KenLM training details | Medium | Affects deployment size and WER improvement |
| 6 | CTC blank suppression | Medium | Silent correctness bug if missed |
| 7 | Schedule-Free fallback | Medium | Risk mitigation for unproven optimizer on Russian |
| 8 | FP16→BF16 explanation | Low | Implicitly correct, just needs documenting |
| 9 | AED greedy for validation | Medium | CTC-only validation can miss decoder problems |
| 10 | Data quality verification | Medium | Garbage in, garbage out |
| 11 | Random seed | Low | Standard practice, one line |
| 12 | Cloud provider specifics | Low | Budget accuracy |
| 13 | Gradient clipping in config | Low | Cosmetic |
| 14 | English sanity check | Low | 10-minute validation |
| 15 | Transfer learning warmup | Low | Only needed if transfer learning is used |
| 16 | Augmentation disabled in eval | Low | Obvious, but explicit is better |

### What the Plan Still Gets Right After 3 Rounds of Review

The plan has been thoroughly reviewed and is now very comprehensive. The remaining gaps are mostly operational/engineering details rather than architectural or methodological issues. The core design — two-track architecture, CTC auxiliary loss, cache-aware streaming, PoC test gates, tiered WER targets — is sound.

### Top 3 New Recommendations

1. **Specify v2.1 Tiny multi-scale split** — Phase 1 trains Tiny v2.1. The 6-layer split needs to be defined. Without it, the ablation results are uninterpretable.

2. **Add go/no-go threshold** — A single table (Tiny WER → decision) prevents the most expensive mistake in the project: spending cloud money on a broken pipeline.

3. **Suppress CTC blank during AED decoding** — One-line fix, silent correctness bug. Easy to miss, produces garbage output if missed.
