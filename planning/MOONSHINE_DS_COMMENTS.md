# Moonshine Two-Track Plan: Comments & Reality Check

## **Critical Reality Check**

**The Moonshine v2 training pipeline is NOT open source.** The GitHub repo contains inference code only. You cannot "use existing tooling" for training v2 — you must implement the entire training pipeline from scratch based on the paper.

**Clarification**: When you say "existing tooling", you mean **inference tooling** (ONNX export, C++ runtime, iOS/Android bindings). This is correct — v2 uses Moonshine's existing inference ecosystem. v2.1 requires custom ONNX graphs but same runtime API.

## **Key Comments**

### 1. **Training Pipeline Reality**
- **You're not "using existing tooling" for training** — you're building it from scratch based on a 7-page paper
- **No training scripts, configs, or hyperparameters** are public
- **300K hours training pipeline** is a massive engineering project (not just running `train.py`)
- **Data loading, distributed training, checkpointing** — all must be built

### 2. **Architecture Implementation — Easier Than Expected**
- **Sliding-window attention**: Solved problem. HuggingFace has native `sliding_window` support (Mistral models), Flash Attention has `window_size` parameter, raw PyTorch needs ~5 lines for a band-diagonal mask. Not a gap.
- **`asinh` preprocessor**: Fully defined in public code (`moonshine/model.py` `AudioPreprocessor` class, also in HuggingFace Transformers). No mystery.
- **Schedule-Free optimizer**: Standard pip package (`pip install schedulefree`, v1.4.1, by Facebook Research, Apache 2.0). Drop-in replacement for AdamW. Only caveat: call `optimizer.train()`/`optimizer.eval()` alongside `model.train()`/`model.eval()`.
- **Full model architecture IS public**: PyTorch definitions in HuggingFace `transformers.models.moonshine`, Keras in `keras_hub.models`. You have the architecture code — what's missing is the training loop and data pipeline.
- **ONNX export** for streaming models with KV cache is the one genuinely complex item. Standard ONNX export is straightforward; cache-aware streaming export requires careful graph design.

### 3. **Data Scale Reality**
- **5.4K hours vs 300K hours** — Moonshine's accuracy comes from massive scale
- **Your 3090 validation** (700 hours, 3 epochs) won't tell you if the architecture works at scale
- **Phase 2 (5.4K hours)** is still 55× less data than Moonshine used

### 4. **v2.1 Architecture Comments**
- **Multi-scale U-Net + causal conv** adds significant complexity
- **SSC cross-window context** — clever but adds masking complexity
- **These changes require major PyTorch implementation work** beyond the already complex v2
- **v2.1 ONNX export** will be custom — can't use Moonshine's export scripts directly

### 5. **CTC Auxiliary Loss — Well-Established, Not Experimental**
- **CTC + attention joint training** has been standard practice since 2017 (ESPnet, NeMo, Wenet all implement it). It's well-understood and well-documented.
- **Fixed α = 0.2-0.3** works fine for most cases — no complex weight scheduling needed. The `λ = 0.8 → 0.3` schedule mentioned earlier was unnecessarily complicated.
- **Gradient clipping** (norm 5.0) handles stability issues. This is standard in any ASR training loop.
- **Implementation is trivial**: one `nn.Linear(enc_dim, vocab_size)` layer + `torch.nn.CTCLoss()`. Examples exist in every major ASR toolkit.

### 6. **Russian Tokenizer Vocabulary Size — 256/512 Is Fine, but Consider the Tradeoff**

**Accuracy is NOT the concern**. GigaAM-v3 achieves SOTA Russian WER (8.4%) with exactly 256 SentencePiece tokens. The GigaAM team explicitly tested vocab sizes 160, 256, 512, 1024, 2048 and found 160 and 256 showed identical results for CTC. T-one uses ~35 character tokens (just Russian letters) and achieves 8.6% WER. Russian morphology is rich but finite — BPE starting from 33 characters with ~190 merge rules covers common subwords well.

**The real tradeoff is decoder speed, not accuracy**:
- **Vocab size = number of distinct tokens** the model can produce (like an alphabet of subwords)
- Smaller vocab → more tokens per word → more autoregressive decoder steps → slower inference
- Larger vocab → fewer tokens per word → faster decoding, but bigger embedding matrix

| Vocab | Tokens/word (est.) | Decoder steps | Embedding params (Small, dim=512) |
|-------|--------------------|---------------|-----------------------------------|
| 35 (chars) | ~5 | Many | 18K |
| 256 | ~2-3 | Medium | 131K |
| 1,024 | ~1.5-2 | Fewer | 524K |
| 32,768 (Moonshine English) | ~1 | Fewest | 16.7M |

**Important**: Moonshine's English tokenizer uses **32,768 tokens** (HuggingFace config default). Your plan proposes 256-512, which is 60-120× smaller. This works for accuracy but means the decoder generates 2-3× more tokens per utterance, directly increasing streaming latency. For edge deployment where decoder speed matters, consider 1,024 as a middle ground — GigaAM uses this for their RNNT model specifically for speed reasons.

**Recommendation**: Start with 256 for Tiny validation (fast iteration). For Small production, test 256, 512, and 1,024 — compare WER and decoder latency. The answer depends on whether your bottleneck is encoder or decoder on target hardware.

### 7. **Training Resource Estimates — Batch Size Correction**

- **Tiny (34M)**: Batch 16-32 on 3090 is realistic in FP16. 34M params is small — weights FP16: 68MB, optimizer states: ~272MB, activations for batch 32 at 1500 frames: ~200-400MB. Total well under 24GB.
- **Small (123M)**: Batch 16-24 is optimistic. Weights FP16: 246MB, optimizer states: 984MB, activations for batch 16 at 1500 frames: ~400-800MB. **Realistic batch: 8-12 without gradient checkpointing, 12-16 with gradient checkpointing.**
- **First training runs will fail** — this is universally true for any ML project, not specific to this plan. Budget 1-2 failed runs.
- **Timeline**: The 2-3 day H100 estimate for full training is reasonable for the actual training run, but doesn't account for debugging time. Budget 1-2 extra days for failed runs and hyperparameter iteration.

### 8. **Inference Tooling Compatibility**- **v2**: Full compatibility with Moonshine inference ecosystem (C++ runtime, iOS/Android bindings)
- **v2.1**: Requires custom ONNX graphs but **same runtime API** — the C++/mobile code doesn't care about encoder internals
- **Risk**: Multi-scale encoder may need custom ONNX ops or subgraph partitioning

## **Recommendations (Without Changing Plan)**

### 1. **Phase 0 Reality Check**
- **First task**: Implement the training loop (data loading, loss, backprop). Architecture code already exists in HuggingFace `transformers.models.moonshine` — use it as the model definition.
- **Second task**: Train Russian SentencePiece BPE tokenizer, validate on morphology coverage
- **Third task**: Get a tiny model to overfit on 10 hours of data — this catches implementation bugs fast
- **Only then** proceed to Phase 1

### 2. **Tokenizer Strategy**
- **256 tokens is fine for accuracy** — GigaAM proves this for Russian
- **Test 256 vs 1,024 for Small production** — the tradeoff is decoder speed, not accuracy
- **Validate tokenizer** on Russian morphology: measure average tokens per word, check that common inflections (case endings, verb conjugations) are handled efficiently
- **Character-level (35 tokens)** is a valid alternative — T-one uses it successfully. Simpler, zero OOV, but 5× more decoder steps per word

### 3. **v2.1 Architecture Phasing**
- **Implement v2 first, get it working**
- **Then add ONE v2.1 improvement at a time**:
  1. Causal conv only (validate)
  2. Multi-scale only (validate)  
  3. SSC cross-window (validate)
- **Don't implement all three at once** — debugging will be impossible

### 4. **Training Pipeline Priority**
```
Priority 1: Basic training loop that converges (any WER) — use HF model definitions
Priority 2: Data pipeline (preprocessing, augmentation)
Priority 3: CTC auxiliary loss (one linear layer + CTCLoss, trivial)
Priority 4: Schedule-Free optimizer (pip install schedulefree)
Priority 5: Dynamic chunk training
Priority 6: Pseudo-labeling pipeline
```
**Don't implement all training tricks at once.** Get basic training working first.

### 5. **Inference Optimization Phasing**
```
Phase 1: Basic ONNX export (no caching, no quantization)
Phase 2: Add KV cache (cache-aware inference)
Phase 3: Add INT8 quantization  
Phase 4: Add speculative decoding (if needed)
```
**Basic inference working is more important than optimizations.**

### 6. **Realistic Timeline**
- **Phase 0 (setup)**: 2-3 weeks (implementing training pipeline from paper)
- **Phase 1 (validation)**: 1-2 weeks (debugging training, fixing implementation bugs)
- **Phase 2 (production)**: 3-4 weeks (not 2-3 days)

### 7. **Critical Path Items**
1. **Training loop + data pipeline** — the main engineering work (architecture code already exists)
2. **Russian BPE tokenizer quality** — test 256 vs 1,024 for decoder speed tradeoff
3. **ONNX export of streaming model with KV cache** — the one genuinely complex engineering item
4. **v2.1 multi-scale encoder** — validate each change independently, not all at once

## **Inference Tooling Compatibility Details**

### v2 Compatibility
```
Training (custom) → PyTorch model → Moonshine export scripts → ONNX graphs
                                                              ↓
                                                    Moonshine C++ runtime (existing)
                                                              ↓
                                                    iOS/Android/Python bindings (existing)
```

### v2.1 Compatibility
```
Training (custom) → PyTorch model → Custom export scripts → Custom ONNX graphs
                                                           ↓
                                                    Moonshine C++ runtime (existing, same API)
                                                           ↓  
                                                    iOS/Android/Python bindings (existing)
```

**Key insight**: The Moonshine runtime takes ONNX graphs as input. It doesn't care if the encoder has conv layers or multi-scale structure, as long as:
1. Input/output tensor shapes match expected format
2. The graph uses supported ONNX ops
3. Streaming semantics (KV cache updates) are preserved

**Risk**: Multi-scale encoder with downsampling/upsampling may require custom ONNX ops. Solution: Implement down/up as standard conv/transpose conv ops.

## **Bottom Line**

Your plan is **ambitious but feasible**. The implementation is easier than it initially appears because:
1. **Architecture code IS public** — HuggingFace Transformers has full PyTorch model definitions for Moonshine
2. **Key components have standard implementations** — sliding-window attention, Schedule-Free, SentencePiece BPE are all production-ready
3. **256 tokens works for Russian** — proven by GigaAM achieving SOTA with that exact size
4. **CTC+AED joint training is well-established** — not experimental, standard since 2017

The **main engineering work** is the training loop, data pipeline, and ONNX streaming export — not reimplementing the architecture from scratch.

**Start with v2, get it working, then layer v2.1 improvements one at a time.**