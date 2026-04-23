# Streaming ASR Architecture Review & Moonshine v2 Improvements

## 1. The Streaming ASR Landscape (2024-2026)

### 1.1 The Core Problem

Streaming ASR must balance three competing constraints:

| Constraint | Tension |
|------------|---------|
| **Accuracy** | Full attention > local attention for disambiguation |
| **Latency (TTFT)** | Streaming requires bounded computation per audio frame |
| **Edge deployment** | Memory < 1GB, compute < 1 TOPS on phones/IoT |

Full-attention encoders (Whisper, v1 Moonshine) achieve best accuracy but TTFT grows linearly with utterance length. Streaming architectures solve this by restricting how each frame attends to others.

### 1.2 Taxonomy of Streaming Approaches

```
Streaming ASR
├── Encoder-side streaming (bounded encoder computation)
│   ├── Sliding-window attention (Moonshine v2)
│   ├── Chunked attention (Zipformer, Conformer U2++)
│   ├── Cache-aware chunked inference (Nemotron)
│   └── State-space models (Mamba)
├── Decoder-side streaming (bounded decoder computation)
│   ├── CTC / RNN-T / TDT (token-by-token, monotonic)
│   ├── Chunked encoder-decoder (AED)
│   └── Speculative decoding (SpecASR)
└── Training-time streaming (learn to be streaming)
    ├── Dynamic chunk training
    ├── Causal masking strategies
    └── Latency regularization (FastEmit)
```

---

## 2. Architecture Developments

### 2.1 Sliding-Window Attention (Moonshine v2)

**What it is**: Each encoder frame attends to a fixed local window (left context + optional right context). No global attention.

**What it solves**: Quadratic O(T²) encoder complexity → linear O(T·w). TTFT becomes constant regardless of utterance length.

**Moonshine v2 specifics**:
- Window: (left=16, right=4) for first/last 2 layers, (left=16, right=0) for middle layers
- Right context = 80ms algorithmic lookahead (4 frames × 20ms)
- No positional embeddings → "ergodic" encoder (translation-invariant)
- Achieves 6.65% WER (Medium) vs 7.44% for Whisper Large v3 (1.5B params)

**Limitation**: Pure local attention loses long-range disambiguation. The model must resolve ambiguity from only ~320ms of context (16 frames × 20ms).

### 2.2 Chunked Attention (Zipformer, Conformer U2++)

**What it is**: Audio is partitioned into fixed-size chunks. Within each chunk, attention is full (bidirectional). Across chunks, only left-context carry-over is used.

**What it solves**: Balances local bidirectional accuracy with streaming latency. Unlike sliding-window, chunks get full attention within their boundary.

**Key variant — Zipformer** (NVIDIA, ICLR 2024):
- **Multi-scale U-Net**: Different frame rates at different encoder depths. Early layers operate at high frequency (fine temporal detail), deeper layers at lower frequency (coarse semantic features)
- **Non-Linear Attention (NLA)**: Reuses attention weights across modules, halving attention computation
- **BiasNorm**: Replaces LayerNorm (avoids issues with streaming normalization)
- **SwooshR/SwooshL activations**: Improved over Swish for speech
- **ScaledAdam optimizer**: Scale-invariant parameter updates
- Result: Zipformer-S (1/3 params of Conformer) achieves lower WER with >50% FLOPs savings

**Key variant — Unified Streaming/Non-Streaming Zipformer** (2025):
- Single model with **dynamic right-context masking**
- At inference, right-context is controllable → flexible latency-accuracy tradeoff
- 7.9% relative WER reduction with small latency increase
- Right-context is particularly effective due to multi-scale structure

**Applicability to ru-Moonshine**: HIGH. Multi-scale processing captures both fine phonetic detail and coarse prosodic patterns. NLA-style attention reuse could reduce Tiny/Small compute without changing model size.

### 2.3 Cache-Aware Streaming (Nemotron)

**What it is**: Encoder maintains cached states (attention KV, conv activations) across chunks. Only new audio is processed; cached context is reused.

**What it solves**: Eliminates redundant computation when processing overlapping or incremental audio. Standard chunked inference re-computes attention for the overlap region every chunk.

**Architecture**: Cache-Aware Fast Conformer (24 layers, 0.6B params) + RNNT decoder.
- Trained on 250K+ hours
- 8.20% avg streaming WER across 8 benchmarks
- After INT4 quantization: 0.67GB, faster than real-time on CPU

**Applicability to ru-Moonshine**: HIGH. Moonshine v2 already supports streaming via sliding-window, but doesn't explicitly cache encoder states. Adding cache-aware inference (store attention KV for window-1 frames, only compute attention for new frame) would reduce per-frame compute.

### 2.4 Sequential Sampling Chunks (SSCFormer)

**What it is**: Instead of fixed chunk boundaries, SSC dynamically re-partitions frames across chunks so frames can attend to adjacent-chunk context via attention mask manipulation (no extra computation).

**What it solves**: Cross-chunk context without increasing chunk size or adding carry-over modules. Same O(W) complexity, better accuracy.

**Results**: Consistently reduces CER across chunk sizes on AISHELL-1 vs. standard and shifted chunk schemes.

**Applicability to ru-Moonshine**: MEDIUM. The technique is architecture-agnostic. Could be applied to sliding-window attention to improve cross-window context without increasing window size.

### 2.5 Dynamic Chunking

**What it is**: A lightweight controller (MLP + attention) adjusts chunk width based on speaking rate and encoder state history.

**What it solves**: Fixed chunk sizes are suboptimal — slow speech needs less context, fast speech needs more. Dynamic chunks adapt.

**Results**: 6.23% WER vs. 7.91% for best fixed-chunk baseline on Tibetan ASR.

**Applicability to ru-Moonshine**: LOW-MEDIUM. Russian speech rate is relatively uniform. Fixed windows work well. Dynamic chunking adds complexity for marginal gain.

### 2.6 Conformer Convolution Module

**What it is**: Standard Conformer adds a convolution module (depthwise conv + pointwise conv + GLU) between self-attention and FFN in each layer. The conv module captures local patterns directly.

**What it solves**: Pure self-attention (as in Moonshine) must learn local patterns through attention weights. Convolution learns them more efficiently through parameterized local filters.

**Why Moonshine doesn't use it**: Convolution is less streaming-friendly than sliding-window attention alone. Causal convolutions add latency proportional to kernel size. Moonshine's design prioritizes minimal streaming overhead.

**Applicability to ru-Moonshine**: MEDIUM. Small causal convolutions (kernel 3-7) could be added to Moonshine encoder layers with minimal latency impact (~1-2 frames). May improve Russian phonetic discrimination (consonant clusters, vowel reduction).

---

## 3. Decoder Developments

### 3.1 Autoregressive Decoder (Moonshine v2 current)

Moonshine v2 uses a standard causal Transformer decoder with:
- RoPE positional embeddings
- Cross-attention to encoder outputs
- SwiGLU FFN blocks
- Autoregressive (token-by-token) generation

**Problem**: The decoder is the latency bottleneck for long transcripts. Even with instant encoder features, generating N tokens requires N sequential decoder steps.

### 3.2 Speculative Decoding for ASR (SpecASR, 2025)

**What it is**: A small "draft" ASR model generates candidate tokens in parallel. A larger "target" model verifies them. Accepted tokens skip the serial loop.

**Key insight for ASR**: ASR outputs are highly audio-conditioned, so small and large ASR models produce very similar outputs even when intermediate steps differ. This gives much higher acceptance rates than LLM speculative decoding.

**Results**: 3.04x-3.79x speedup over standard autoregressive decoding, zero accuracy loss.

**Applicability to ru-Moonshine**: HIGH. Train Tiny as draft model for Small/Medium. During streaming, Tiny generates candidates, Small verifies. Could reduce decoder latency by 3x with no accuracy cost. Requires maintaining two models in memory, but Tiny (34M) is negligible overhead next to Small (123M).

### 3.3 CTC / RNN-T / TDT Heads

**What it is**: Replace the autoregressive decoder with a frame-synchronous classification head.

| Decoder | Tokens/frame | Training | Streaming | Latency |
|---------|-------------|----------|-----------|---------|
| CTC | 0 or 1 | Alignment-free | Yes | Frame-level (20ms) |
| RNN-T | 0 or 1 per step | Joint network | Yes | Frame-level |
| TDT | token + duration | Joint prediction | Yes | Frame-level, faster |
| AED (Moonshine) | N tokens per step | Cross-attention | Chunk-level | Bounded by encoder |

**What they solve**: Frame-synchronous decoders emit tokens at audio frame rate (20ms), eliminating the serial decoding bottleneck entirely. No token-by-token loop.

**Trade-off**: CTC/RNN-T typically need larger encoders to match AED accuracy, because the decoder can't "re-read" encoder features. Moonshine's paper notes this explicitly: "Parakeet-class models follow this general direction and shift much of the modeling capacity into a larger encoder."

**Applicability to ru-Moonshine**: MEDIUM for initial version, HIGH for v2. Adding a CTC auxiliary loss during training is nearly free and improves alignment. A CTC head could also enable a "fast path" for streaming (CTC for provisional, AED for finalized).

### 3.4 Double Decoder

**What it is**: A temporary decoder processes only the current chunk's log-probabilities alongside the main decoder. The temporary decoder updates main decoder state without the full look-ahead.

**What it solves**: Achieves same WER as buffered decoding while reducing latency by the look-ahead size. Works on pre-trained models without retraining.

**Applicability to ru-Moonshine**: LOW. More relevant for RNN-T/CTC systems. Moonshine's AED decoder doesn't benefit from this pattern.

---

## 4. State-Space Models (Mamba)

### 4.1 What They Are

SSMs replace attention's O(N²) computation with O(N) state updates. The model maintains a hidden state that is updated incrementally by each input frame. "Selective" SSMs (Mamba) make the state update input-dependent, providing some content-based reasoning.

### 4.2 Results for ASR

- **Bidirectional Mamba variants** (ExtBiMamba) outperform unidirectional for speech
- **Still lag behind Conformer-based models** on ASR benchmarks as of 2025
- O(N) complexity is a significant theoretical advantage for streaming
- Hybrid SSM-attention models (Nemotron 3, Jamba) show promise but are primarily LLMs, not ASR encoders

### 4.3 Applicability to ru-Moonshine

**LOW for now.** Mamba for ASR is not yet production-ready. The accuracy gap vs. attention/Conformer is real. However, Mamba-2's Structured State Space Duality (SSD) creates a theoretical bridge between SSMs and attention, which could lead to hybrid architectures. Worth monitoring for 2027.

---

## 5. Self-Supervised Learning for Streaming

### 5.1 The Pattern

1. Pretrain encoder on massive unlabeled data (100K-4.5M hours) with masked prediction
2. Fine-tune for streaming with chunked masking
3. Add streaming-specific adaptations (causal normalization, chunked positional encoding)

### 5.2 Key Systems

| System | Pretraining Data | Key Innovation |
|--------|-----------------|----------------|
| wav2vec-S | From wav2vec 2.0 | Streaming normalization + chunked relative PE |
| XLSR-Transducer | XLSR-53 (multilingual) | Multi-chunk training, attention sink discovery |
| XEUS | 1M+ hours, 4057 languages | HuBERT + WavLM combined, SOTA on ML-SUPERB |
| SeamlessM4T w2v-BERT 2.0 | 4.5M hours | Conformer + contrastive + MLM, 100 languages |

### 5.3 Key Insight: Attention Sinks

XLSR-Transducer discovered that streaming ASR models disproportionately attend to initial frames regardless of content — the "attention sink" phenomenon previously seen in LLMs. This suggests streaming models need special handling of the first few frames.

### 5.4 Applicability to ru-Moonshine

**MEDIUM.** SSL pretraining could significantly improve Russian ASR accuracy, especially for:
- Accented speech and dialectal variation
- Noisy/telephony conditions
- Low-resource scenarios (we have ~5K hours vs. Moonshine's 300K)

**Challenge**: We don't have 300K hours of Russian labeled data. But we could:
1. Use XLSR-53 or MMS pretrained encoder as initialization
2. Fine-tune with Moonshine architecture using chunked masking
3. This would require modifying the Moonshine encoder to accept pretrained weights

**Cost**: Adds complexity to the pipeline. Not recommended for Phase 1 validation, but a strong candidate for Phase 2 accuracy improvement.

---

## 6. Efficient Inference Techniques

### 6.1 Quantization

**INT4 k-quant** (arXiv:2604.14493, April 2026):
- Nemotron 0.6B: 2.47GB → 0.67GB, <1% WER degradation
- Importance-weighted quantization > uniform quantization
- ONNX Runtime with operator fusion enables real-time CPU inference

**Applicability**: HIGH. Post-training quantization is free accuracy-wise and halves model size. Should be standard for deployment.

### 6.2 Attention Weight Reuse (NLA)

**What it is**: In Zipformer, Non-Linear Attention reuses computed attention weights across multiple modules in the same block.

**What it solves**: Halves attention computation without accuracy loss.

**Applicability**: MEDIUM. Could be applied to Moonshine's encoder blocks. Would require architecture changes.

### 6.3 Low-Rank Compression (LiteASR)

**What it is**: PCA-based decomposition of encoder linear transformations into low-rank matrix chains.

**What it solves**: 2-5x encoder compute reduction with minimal WER impact.

**Applicability**: MEDIUM. Could compress a trained Small model for edge deployment without retraining.

### 6.4 Knowledge Distillation from Whisper

**What it is**: Use large offline model (Whisper Large) to pseudo-label raw audio, then train streaming model on pseudo-labels.

**What it solves**: Eliminates the need for ground-truth transcriptions. Enables training on orders of magnitude more data.

**Results**: Surprisingly effective — pseudo-labeled data produces competitive streaming ASR without any supervised data. Adding supervised data as regularization helps for smaller models.

**Applicability to ru-Moonshine**: HIGH. We could:
1. Train Small on our 5.4K hours of labeled data
2. Use it to pseudo-label OpenSTT's 20K hours (CC-BY-NC)
3. Retrain on combined dataset
This would effectively give us 25K+ hours for the price of running inference.

---

## 7. Training Techniques for Streaming

### 7.1 Dynamic Chunk Training

During training, randomly sample chunk sizes per batch. Model learns to handle variable amounts of right context. Essential for streaming — decoding a non-streaming-trained model in streaming mode causes significant degradation.

**Applicability**: HIGH. Even Moonshine's fixed sliding window could benefit from training with variable window sizes, making the model robust to different latency budgets.

### 7.2 FastEmit (Latency Regularization)

Sequence-level emission regularization that penalizes late token emissions. Directly optimizes for low latency.

**Applicability**: LOW for AED architecture (latency is already bounded by encoder). HIGH if we add RNN-T/CTC decoder.

### 7.3 Multi-Task Training (CTC + Attention)

Joint training with CTC auxiliary loss:
- Improves encoder alignment quality
- Provides a CTC fast-path for streaming inference
- Regularizes training

**Applicability**: HIGH. Nearly free to add during training. Enables dual-mode deployment (CTC streaming + AED full accuracy).

---

## 8. Moonshine v2 Architecture Analysis

### 8.1 Current Architecture Summary

```
Audio (16kHz) → Preprocessor (CMVN + asinh + 2× stride-2 conv) → 50Hz features
    → Encoder (Transformer, sliding-window (16,4)/(16,0), no positional enc)
    → Adapter (adds positional embedding, linear projection)
    → Decoder (Transformer, RoPE, cross-attention, SwiGLU, autoregressive)
    → Text tokens
```

| Component | Params (Tiny/Small/Medium) | Notes |
|-----------|---------------------------|-------|
| Preprocessor | 2.08M / 7.74M / 11.86M | CMVN + asinh + 2 causal conv |
| Encoder | 7.39M / 43.49M / 93.66M | Sliding-window, no pos enc |
| Adapter | 1.31M / 2.86M / 3.64M | Adds positional embedding |
| Decoder | 22.80M / 69.27M / 135.77M | SwiGLU, RoPE, cross-attn |
| **Total** | **33.57M / 123.36M / 244.93M** | |

### 8.2 Key Design Choices

1. **No positional embeddings in encoder** — enables natural streaming (translation-invariant)
2. **asinh nonlinearity** — balances compression and dynamic range better than tanh/log-mel
3. **Sliding window (16,4)/(16,0)** — 320ms total context per frame, 80ms lookahead
4. **Positional embeddings only in decoder** — via Adapter layer
5. **SwiGLU in decoder only** — encoder uses standard FFN
6. **Schedule-Free optimizer** — no LR schedule needed
7. **Trained on 300K hours** — massive proprietary English dataset

### 8.3 Identified Weaknesses

| # | Weakness | Impact | Difficulty |
|---|----------|--------|------------|
| W1 | **No convolution in encoder** — pure attention must learn local patterns | Medium accuracy gap vs Conformer | Low |
| W2 | **No multi-scale processing** — all layers see same frame rate | Misses multi-resolution features | Medium |
| W3 | **No cache-aware inference** — redundant computation across chunks | Higher per-frame compute | Low |
| W4 | **Autoregressive decoder bottleneck** — serial token generation | High latency for long transcripts | Medium |
| W5 | **No CTC auxiliary loss** — misses alignment signal and fast-path | Lower accuracy + no frame-sync option | Low |
| W6 | **Fixed sliding window** — no cross-window context mechanism | Limited long-range disambiguation | Low |
| W7 | **No SSL pretraining** — trains from scratch, needs 300K hours | Data-hungry; less robust | High |
| W8 | **asinh frontend unvalidated** — authors note "informed guesses" | Potentially suboptimal features | Medium |

---

## 9. Recommended Improvements for ru-Moonshine

### 9.1 Priority Matrix

Improvements ranked by **impact × ease / cost**:

| Priority | Improvement | Expected Impact | Effort | Phase |
|----------|------------|----------------|--------|-------|
| **P0** | CTC auxiliary loss during training | +5-15% relative WER improvement | Low | 1 |
| **P1** | Cache-aware encoder inference | 2-3x lower per-frame compute | Low | 1 |
| **P1** | INT8/INT4 quantization for deployment | 2-4x smaller model, faster inference | Low | 1 |
| **P2** | Causal convolution in encoder layers | +3-8% relative WER improvement | Medium | 2 |
| **P2** | Multi-scale encoder (Zipformer-style) | +5-10% relative WER improvement | Medium | 2 |
| **P2** | Pseudo-labeling with Whisper/Large | Access to 20K+ extra hours | Medium | 2 |
| **P3** | Speculative decoding (Tiny drafts for Small) | 3x decoder speedup | Medium | 3 |
| **P3** | Dynamic chunk training | Robustness to variable latency | Low | 2 |
| **P3** | SSC-style cross-window context | Better cross-boundary accuracy | Medium | 2 |
| **P4** | SSL pretraining (XLSR/MMS init) | +10-20% relative WER improvement | High | 3 |
| **P4** | RNN-T / TDT decoder head | Frame-level streaming option | High | 3 |

### 9.2 P0: CTC Auxiliary Loss (Phase 1)

**What**: Add a linear CTC head on top of the encoder. Train with joint loss: `L = L_AED + α · L_CTC` where α ≈ 0.2-0.3.

**Why**: CTC forces the encoder to learn monotonic alignments between audio frames and text tokens. This:
- Improves encoder representations (better feature quality for the AED decoder)
- Provides a frame-synchronous "fast path" for ultra-low-latency streaming
- Is nearly free to implement (one linear layer + CTC loss)

**How**:
```python
# Add to encoder output
ctc_head = nn.Linear(enc_dim, vocab_size)
ctc_loss = CTCLoss()

# In training loop
encoder_out = model.encode(audio)
ctc_logits = ctc_head(encoder_out)
loss = aed_loss + 0.3 * ctc_loss(ctc_logits, text)
```

**Risk**: Minimal. CTC auxiliary loss is standard in modern ASR. If it doesn't help, α=0 removes it.

### 9.3 P1: Cache-Aware Inference (Phase 1)

**What**: Store encoder attention KV-cache and convolution states between frames. Only compute attention for new frames.

**Why**: Moonshine's sliding-window (left=16) means each new frame recomputes attention over 16 previous frames. With caching, only the new frame's contribution needs computing. This reduces per-frame encoder cost by ~8-16x.

**How**: At inference time:
1. Maintain a circular buffer of encoder KV-cache (16 frames deep per layer)
2. For each new audio frame: compute self-attention only for the new query against cached keys/values
3. Update cache: evict oldest frame, add new frame
4. Feed encoder output to decoder

**Risk**: Low. This is an inference-only optimization. No training changes.

### 9.4 P2: Causal Convolution in Encoder (Phase 2)

**What**: Add a lightweight causal depthwise convolution (kernel 3-7) to each encoder layer, following the Conformer pattern but keeping it streaming-safe.

**Why**: Russian has phonetic challenges that benefit from explicit local pattern modeling:
- Consonant clusters (вств, нтств, здн)
- Vowel reduction (unstressed о→а, е→и)
- Palatalization (soft vs. hard consonants)

Pure attention must learn these patterns implicitly through 16-frame windows. Convolution learns them directly with fewer parameters.

**How**: Modify encoder block from:
```
x → Self-Attention → FFN → output
```
to:
```
x → Self-Attention → CausalConv → FFN → output
```
Where CausalConv is depthwise conv (kernel=7, causal, groups=d_model) + GLU + pointwise conv.

**Latency impact**: Kernel 7 causal conv adds ~6 frames (120ms) of algorithmic delay. Can reduce to kernel 3 (40ms) for minimal impact.

**Risk**: Medium. Changes encoder architecture. Requires retraining. The Moonshine team likely omitted convolutions deliberately for streaming minimalism — need to validate that the accuracy gain justifies the added latency.

### 9.5 P2: Multi-Scale Encoder (Phase 2)

**What**: Adopt Zipformer's U-Net multi-scale structure. Early encoder layers operate at 50Hz (fine detail), middle layers at 25Hz, later layers at 12.5Hz (coarse semantic features).

**Why**: Different phonetic phenomena operate at different time scales:
- **50Hz** (20ms): Stop bursts, frication, voicing onset
- **25Hz** (40ms): Formant transitions, vowel identity
- **12.5Hz** (80ms): Prosodic patterns, word boundaries, stress

Moonshine processes all layers at 50Hz, which is wasteful for capturing slow-varying features.

**How**: Insert stride-2 downsampling between encoder blocks (similar to Zipformer's encoder stack). Later blocks operate on fewer frames → less attention computation.

**Expected benefit**: Both accuracy improvement (multi-scale features) AND compute reduction (fewer frames in deeper layers). This is the most impactful architectural change.

**Risk**: High. Major architecture change. Requires significant engineering. Worth prototyping on Tiny first.

### 9.6 P2: Pseudo-Labeling Pipeline (Phase 2)

**What**:
1. Train ru-Moonshine Small on 5.4K labeled hours
2. Run inference on OpenSTT 20K hours → generate pseudo-labels
3. Filter low-confidence pseudo-labels
4. Retrain on combined 25K+ hours

**Why**: The single biggest accuracy lever is more data. OpenSTT has 20K hours but is CC-BY-NC (non-commercial). Pseudo-labeling converts unlabeled audio into training data using our own model.

**Expected benefit**: +10-30% relative WER improvement depending on pseudo-label quality.

**Risk**: Low technical risk. Label quality depends on initial model quality → iterative process.

### 9.7 P3: Speculative Decoding (Phase 3)

**What**: Use Tiny (34M) as draft model for Small (123M) decoder. Tiny proposes 4-8 tokens in parallel, Small verifies in one forward pass.

**Why**: ASR outputs are highly audio-conditioned → small and large models agree on most tokens → high acceptance rate (>80%). 3-4x decoder speedup with zero accuracy loss.

**Memory cost**: +34M params (Tiny draft). Total: 157M. Still fits comfortably on any phone.

**Risk**: Medium. Requires careful implementation of draft-verify loop. ONNX export needs to handle two models.

---

## 10. Proposed Architecture: ru-Moonshine v2.1

For Phase 2 (production Small model), I recommend these modifications to the base Moonshine v2 architecture:

```
Audio (16kHz)
  → Preprocessor (CMVN + asinh + 2× stride-2 causal conv) → 50Hz features
  → Encoder (modified):
      ├── Block 1-2:  Sliding-window attention (16,4) + causal depthwise conv (k=7)
      ├── Block 3-4:  Sliding-window attention (16,0) + causal depthwise conv (k=5) + stride-2 downsample
      ├── Block 5-8:  Sliding-window attention (8,0) @ 25Hz + causal depthwise conv (k=5)
      ├── Block 9-10: Sliding-window attention (8,0) @ 25Hz + stride-2 upsample
      └── Block 11-12: Sliding-window attention (16,4) + causal depthwise conv (k=7)
      + CTC head (linear, for auxiliary training + fast-path streaming)
  → Adapter (positional embedding + linear projection)
  → Decoder (standard Moonshine v2: RoPE, cross-attention, SwiGLU)
  → Text tokens
```

**Key changes from vanilla Moonshine v2**:
1. Causal depthwise conv in each encoder block
2. U-Net multi-scale structure (50Hz → 25Hz → 50Hz)
3. CTC auxiliary head
4. Cache-aware inference (runtime optimization)
5. Trained with dynamic chunk sizes + CTC multi-task loss

**Expected outcome**:
- ~8-10% WER on Russian benchmarks (vs ~14% Vosk, ~8.4% GigaAM)
- < 100ms TTFT on MacBook CPU
- < 500MB ONNX model (INT8 quantized)
- 2-3x decoder speedup available via speculative decoding

---

## 11. References

### Papers
- Moonshine v2: arXiv:2602.12241 (Feb 2026)
- Moonshine v1: arXiv:2410.15608 (Oct 2024)
- Zipformer: ICLR 2024, Povey et al.
- Nemotron Streaming: arXiv:2312.17279
- SpecASR: arXiv:2507.18181 (Jul 2025)
- SSCFormer: arXiv:2211.11419
- Chunked AED: arXiv:2309.08436
- Unified Streaming Zipformer: arXiv:2506.14434
- XLSR-Transducer: EMNLP 2024 Findings
- wav2vec-S: ACL Findings 2024
- Mamba in Speech: IEEE TASLP 2025
- INT4 Streaming ASR: arXiv:2604.14493 (Apr 2026)
- FastEmit: ICASSP 2023
- Whisper: arXiv:2212.04356
- SeamlessM4T: Nature 2024

### Systems
- NVIDIA Parakeet: Fast Conformer RNNT/CTC
- NVIDIA Nemotron Speech Streaming: Cache-Aware FastConformer
- Meta SeamlessStreaming: w2v-BERT 2.0
- GigaAM-v3: HuBERT-CTC SSL + RNNT (Russian)
- T-one: Conformer-RNNT telephony (Russian)
- Vosk: TDNN+biLSTM (Russian, edge)
