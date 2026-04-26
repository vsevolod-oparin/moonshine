# ru-Moonshine: Russian Edge Streaming ASR

## 1. Why

### The Gap

There is no Russian ASR model that simultaneously satisfies these constraints:

| Requirement | GigaAM-v3 | T-one | Vosk | Whisper | Moonshine v2 |
|-------------|-----------|-------|------|---------|-------------|
| Offline (no network) | Yes | Yes | Yes | Yes | Yes |
| Streaming (<200ms TTFT) | Partial | Yes | Yes | No | Yes |
| Edge / phone / MacBook | Heavy (240M) | Heavy (71M, server-optimized) | Yes (65M) | No (30s window) | Yes (34-245M) |
| Russian | Yes | Yes | Yes | Yes | **No** |
| Low WER (<10% avg) | 8.4% | 8.6% telephony | ~14% | 12-25% | N/A |

**Target use case**: Offline streaming ASR running locally on a phone or MacBook — voice assistants, real-time captioning, dictation — for Russian language.

### Why Moonshine v2 Architecture

1. **Ergodic streaming encoder** — sliding-window attention (16 frames) gives bounded TTFT regardless of utterance length. No other open Russian ASR model has this
2. **Edge-optimized by design** — Tiny (34M/140MB) fits any phone, Small (123M/490MB) fits any modern device
3. **Proven latency** — 34ms (Tiny) to 107ms (Medium) on MacBook Pro for English. Russian should be comparable
4. **MIT-licensed code** — training from scratch on Russian data means we own the model weights with no Community License restrictions
5. **Outperforms Vosk** — Vosk is the only current option for offline+streaming+edge+Russian, but uses older TDNN+biLSTM architecture with ~14% avg WER. Moonshine's transformer architecture should achieve better WER

### Why Not Alternatives

- **GigaAM-v3**: SOTA Russian WER (8.4%) but 240M params, no mobile-optimized streaming, no edge deployment story. Best for server-side
- **T-one**: Great for telephony streaming but server-oriented (300ms chunks, KenLM decoder, no mobile deployment). Apache 2.0
- **Fine-tune Whisper**: No streaming, 30s fixed window, too heavy for edge
- **Vosk**: Only current edge+streaming+Russian option but outdated architecture and high WER

---

## 2. Two-Track Architecture Strategy

We train **two model families** sharing the same tokenizer, data pipeline, and deployment tooling, but differing in encoder architecture:

```
                     Shared
    ┌─────────────────────────────────────┐
    │  Tokenizer (Russian BPE)            │
    │  Data pipeline (preprocessing)      │
    │  Training tricks (CTC loss, augment)│
    │  Inference tricks (cache, quant)    │
    │  Evaluation & benchmarks            │
    │  ONNX export & deployment           │
    └──────────┬──────────────┬───────────┘
               │              │
        ┌──────┴──────┐ ┌────┴──────────┐
        │  ru-Moonshine│ │ ru-Moonshine  │
        │     v2       │ │    v2.1       │
        │ (vanilla)    │ │ (improved)    │
        └──────────────┘ └───────────────┘
```

### Why Two Tracks

| | ru-Moonshine v2 | ru-Moonshine v2.1 |
|---|---|---|
| **Goal** | Baseline, fast to train, uses existing tooling | Push accuracy and efficiency further |
| **Architecture** | Vanilla Moonshine v2 (no changes) | Modified encoder (conv, multi-scale) |
| **Tooling** | Existing Moonshine repo — export, deploy | Custom fork — new encoder blocks |
| **Risk** | Low — proven architecture | Medium — new architecture needs validation |
| **Compatibility** | Full Moonshine ecosystem (iOS, Android, Pi, C++) | Custom ONNX graphs, same inference API |
| **When to use** | Production baseline, quick iteration | When v2 baseline isn't enough, or for research |

**Strategy**: Train v2 first. If it meets targets, ship it. If not, v2.1 is ready to go with architectural improvements. Both share training tricks and inference optimizations.

---

## 3. Architecture

### 3.1 Shared Components (v2 + v2.1)

These are identical across both tracks:

**Decoder** (standard Moonshine v2):
- Causal Transformer with RoPE
- Cross-attention to encoder outputs
- SwiGLU FFN blocks
- Autoregressive token generation

**Audio Preprocessor** (standard Moonshine v2):
- 80-sample non-overlapping windows (5ms at 16kHz)
- Per-frame CMVN (cepstral mean and variance normalization)
- `asinh` nonlinearity
- Two causal stride-2 convolutions → 50Hz feature rate

**Adapter**:
- Learned positional embedding added to encoder outputs
- Linear projection to match decoder dimension

### 3.2 ru-Moonshine v2 (Vanilla)

Exact replica of Moonshine v2 architecture. Uses existing export tooling unchanged.

```
Audio → Preprocessor → Encoder (vanilla) → Adapter → Decoder → Text
                         │
                         └─ Transformer, sliding-window (16,4)/(16,0)
                            No positional embeddings
                            No convolution
```

| Variant | Params | Enc Layers | Dec Layers | Enc Dim | Dec Dim |
|---------|--------|-----------|-----------|---------|---------|
| Tiny | 34M | 6 | 6 | 320 | 320 |
| Small | 123M | 10 | 10 | 620 | 512 |
| Medium | 245M | 14 | 14 | 768 | 640 |

**Encoder block** (repeated N times):
```
x → LayerNorm → Sliding-Window Self-Attention → Residual → LayerNorm → FFN → Residual → output
```
- Layers 0-1 and N-2 to N-1: window (16, 4) — 80ms lookahead
- Middle layers: window (16, 0) — strictly causal

### 3.3 ru-Moonshine v2.1 (Improved)

Modified encoder architecture based on the streaming ASR review. Three changes from vanilla:

```
Audio → Preprocessor → Encoder (improved) → Adapter → Decoder → Text
                         │
                         ├─ Causal depthwise conv in each block
                         ├─ Multi-scale U-Net (50Hz → 25Hz → 50Hz)
                         └─ SSC-style cross-window context
```

#### Change 1: Causal Depthwise Convolution

Added to each encoder block after self-attention:

```
x → LayerNorm → Sliding-Window Attention → Residual
  → LayerNorm → CausalDepthwiseConv(k=7) + GLU → Residual
  → LayerNorm → FFN → Residual → output
```

**Why**: Russian phonetics need explicit local pattern modeling:
- Consonant clusters (вств, нтств, здн)
- Vowel reduction (unstressed о→а, е→и)
- Palatalization (soft vs. hard consonants)

Pure attention must learn these patterns implicitly through 16-frame windows. Causal conv (kernel=7) captures 140ms of local context directly with few parameters (depthwise = `d_model` params per layer).

**Latency cost**: Kernel 7 adds ~6 frames (120ms) of algorithmic delay. Acceptable for our <250ms target.

#### Change 2: Multi-Scale U-Net Encoder

Restructure encoder into three stages with different frame rates. For the **Small variant (10 encoder layers)**:

```
Stage 1 (fine):    50Hz, layers 0-2,   window (16,4)  — phonetic detail
Stage 2 (coarse):  25Hz, layers 3-6,   window (8,2)   — word-level features
Stage 3 (fine):    50Hz, layers 7-9,   window (16,4)  — final refinement
                                            ↑ upsampling + skip connection from Stage 1
```

**Why**:
- Different phonetic phenomena operate at different time scales
- 50Hz (20ms): stop bursts, frication, voicing onset
- 25Hz (40ms): formant transitions, vowel identity, stress patterns
- Deeper layers at 25Hz process 2x fewer frames → less attention compute
- Skip connection preserves fine detail lost during downsampling

**Downsample** (Stage 1 → Stage 2): Causal learned convolution with stride 2 (kernel=2, stride=2, left-only padding). This is strictly causal — no future frame leakage.

**Upsample** (Stage 2 → Stage 3): Nearest-neighbor repeat (each frame duplicated) followed by a learned 1x1 convolution to blend adjacent frames. Strictly causal.

**Skip connection**: Element-wise addition of Stage 1 output (50Hz) to Stage 3 output (after upsampling to 50Hz). Preserves fine phonetic detail.

**Compute savings**: Stage 2 operates on half the frames. ~20% fewer total FLOPs for the encoder.

**Latency cost**: None additional — all down/up operations are causal.

#### Change 3: SSC-Style Cross-Window Context

Modify the sliding-window attention mask so frames near chunk boundaries can attend to adjacent-window frames without increasing the window size.

**Why**: Sliding-window attention creates blind spots at window boundaries. Adjacent frames that belong to the same word may be split across windows. SSC (Sequential Sampling Chunks) lets boundary frames "peek" into neighboring windows via attention mask manipulation — no extra parameters, no extra compute.

**Implementation**: Generate attention masks that allow the first 2 and last 2 frames of each window to attend to the corresponding frames of the previous/next window.

**Latency cost**: Zero — this is a masking strategy, not an architecture change.

#### v2.1 Parameter Estimates (Small, 10 encoder layers)

| Component | v2 Small | v2.1 Small (est.) | Delta |
|-----------|----------|-------------------|-------|
| Preprocessor | 7.74M | 7.74M | 0 |
| Encoder | 43.49M | ~48-52M | +5-9M (conv layers) |
| Adapter | 2.86M | 2.86M | 0 |
| Decoder | 69.27M | 69.27M | 0 |
| **Total** | **123.36M** | **~128-132M** | **+5-9M** |

The added convolution is depthwise (1 filter per channel), so parameter increase is minimal: `d_model × kernel_size` per layer ≈ 620 × 7 = 4.3K params per layer, 10 layers ≈ 43K params. Negligible.

**Note**: Both v2 and v2.1 Small use 10 encoder layers. The v2.1 restructuring fits within the same layer budget (3+4+3). This ensures fair comparison — improvements come from architecture, not capacity.

**FLOPs comparison** (Small encoder, T=250 frames at 50Hz):

| | v2 Small | v2.1 Small |
|---|---|---|
| Attention FLOPs | 10 layers × O(T×w) = 10 × 250×16 | 6 layers × O(250×16) + 4 layers × O(125×8) |
| FFN FLOPs | 10 layers × O(T×d²) | 6 layers × O(250×d²) + 4 layers × O(125×d²) |
| **Total (relative)** | **1.0x** | **~0.8x** |

v2.1 processes 4 of 10 layers at half the frame rate (25Hz), reducing encoder FLOPs by ~20% despite the added conv modules. This means v2.1 should be both more accurate AND faster than v2.

### 3.4 Streaming Decoder Mechanics

The encoder produces streaming outputs via sliding-window attention. The **decoder** must consume these incrementally:

**Decoder trigger policy**: The decoder fires when VAD detects an end-of-speech boundary (pause > 300ms), or when the encoder has accumulated N frames (N=64, configurable), whichever comes first. This balances latency (don't wait too long) and accuracy (more context = better decoding).

**Cross-attention window**: The decoder cross-attends to **all encoder frames accumulated in the current utterance segment**. For short utterances (< 3s), this is the full utterance. For long utterances, the decoder sees a growing window. The encoder's sliding-window already ensures each encoder output captures local context, so even growing cross-attention is bounded in compute per frame.

**Token emission**: The decoder generates tokens autoregressively. In streaming, the decoder is invoked once per trigger event (not per frame). Tokens are emitted as a batch for that segment. Deduplication handles overlap between consecutive segments.

**Segment deduplication**: Consecutive decoder invocations may produce overlapping text. On each new decoder output, compare the last 2 words of the previous segment's output with the first 2 words of the current output. Discard matching prefix words. This simple word-level overlap handling is O(N) and covers the common case of segment boundary repetition.

**Decoder KV cache**: The decoder's self-attention KV cache is carried across segments within the same utterance (continuous speech). It is reset at VAD boundaries (new utterance). This preserves coherence within a single speaker turn.

**Streaming vs non-streaming comparison**: The model supports a non-streaming mode where the encoder processes the full utterance at once and the decoder cross-attends to all frames. Phase 1 evaluation reports both modes to quantify the streaming accuracy cost.

```
Streaming inference timeline:
Audio: [silence] [speech segment 1...] [pause] [speech segment 2...] [pause]
                  ↓ 64 frames or VAD    ↓ decoder fires    ↓ decoder fires
                  ↓ encoder (cached)     ↓ + cross-attn     ↓ + cross-attn
                  ↓ per-frame            ↓ emit tokens      ↓ emit tokens
```

---

## 4. Tokenizer (Shared)

Replace the English BPE tokenizer with a Russian one.

### Design

- **Algorithm**: SentencePiece BPE
- **Vocabulary size**: 256 for Tiny validation, 512 for Small/Medium production
- **Training data**: Transcripts from all training datasets combined (Russian text only) + top 2000 most frequent English loanwords as they appear in Russian web text
- **Normalization**: Lowercase, strip punctuation from tokenizer training
- **Special tokens**: `<blank>` (CTC), `<sos/eos>`, `<pad>`, `<COMMA>`, `<PERIOD>`, `<QUESTION>`

### Considerations

- Russian has ~33 characters + common punctuation
- 256 tokens → short subword units (1-3 characters) → lower per-token decoder latency (good for streaming)
- 512 tokens → more efficient encoding (fewer tokens per word) → better accuracy but slightly higher decoder cost
- The `<blank>` token is needed for CTC auxiliary loss (shared training trick)
- Punctuation tokens (`<COMMA>`, `<PERIOD>`, `<QUESTION>`) enable the model to predict basic punctuation directly, producing immediately usable output without a separate punctuation model

### Code-Switching

Russian speakers frequently use English loanwords in tech and professional contexts: "email", "server", "iPhone", "файл". A pure Russian BPE tokenizer fragments these into individual characters ("email" → 5 tokens), degrading both accuracy and speed.

**Mitigation**: Include the top 2000 English loanwords (as they appear in Russian web text) in the BPE training corpus. This adds ~50-100 useful BPE merge rules — minimal vocabulary impact, significant robustness gain for real-world Russian speech.

### Tokenizer Training

Train custom (not reuse Whisper's multilingual tokenizer):
- Whisper's 51,865 tokens would waste 21% of Small's parameter budget on unused Chinese/Korean tokens
- Custom training is ~10 lines of SentencePiece code and 2 minutes of compute
- Training data comes from the same transcripts used for ASR training — no extra work

Test 256, 512, and 1024 for Small production. The tradeoff is decoder speed (more tokens per word at smaller vocab) vs embedding size. GigaAM achieves SOTA Russian WER (8.4%) with 256 tokens, proving vocab size does not limit accuracy for Russian.

### Subword Regularization

During training, enable SentencePiece stochastic subword sampling (`nbest_size=5`, `alpha=0.1`) to sample different valid BPE segmentations per epoch. This makes the model robust to tokenization boundary ambiguity — especially valuable for Russian where word-internal morpheme boundaries are ambiguous (e.g., "переписывать" → "пере/пис/ывать" or "перепис/ывать"). At inference, use deterministic encoding (`nbest_size=1`). One-line parameter change, minor robustness gain.

---

## 5. Data (Shared)

### Dataset Plan

#### Phase 1: Validation (Tiny, quick iteration)

| Dataset | Hours | License | Use |
|---------|-------|---------|-----|
| Common Voice 19 (ru) | ~500 subset | CC-0 | Train + validation |
| Golos (subset) | ~200 subset | CC-BY 4.0 | Train |

~700 hours is enough to validate the pipeline and get initial WER numbers.

#### Phase 2: Full Training (Small/Medium)

| Priority | Dataset | Hours | License | Commercial? |
|----------|---------|-------|---------|-------------|
| 1 | Common Voice 19 (ru) | ~3,000 | CC-0 | Yes |
| 2 | Golos | ~1,240 | CC-BY 4.0 | Yes (attribution) |
| 3 | Multilingual LibriSpeech (ru) | ~1,000 | CC-BY 4.0 | Yes (attribution) |
| 4 | Russian LibriSpeech (RuLS) | ~98 | Public domain | Yes |
| 5 | SOVA | ~100 | Open | Yes |
| **Total (commercial-safe)** | | **~5,438** | | **Yes** |
| 6 | OpenSTT (clean subsets only) | ~14,000 | CC-BY-NC | No (research only) |

### Data Versioning

Pin dataset versions for reproducibility:
- Common Voice: corpus 21.0 via `artyomboyko/common_voice_21_0_ru` on HuggingFace (CV19 requires license acceptance on mozilla.org and isn't on HF)
- Golos: `SberDevices/Golos` is empty on HuggingFace, requires manual download from Sber sources
- MLS: Russian config doesn't exist in `facebook/multilingual_librispeech` (only: dutch, french, german, italian, polish, portuguese, spanish). Omit.
- RuLS: HuggingFace `istupakov/russian_librispeech`, pin revision hash
- SOVA: pin commit hash

Record versions in a `data/versions.json` manifest. This ensures results are reproducible across phases and by other researchers.

### Data Mixing

Temperature-based sampling with α=0.7 to prevent overfitting to the dominant/cleanest domain:

```
p_i ∝ hours_i^0.7
```

This slightly upweights smaller datasets (SOVA, RuLS) relative to raw proportions, ensuring the model sees enough far-field and telephony audio. Whisper uses this approach.

Validate mixing by checking per-dataset validation WER during Phase 1. If any dataset's WER is >2x the average, adjust temperature.

### Data Preprocessing

1. **Resample** all audio to 16kHz mono WAV
2. **Split** long audio at silence boundaries (energy below threshold for >300ms). Enforce min 1s, max 30s per segment. Discard segments <1s after splitting. This preserves utterance integrity rather than cutting mid-sentence
3. **Filter** by duration: keep 1-30 second clips
4. **VAD filter**: remove clips with >50% silence
5. **Text normalization**:
   - Lowercase
   - Normalize numbers to words using `num2words` with `lang='ru'` (handles Russian declension: "1 тысяча" / "2 тысячи" / "5 тысяч", gendered forms, compound numbers). `ru_num2words` is not on PyPI; `num2words` provides equivalent Russian support. Manual review of edge cases on 1000 training samples
   - Abbreviation expansion: dictionary of ~200 common Russian abbreviations with spoken forms (США → "с ш а", МГУ → "эм гэ у", ФСБ → "эф эс бэ"). Some are spelled out, some read as words — use the most common spoken form
   - Date and time normalization: gendered and declined forms ("1 мая" → "первого мая", "23 февраля" → "двадцать третьего февраля")
   - Hyphenated words: keep as single tokens (standard for Russian NLP — "какой-то", "из-за", "по-русски")
   - Remove non-speech markers (laughter, cough annotations)
   - Unified punctuation handling
6. **Deduplicate** across datasets
7. **Split**: 95% train / 5% validation, no speaker overlap with test. Within each dataset, split by speaker ID (available for Common Voice, MLS). Cross-dataset speaker overlap is accepted as minor noise — different speaker ID namespaces make deduplication impractical

### Data Augmentation

In addition to SpecAugment (Section 6.2), add real-world noise robustness:

- **MUSAN corpus**: Mix training audio with noise/music/babble samples at SNR 0-20dB. Applied to 30% of training batches
- **Room Impulse Response (RIR) simulation**: Convolve training audio with synthetic RIRs to simulate far-field/reverberant conditions. Critical for phone-based ASR where mic is 0.5-1m from speaker
- **Babble noise**: Overlapping speech from other speakers — the most challenging noise type for ASR

These augmentations are standard in ESPnet, Kaldi, and NeMo. Download MUSAN (freely available) and generate RIRs using Kaldi's RIR generators. Minimal engineering effort, significant robustness gain.

---

## 6. Shared Training Tricks (v2 + v2.1)

These techniques apply to both architectures and improve accuracy or training efficiency without changing the model structure.

### 6.1 CTC Auxiliary Loss

Add a linear CTC head on top of the encoder. Train with joint loss:

```
L_total = L_AED + α · L_CTC      where α ≈ 0.2-0.3
```

**What it does**:
- Forces encoder to learn monotonic audio-to-text alignments
- Improves encoder representations → better features for AED decoder
- Provides a frame-synchronous "fast path" for ultra-low-latency streaming (optional at inference)

**Expected impact**: +5-15% relative WER improvement

**Implementation**: One `nn.Linear(enc_dim, vocab_size)` layer + `torch.nn.CTCLoss()`. The CTC head is discarded at inference unless used for the fast path.

**Applicability**: v2 and v2.1. For v2, this is the single most impactful training-only improvement. For v2.1, it compounds with the architectural improvements.

### 6.2 SpecAugment + Speed Perturbation + Noise Augmentation

Standard ASR data augmentation:

| Augmentation | Params | Purpose |
|-------------|--------|---------|
| Time masking | max 10 frames | Robustness to temporal distortion |
| Frequency masking | max 8 bins | Robustness to speaker variation |
| Speed perturbation | 0.9x, 1.0x, 1.1x | Effectively 3x training data |
| MUSAN noise | SNR 0-20dB, 30% of batches | Real-world noise robustness |
| RIR simulation | Synthetic room responses | Far-field/reverberant conditions |
| Babble noise | Overlapping speakers | Challenging noise robustness |

**Expected impact**: +2-5% relative WER improvement (SpecAugment alone). Noise/RIR augmentation provides additional robustness that is critical for phone-based deployment but not captured by clean benchmark WER.

### 6.3 Dynamic Chunk Training

During training, randomly sample the attention window size per batch from a range. The model learns to handle variable amounts of context.

**What it does**: Makes the model robust to different latency budgets at inference. A model trained with only fixed-window (16,4) may degrade if deployed with smaller windows for lower latency.

**Implementation**: Sample `w_left ∈ [8, 16]` and `w_right ∈ [0, 4]` per batch during training.

**Expected impact**: No WER improvement at default window, but enables flexible latency-accuracy tradeoff at deployment without retraining.

### 6.4 Pseudo-Labeling Pipeline (Phase 3)

```
Step 1: Train ru-Moonshine Small on 5.4K labeled hours
Step 2: Run inference on OpenSTT 20K hours → pseudo-labels
Step 3: Filter low-quality labels (multi-stage)
Step 4: Retrain on combined 25K+ hours
```

**What it does**: Converts unlabeled audio into training data using our own model. The biggest accuracy lever available.

**Filtering strategy** (multi-stage):
1. Confidence threshold: keep top 60th percentile by beam score
2. Length sanity: 3-500 characters
3. Language model filter: KenLM trained on Russian Wikipedia, reject if perplexity > 95th percentile of labeled data
4. Spot-check: manually review 200 random samples per iteration

Expected noise rate: 10-30%. Retrain only on filtered subset. Iterate if noise > 25%.

**Expected impact**: +10-30% relative WER improvement, depending on pseudo-label quality.

**Risk**: Label quality depends on initial model quality. Iterative: first pass labels are noisy, but retrained model produces cleaner labels for round 2.

**License note**: OpenSTT is CC-BY-NC. Models trained with it can't be used commercially. If commercial use is needed, stick to the 5.4K commercial-safe hours.

### 6.5 Schedule-Free Optimizer

Moonshine v2 uses Schedule-Free (Defazio et al., 2024) instead of AdamW with cosine scheduling.

**What it does**: Eliminates the need for learning rate schedule tuning. The optimizer automatically adapts LR during training.

**Expected impact**: Simpler training, potentially better final WER. Moonshine team used it successfully at scale.

**Note**: Call `optimizer.train()` / `optimizer.eval()` alongside `model.train()` / `model.eval()`.

### 6.6 Label Smoothing

Standard regularization for autoregressive decoders. Apply label smoothing ε=0.1 to the decoder's cross-entropy loss.

**What it does**: Prevents the decoder from becoming overconfident, improving generalization.

**Expected impact**: +1-3% relative WER improvement. Nearly free to implement (`CrossEntropyLoss(label_smoothing=0.1)` in PyTorch 2.0+).

### 6.7 Transfer Learning from English Moonshine (Optional)

English Moonshine was trained on 300K hours. The audio preprocessor and encoder learn acoustic features that transfer across languages. Initialize from English weights:

- **Keep**: Preprocessor weights, encoder attention/FFN weights, decoder attention/FFN weights
- **Replace**: Tokenizer, decoder embedding matrix, decoder output projection (softmax layer)
- **Fine-tune**: All layers on Russian data with reduced LR (1/10 of base LR for transferred layers, full LR for new layers)

**Why it helps**: The encoder has learned phonetic feature extraction, attention patterns, and acoustic variation modeling from 300K hours. Russian and English share acoustic properties (formants, voicing, frication). Transfer learning typically gives +10-30% relative WER improvement vs training from scratch on the same data.

**When to use**: If training from scratch doesn't meet targets in Phase 1, try transfer learning before committing to v2.1 architectural changes.

**Risk**: The decoder's new random embeddings may cause instability early in fine-tuning. Mitigate with LR warmup and lower initial LR for transferred layers.

### 6.8 Weight Initialization

- **From scratch**: Kaiming normal for attention/FFN layers. Uniform init for embeddings. Small init for output projection (stabilizes early training).
- **Transfer learning**: Initialize from English Moonshine checkpoint. Random init only for new layers (embedding matrix, output projection).

### 6.9 Summary: Training Tricks Impact

| Trick | WER Impact | Effort | Phase | Applies To |
|-------|-----------|--------|-------|------------|
| CTC auxiliary loss | +5-15% rel. | Low | 1 | v2 + v2.1 |
| SpecAugment + speed pert. + noise | +2-5% rel. | Low | 1 | v2 + v2.1 |
| Dynamic chunk training | Enables flex latency | Low | 1 | v2 + v2.1 |
| Label smoothing (ε=0.1) | +1-3% rel. | Low | 1 | v2 + v2.1 |
| Pseudo-labeling | +10-30% rel. | Medium | 3 | v2 + v2.1 |
| Schedule-Free optimizer | Simplifies tuning | Low | 1 | v2 + v2.1 |
| Transfer learning (optional) | +10-30% rel. | Low | 1b | v2 + v2.1 |
| Dropout + attention dropout | +1-3% rel. | Low | 1 | v2 + v2.1 |

---

### 6.10 Regularization

English Moonshine's 300K hours provided implicit regularization. With 5.4K hours, explicit regularization is needed to prevent overfitting to training speakers and acoustic conditions.

**Configuration**:
- Attention dropout: 0.1 — prevents attention overfitting to specific positions
- FFN dropout: 0.1 — prevents co-adaptation of features
- Drop path / stochastic depth: 0.1 (v2.1 only, with deeper multi-scale encoder) — randomly skips encoder layers during training, regularizes deep encoders

**Expected impact**: +1-3% relative WER improvement. Standard in production ASR (GigaAM, Conformer models). Tunable via Phase 1 validation WER.

---

## 7. Shared Inference Optimizations (v2 + v2.1)

These are runtime optimizations that apply to both architectures at inference time. No retraining needed.

### 7.1 Cache-Aware Encoder Inference

**Problem**: Sliding-window attention (left=16) recomputes attention over 16 previous frames for every new frame.

**Solution**: Maintain a circular buffer of encoder attention KV-cache (16 frames per layer). For each new audio frame, compute attention only for the new query against cached keys/values.

**Expected impact**: 8-16x reduction in per-frame encoder compute after warmup.

**Implementation**:
```
On each new audio frame:
1. Preprocess → single feature frame
2. For each encoder layer:
   a. Append new KV to cache, evict oldest
   b. Compute attention: new_query × all_cached_KV
   c. Pass through FFN
3. Feed encoder output to decoder
```

**Compatibility**: v2 — directly applicable (fixed window size makes caching trivial). v2.1 — multi-scale structure requires per-stage caches, but same principle.

### 7.2 INT8 / INT4 Quantization

Post-training quantization via ONNX Runtime:

| Quantization | Tiny Size | Small Size | WER Impact |
|-------------|-----------|-----------|------------|
| FP32 | ~140MB | ~490MB | Baseline |
| INT8 dynamic | ~50MB | ~170MB | <0.5% abs. |
| INT4 k-quant | ~30MB | ~100MB | <1% abs. |

**Expected impact**: 2-4x smaller models, 2-3x faster CPU inference, minimal accuracy loss.

**Recommendation**: INT8 dynamic quantization as default. INT4 only for extreme edge (watch, IoT).

### 7.3 Speculative Decoding (Phase 3)

Use Tiny (34M) as draft model for Small/Medium decoder at inference time.

```
1. Tiny decoder proposes 4-8 tokens in parallel
2. Small decoder verifies in one forward pass
3. Accepted tokens are final
4. Rejected tokens: Small generates from first rejection point
```

**Why it works for ASR**: ASR outputs are highly audio-conditioned → small and large models agree on >80% of tokens. Much higher acceptance rates than LLM speculative decoding.

**Expected impact**: 3-4x decoder speedup, zero accuracy loss.

**Memory cost**: +34M params (Tiny draft model alongside Small). Total: ~157M. Fits any phone.

**Compatibility**: v2 and v2.1 — requires the Tiny variant of the same track as draft model.

### 7.4 ONNX Export + Operator Fusion

Export encoder and decoder as separate ONNX graphs. Apply graph-level operator fusion via ONNX Runtime:

- Fuse LayerNorm + attention projections into single ops
- Fuse conv + activation into single ops
- Enable ONNX Runtime's built-in attention fusion (Flash Attention on GPU, optimized CPU kernels)

**Streaming ONNX export**: Requires explicit KV cache tensors as graph inputs/outputs with `dynamic_axes` for variable-length sequences. The Moonshine repo's existing export scripts provide a working starting point for v2. v2.1 multi-scale encoder requires custom cache management (per-stage buffers). PoC tests T14-T15 validate this before committing to training.

**Expected impact**: 20-40% inference speedup on CPU without any model changes.

### 7.5 Summary: Inference Optimization Impact

| Optimization | Latency Impact | Size Impact | Phase | Applies To |
|-------------|---------------|-------------|-------|------------|
| Cache-aware inference | 8-16x less compute/frame | None | 1 | v2 + v2.1 |
| INT8 quantization | 2-3x faster CPU | 2-4x smaller | 1 | v2 + v2.1 |
| Speculative decoding | 3-4x decoder speedup | +34M (draft model) | 3 | v2 + v2.1 |
| ONNX operator fusion | 20-40% faster | None | 1 | v2 + v2.1 |
| Beam search + LM fusion | None (more compute) | +LM (~50MB) | 2 | v2 + v2.1 |

### 7.6 Decoding Strategy

The plan already trains a KenLM on Russian Wikipedia for pseudo-label filtering (Section 6.4). This same LM is reused for shallow fusion during decoding — zero additional training cost.

**Decoding modes**:

| Mode | Description | When to Use |
|------|-------------|-------------|
| Greedy | argmax at each decoder step | Phase 1 baseline, streaming where latency is critical |
| Beam search (width 5-8) | Explore top-K hypotheses | Non-streaming mode, accuracy-focused |
| Beam + shallow LM fusion | Beam search with external KenLM log-prob weighted into hypothesis scoring | Production deployment, both streaming and non-streaming |

**LM fusion weight**: Tune on validation set (typical range 0.1-0.3). The KenLM log-probability is interpolated with the model's token log-probability: `score = model_score + λ · lm_score`.

**Expected impact**: +10-15% relative WER improvement for beam+LM over greedy. Russian benefits particularly due to rich inflection and phonetically similar case endings where LM context helps disambiguate.

**Streaming latency note**: Beam search adds decoder latency (K sequential decoder steps per beam position). For streaming where decoder latency budget is tight, use greedy or beam width 3. For non-streaming, use beam width 8 + LM.

**Reporting**: Always report both greedy and beam+LM WER to quantify the decoding strategy's contribution.

---

## 8. Proof-of-Concept Tests (Gate Before Training)

These tests run **before Phase 1 training**. Each catches a different class of bugs. If any test in a tier fails, stop and fix before proceeding.

### Tier 1: Architecture Sanity (Day 1-2)

These use random data, no real training needed.

| # | Test | What It Catches | Pass Criteria |
|---|------|----------------|---------------|
| T1 | **Forward pass smoke test** | Shape mismatches, NaN in architecture, wrong tensor ranks | Random audio (16kHz, 5s) → preprocessor → encoder → adapter → decoder → logits. No crash, no NaN, logits shape = (seq_len, vocab_size) |
| T2 | **Backward pass test** | Disconnected computation graph, zero gradients, gradient explosion | Single forward + backward on random batch. Check `grad` is not None and not NaN for all trainable params. Check gradient norm is finite (< 1e6) |
| T3 | **Sliding-window mask test** | Attention mask is wrong — attending to future or missing past | Manual verification: for frame at position t with window (16,0), check that attention mask allows t-16..t but blocks t+1..end. For (16,4), check t-16..t+4 |
| T4 | **Feature extraction test** | Preprocessor produces wrong shapes or values | 5s of 16kHz audio (80,000 samples) → preprocessor → shape (250, enc_dim). Verify 50Hz rate (5s × 50 = 250 frames). Verify no NaN/Inf in features |

### Tier 2: Overfit Tests (Day 2-4)

These use real Russian audio. Proves the model CAN learn.

| # | Test | What It Catches | Pass Criteria |
|---|------|----------------|---------------|
| T5 | **Overfit 10 samples** | Fundamental loss bugs, tokenizer-encoder mismatch, broken attention | Train 500 steps on 10 Russian audio clips with their transcripts. Loss → near 0. WER → 0 on those same 10 clips |
| T6 | **Overfit 100 samples** | Tokenizer can't represent Russian text, convergence too slow | Train 2000 steps on 100 clips. WER < 5% on same clips. If this fails, check tokenizer coverage first |
| T7 | **CTC head sanity** | CTC loss computes incorrectly, wrong blank token, wrong label ordering | Random encoder output + real transcript → CTC loss is finite, positive, and decreases over 50 optimization steps on the CTC head alone |
| T8 | **Joint loss convergence** | CTC + AED losses fight each other, bad α weight | Train 500 steps with L_AED + 0.3·L_CTC on 100 clips. Both losses decrease. No NaN. WER improves vs T5 at same step count |

### Tier 3: Streaming Correctness (Day 4-6)

These validate that streaming mode actually works.

| # | Test | What It Catches | Pass Criteria |
|---|------|----------------|---------------|
| T9 | **Streaming vs non-streaming parity** | Sliding-window chunked inference produces different outputs than full-context | Same audio: (a) encode all frames at once, (b) encode chunk-by-chunk (chunk=32 frames) with state carry-over. Cosine similarity of encoder outputs > 0.99 on overlapping frames |
| T10 | **TTFT is bounded** | Encoder latency grows with audio length (sliding window not actually sliding) | Measure encoder forward time for 1s, 5s, 10s, 30s audio. TTFT should be constant (±10%). If it grows linearly, attention is not actually windowed |
| T11 | **KV cache correctness** | Cache reuse produces different outputs than full recomputation | Encoder with KV cache (incremental) vs encoder without cache (full recomputation). Logits diff < 1e-5. Test with 5s and 30s audio |

### Tier 4: Pipeline End-to-End (Day 5-7)

These validate the full toolchain.

| # | Test | What It Catches | Pass Criteria |
|---|------|----------------|---------------|
| T12 | **Tokenizer roundtrip** | BPE encode/decode is lossy, special token handling broken | 1000 Russian sentences from Common Voice/RuLS → encode → decode → exact string match. Test with: normal text, numbers, hyphenated words, quoted text, English loanwords, abbreviations (США, МГУ), dates ("1 мая", "23 февраля"). Include 200 number-containing sentences to validate `num2words` normalization |
| T13 | **Tokenizer morphology coverage** | BPE vocab too small → excessive fragmentation → slow decoder | Measure average tokens per word on 10K Russian sentences. Target: ≤ 4.0 tokens/word for vocab 256, ≤ 3.5 for vocab 512, ≤ 3.0 for vocab 1024 (adjusted from original English-based targets — Russian Cyrillic + rich morphology requires larger vocab for equivalent coverage). Also measure: % of words encoded as single token. Measure fragmentation of top 100 English loanwords |
| T14 | **ONNX export smoke test** | Export graph breaks, unsupported ops, shape mismatches | PyTorch model → `torch.onnx.export` → load in ONNX Runtime → run inference. Output diff < 1e-4 vs PyTorch. Test encoder and decoder separately |
| T15 | **ONNX streaming test** | Streaming semantics lost in export (KV cache not preserved) | Export encoder with cache inputs/outputs. Run chunk-by-chunk inference in ONNX Runtime. Compare to PyTorch streaming output. Diff < 1e-4 |

### Tier 5: Scale-Up Readiness (Day 6-8)

These run before committing cloud GPU budget.

| # | Test | What It Catches | Pass Criteria |
|---|------|----------------|---------------|
| T16 | **100-hour convergence** | Training diverges at real scale, wrong LR, data pipeline issues | Train 1 epoch on 100h subset of Common Voice Russian. Loss decreases monotonically. WER < 30% after 1 epoch (anything < 30% means the model is learning) |
| T17 | **GPU memory profiling** | OOM at real batch sizes on 3090 and H100 | Profile peak VRAM: Tiny batch=32, Small batch=8 (3090), Small batch=24 (H100). Record actual vs estimated. Adjust batch sizes for Phase 1/2 |
| T18 | **INT8 quantization sanity** | Accuracy collapse after quantization | Train Tiny for 1K steps → export FP32 and INT8 ONNX → run on 100 test clips. WER diff < 1% absolute |

### Test Schedule and Gates

```
Day 1-2:  T1, T2, T3, T4  (architecture sanity — random data, fast)
           ↓ GATE: all pass → continue. T1/T2 failure = architecture bug, must fix.
Day 2-4:  T5, T6, T7, T8  (overfit tests — need real Russian audio)
           ↓ GATE: T5 pass = model can learn. T5 fail = fundamental issue.
           ↓ GATE: T6 pass = tokenizer works. T6 fail = check tokenizer first.
Day 4-6:  T9, T10, T11    (streaming correctness — needs trained model from T5)
           ↓ GATE: T9 pass = streaming actually works. T9 fail = attention mask bug.
Day 5-7:  T12, T13, T14, T15  (pipeline end-to-end — can overlap with above)
           ↓ GATE: T12+T14 pass = can export and run inference.
Day 6-8:  T16, T17, T18  (scale-up readiness — validates before cloud spend)
           ↓ GATE: T16 pass = ready for Phase 1 full training.
           ↓ All pass → proceed to Phase 1 training on 3090
```

**Failure protocol**:
- T1-T4 failure: Fix architecture code. Re-run from T1.
- T5 failure: Debug loss function, data loading, or gradient flow. Do NOT proceed to T6.
- T6 failure: Check tokenizer (run T12 early). If tokenizer passes, increase model capacity or debug convergence.
- T9 failure: Debug sliding-window mask or state carry-over. Streaming is the core feature.
- T14/T15 failure: Debug ONNX export. Can proceed with PyTorch-only training but must fix before deployment.
- T16 failure: Tune LR, check data quality, reduce CTC weight. Do NOT spend cloud budget.

---

## 9. Training Plan

### Training Infrastructure

**Checkpoint policy**: Save every 5K steps. Retain top-3 by validation WER + latest. Auto-resume from latest on restart. Storage estimate: ~250MB per Small BF16 checkpoint, ~5GB total per run.

**Validation during training**: Evaluate WER on validation set every 2K steps via CTC greedy decoding. Log WER alongside loss. Select checkpoints by WER, not loss. Early stopping: abort if WER stops improving for 3 consecutive evaluations (patience = 6K steps).

**Effective batch size**: Target 128-256 via gradient accumulation. On H100 with Small batch=16 per GPU, accumulation steps = 8-16. On 3090 with Small batch=8, accumulation steps = 16-32.

**Distributed training**: Phase 1-2 (Tiny/Small on 3090 or single H100): single GPU with gradient accumulation. Phase 3 Medium (245M): if single H100 OOM, wrap model with PyTorch FSDP and gradient checkpointing — no architecture changes needed.

**Experiment tracking**: Support both Weights & Biases (wandb) and TensorBoard, selectable via config. Default: W&B.

```yaml
# In training config YAML:
logging:
  backend: wandb    # "wandb" or "tensorboard"
  project: ru-moonshine
  name: v2-tiny-phase1
  log_every: 100         # per-step metrics
  eval_every: 2000       # validation WER
```

Implementation: thin wrapper that calls wandb or TensorBoard behind the same interface:

```python
class Logger:
    def __init__(self, backend, project, name, config):
        if backend == "wandb":
            import wandb
            wandb.init(project=project, name=name, config=config)
        elif backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=f"runs/{name}")

    def log(self, metrics: dict, step: int):
        if self.backend == "wandb":
            wandb.log(metrics, step=step)
        else:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, step)
```

Set `logging.backend: tensorboard` if W&B is unavailable (firewall, cloud restrictions, offline training). TensorBoard logs to local `runs/` directory — view with `tensorboard --logdir runs/`. For cloud H100, SSH tunnel: `ssh -L 6006:localhost:6006 user@cloud-host`.

W&B advantages (when available): cloud-hosted (no SSH needed), auto-captured system metrics, cross-run comparison UI. TensorBoard advantages: always works, zero dependencies beyond PyTorch, no account needed.

All runs record: full config YAML, git commit hash, random seed — regardless of backend.

**Training Charts** — same metrics regardless of backend:

Logged every training step:

| Chart | Y-axis | What to look for |
|-------|--------|------------------|
| `loss/total` | Total loss (AED + α·CTC) | Should decrease monotonically. Plateau = convergence. Spike = instability |
| `loss/aed` | Decoder cross-entropy | Decreases slower than CTC. Plateau at ~1-3 is normal |
| `loss/ctc` | CTC loss | Should decrease fast early (encoder learns alignments quickly) |
| `train/grad_norm` | Gradient L2 norm | Should stay < 100. Spikes > 1000 = learning rate too high or data issue |
| `train/lr` | Learning rate | Schedule-Free auto-adapts. Should show decay pattern |

Logged every 2K steps:

| Chart | Y-axis | What to look for |
|-------|--------|------------------|
| `val/wer` | Validation WER (greedy) | **Primary metric.** Should decrease. Plateau for 3 evals = early stop |
| `val/wer_cv` | WER on Common Voice subset | Track per-dataset to catch domain imbalance |
| `val/wer_golos` | WER on Golos subset | Far-field WER. If >> val/wer, need more far-field data/augmentation |
| `val/cer` | Character error rate | Should track WER. Divergence = morphological issues |

Auto-captured system metrics (W&B only — TensorBoard requires manual logging):

| Chart | What it shows |
|-------|---------------|
| GPU utilization | Should be > 85% during training. < 70% = data loading bottleneck |
| GPU memory (VRAM) | Should be stable. Growing = memory leak |
| GPU temperature | Should stabilize at 70-85°C on 3090. > 90°C = thermal throttle |
| GPU power draw | 3090: ~250-350W sustained. Lower = underutilized |

**Run organization**:

```
Project: ru-moonshine
├── Group: phase1-tiny
│   ├── Run: v2-tiny-700h
│   └── Run: v21-tiny-700h
├── Group: ablation
│   ├── Run: abl-B-conv-only
│   ├── Run: abl-C-multiscale-only
│   ├── Run: abl-D-conv-multiscale
│   └── Run: abl-E-full-v21
├── Group: phase2-small
│   ├── Run: v2-small-5k4h
│   └── Run: v21-small-5k4h
└── Group: hp-search
    ├── Run: hp-lr1e4-ctc03
    ├── Run: hp-lr3e4-ctc03
    └── Run: hp-lr3e4-ctc05
```

W&B: cross-run comparison and overlay built-in. TensorBoard: use `--logdir_spec` to compare runs.

**Reproducibility**: Pin all dependencies (PyTorch, CUDA, Python versions) in a Dockerfile for cloud GPU runs. Each training run gets a unique ID + config file. Released weights include a model card with training details.

### Phase 0: Setup (3-5 days, local 3090, $0)

- Clone Moonshine repo, set up environment (Dockerfile with pinned dependencies)
- Train Russian SentencePiece BPE tokenizer (256 tokens)
- Build data pipeline: download, preprocess, create manifests
- Run PoC tests T1-T18 (see Section 8)
- Fork v2.1 encoder architecture

### Phase 1: Validation (2-3 days, RTX 3090, $0)

**Goal**: Verify the full pipeline works for Russian. Get initial WER. Validate both v2 and v2.1 encoder.

| Parameter | Value |
|-----------|-------|
| Model | Tiny (34M) — train BOTH v2 and v2.1 |
| Data | ~700 hours (CV + Golos subset) |
| Epochs | 3 |
| Batch size | 16-32 (effective 128 via gradient accumulation) |
| Precision | FP16 (mixed) |
| Loss | L_AED + 0.3 · L_CTC |
| Label smoothing | ε=0.1 on decoder loss |
| Dropout | Attention 0.1, FFN 0.1 |
| Optimizer | Schedule-Free, LR 2e-3 |
| Augmentation | SpecAugment + speed perturbation + MUSAN noise |
| Hardware | RTX 3090 (24GB) |
| Time | ~2-3 days per model, run sequentially |
| Cost | $0 |

**Success criteria**:
- Both v2 and v2.1 converge (loss plateaus)
- v2.1 shows measurable WER improvement over v2 on validation set
- ONNX export works for both
- Cache-aware inference reduces per-frame latency vs naive inference
- Streaming inference on CPU < 300ms TTFT
- Non-streaming mode WER quantifies streaming accuracy cost

**If v2.1 doesn't improve over v2 at Tiny scale**: Re-evaluate v2.1 architecture changes before scaling up. May drop conv or multi-scale if they don't help at Tiny scale.

### Phase 1b: Evaluate and Iterate (2-3 days, local, $0)

- Compare v2 vs v2.1 WER on validation set
- Run **ablation study** to attribute improvements:

| Run | Architecture | Purpose |
|-----|-------------|---------|
| A | v2 baseline | Control |
| B | v2 + causal conv (k=7) | Isolate conv contribution |
| C | v2 + multi-scale U-Net | Isolate multi-scale contribution |
| D | v2 + conv + multi-scale | Combined without SSC |
| E | Full v2.1 (conv + multi-scale + SSC) | Full stack |

Compare WER on held-out test. If B ≈ E, skip multi-scale and SSC — they add complexity without gain.

- Analyze errors: phonetic confusion, word boundaries, rare words
- Tune hyperparameters (LR, CTC weight α, augmentation params)
- Validate cache-aware inference implementation
- Validate INT8 quantization (<0.5% WER increase)
- If WER is unsatisfactory: try transfer learning from English Moonshine (Section 6.7)

### Phase 2: Production Training (~3-5 days, cloud H100, ~$200-330)

**Goal**: Train Small (123M) models.

#### Step 2a: Hyperparameter Search (1-2 days, H100, ~$50-80)

Run 2-3 variants per track on 1K-hour subset:
- LR: {1e-4, 3e-4, 5e-4}
- Epochs: {3, 5}
- CTC weight α: {0.1, 0.3, 0.5}

Use v2 for HP search (faster iteration). Apply best HP to v2.1.

#### Step 2b: Full Training (2-3 days, H100, ~$150-250)

Train BOTH v2 and v2.1 Small with best HP:

| Parameter | Value |
|-----------|-------|
| Model | Small (123M / ~130M) |
| Data | ~5,400 hours (all commercial-safe) |
| Epochs | Best from HP search (likely 3-5) |
| Batch size | 16-24 per GPU (effective 128-256 via gradient accumulation) |
| Precision | BF16 (mixed) |
| Loss | L_AED + α_best · L_CTC |
| Label smoothing | ε=0.1 on decoder loss |
| Dropout | Attention 0.1, FFN 0.1, drop path 0.1 (v2.1 only) |
| Optimizer | Schedule-Free, LR from HP search |
| Augmentation | SpecAugment + speed perturbation + MUSAN noise + RIR |
| Dynamic chunks | w_left ∈ [8,16], w_right ∈ [0,4] |
| Hardware | 1× H100 |
| Time | ~2-3 days per model |

**Optional curriculum**: If validation WER plateaus early in training, try ordering data: epochs 1-2 on cleaner subsets (Common Voice validated + MLS), epochs 3+ with full mix including noisier data (Golos far-field, SOVA). Evaluate empirically — if no difference vs. random shuffling, drop it.

**WER targets** (tiered expectations):

| Scenario | v2 Small WER | v2.1 Small WER | Condition |
|----------|-------------|----------------|-----------|
| Phase 2 baseline (5.4K hrs) | 12-15% | 10-13% | Realistic |
| Phase 3 + pseudo-labeling (25K hrs) | 10-12% | 8-10% | Aspirational |
| Stretch goal | <10% | <8% | Requires SSL pretrain or 50K+ hrs |

**Success criteria**:
- v2 Small: WER beats Vosk (~14%)
- v2.1 Small: WER lower than v2 Small (architecture improvement validated at scale)
- Both models: streaming latency < 100ms on MacBook CPU
- Both models: ONNX INT8 < 200MB
- Report both streaming and non-streaming WER to quantify streaming cost

### Phase 3: Scale and Refine (optional, ~3-7 days, cloud H100, ~$250-450)

If Small doesn't meet targets:

| Option | Description | Cost |
|--------|-------------|------|
| A: Train Medium v2 | 245M params, same data | ~$250-350 |
| B: Pseudo-labeling | Label OpenSTT 20K hrs, retrain Small | ~$150-250 |
| C: Both | Medium + pseudo-labeling | ~$400-600 |

### Phase 4: Distillation + Deployment (optional)

Distill Small → Tiny using:
- Teacher: trained Small model (v2 or v2.1, whichever is better)
- Student: Tiny architecture (same track as teacher)
- Method:
  1. KL divergence on logits (soft targets from teacher with temperature 2.0)
  2. CTC auxiliary loss on student encoder output
  3. Hidden-state matching: L2 loss between student and teacher encoder outputs (with learned projection to match dimensions)
- Training: Train student for 50K steps on same data, with loss = α·L_KL + β·L_CTC + γ·L_hidden where α=0.7, β=0.2, γ=0.1
- Benefit: Tiny model approaching Small-level accuracy for phone-class devices. Expected: Tiny distilled WER within 1-2% of Small (vs 3-5% gap from training Tiny directly)

---

## 10. Evaluation (Shared)

### Test Sets

**Locked test set** (never used for validation or HP tuning):
- Common Voice 19 ru official test split
- Golos official test split
- Total: ~100-150 hours

Report test WER **once** per finalized model. Do not iterate on test.

**Validation set** (for checkpoint selection and HP tuning):
- 5% of training data, no speaker overlap with train or test

### Metrics

| Metric | Description | Target (Small) |
|--------|-------------|----------------|
| WER (Word Error Rate) | Primary accuracy, greedy decoding | See tiered targets (Section 9) |
| WER (beam + LM) | Beam search (width 8) + shallow LM fusion | 10-15% relative improvement over greedy |
| CER (Character Error Rate) | Character-level accuracy | < 3% |
| G2P-normalized WER | Convert reference+hypothesis to phonemes before WER | Secondary metric |
| TTFT (Time-to-First-Token) | Streaming latency | < 100ms on MacBook CPU |
| RTF (Real-Time Factor) | Throughput | < 0.3 (faster than real-time) |
| Model size (ONNX INT8) | Deployment footprint | < 200MB |
| Non-streaming WER | Full-context encoder + decoder | Quantifies streaming cost |

**G2P-normalized WER**: Convert both reference and hypothesis to phonemes using a rule-based Russian G2P before computing WER. Russian is nearly phonetic, so G2P is straightforward. This metric ignores orthographic case-ending variation (e.g., "большой дом" vs "большого дома" — acceptable errors) while catching phonetic errors (real errors). Useful for Russian where case endings are phonetically reduced in connected speech.

### Error Analysis

After each training phase, run error categorization on 500+ validation samples. Classify each word error into:

| Error Type | Example | Diagnostic Value |
|-----------|---------|------------------|
| Consonant cluster | "вств" → "ст" | Causal conv effectiveness (v2.1) |
| Vowel reduction | unstressed "о" → "а" predicted as "о" | Natural speech data sufficiency |
| Proper noun | Names, places, brands | Training text diversity |
| Code-switching | "email" garbled | English loanword tokenizer coverage |
| Morphological | Wrong case ending | LM rescoring impact |
| Boundary | Missing/extra words at chunk edges | Streaming chunking quality |
| Homophone | "код" vs "кот" | Context modeling quality |

Identify top-3 error categories by frequency after each phase. Target next phase improvements toward those categories. One-time script: word-level alignment + rule-based categorization for Russian.

### Benchmarking Protocol

Latency claims are reproducible only with a defined methodology:

```
Hardware: MacBook Pro M2 (specific model year), 16GB RAM
Runtime: ONNX Runtime 1.x, CPU execution provider
Model: Small, INT8 dynamic quantization
Audio: 100 utterances from Common Voice test split, 5-15s each
Measurement: Average TTFT over 100 utterances
Precondition: 10 warmup runs, then measure
Environment: No other CPU-intensive processes
Decoding: Greedy (for latency), Beam+LM (for accuracy)
```

Apply equivalent methodology per target platform (iPhone, Android, Pi). Report hardware model, runtime version, and decoding mode alongside every latency number.

### Attention Monitoring

During Phase 1 evaluation, visualize encoder attention patterns on 10 validation utterances. Check for attention sink behavior (disproportionate attention to initial frames, as reported by XLSR-Transducer). Moonshine's ergodic encoder (no positional embeddings) may be less susceptible. If sinks appear, add a learnable sink token to encoder input in Phase 2.

### Known Coverage Gaps

**Accented and dialectal speech**: Training data is predominantly metropolitan Russian. Dialectal variation (Southern Russian, Ukrainian-accented, Central Asian Russian) is not covered. Phase 4 stretch goal: collect 50-100 diverse-accent samples for coverage testing.

### Benchmarks

| Test Set | Domain | Why |
|----------|--------|-----|
| Common Voice 19 (ru test split) | Read speech | Standard benchmark |
| Golos (test split) | Far-field + close-talk | Sber's benchmark |
| OpenSTT validation sets | Phone calls, audiobooks | Real-world diversity |
| Self-recorded samples | Your own voice/phone | Practical validation |

### Comparison Targets

| Model | Expected WER | Notes |
|-------|-------------|-------|
| Vosk-model-ru | ~14% | Minimum bar |
| T-one | ~8.6% (telephony) | Aspirational |
| GigaAM-v3 | ~8.4% avg | SOTA (server) |
| **ru-Moonshine v2 Small** | **12-15% baseline** | Realistic target |
| **ru-Moonshine v2.1 Small** | **10-13% baseline** | Improved target |

---

## 11. Deployment (Shared)

### ONNX Export

- Export encoder and decoder as separate ONNX graphs
- v2: uses existing Moonshine export tooling directly
- v2.1: requires custom export (additional conv ops, multi-scale structure)
- Both: same inference API, same runtime

### End-to-End Streaming Latency Budget (Small, INT8, phone-class CPU)

| Stage | Latency | Notes |
|-------|---------|-------|
| Audio capture + buffering | 20-40ms | One frame at 50Hz = 20ms |
| VAD (Silero) | <5ms | Runs on buffered audio |
| Feature extraction | <5ms | CMVN + asinh + conv |
| Encoder (cached, per frame) | 2-5ms | KV cache reuse |
| Decoder (N tokens) | N × 3-8ms | N=3-8 typical for Russian |
| **Total TTFT** | **30-60ms** | Within <100ms target |

Note: This is inference latency only. Audio capture pipeline (AVAudioEngine/AudioRecord) adds OS-dependent overhead (10-30ms) outside the model's control.

### Target Platforms

| Platform | Engine | Model | Size (INT8) | Latency |
|----------|--------|-------|-------------|---------|
| MacBook M1/M2/M3 | ONNX Runtime (CPU/ANE) | Small | ~170MB | < 100ms |
| iPhone 14+ | ONNX Runtime / CoreML | Tiny | ~50MB | < 80ms |
| Android (flagship) | ONNX Runtime (NNAPI) | Small | ~170MB | < 150ms |
| Raspberry Pi 5 | ONNX Runtime (CPU) | Tiny | ~50MB | < 300ms |

### Inference Pipeline

```
1. Audio input (16kHz mono)
2. VAD (Silero VAD, ~2MB) — detect speech segments
3. Feature extraction (CMVN + asinh + stride-2 conv → 50Hz)
4. Encoder (streaming, cache-aware):
   a. For each new frame: compute attention vs cached KV
   b. Update cache
5. Decoder (autoregressive, with KV cache):
   a. Fires at VAD boundaries or every 64 encoder frames
   b. Cross-attends to accumulated encoder outputs in current segment
   c. Decoder KV cache carried within utterance, reset at VAD boundary
   d. Optional: speculative decoding with Tiny draft
6. LM rescoring (optional): apply KenLM shallow fusion during beam search
7. Segment deduplication: word-level overlap check between consecutive segments
8. Text output (post-process)
```

### Streaming Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Truncated first word | VAD misses speech onset (< 200ms) | Minimum speech duration filter: reject VAD segments < 200ms. Buffer last 200ms of "silence" before speech onset |
| Spurious decoder output | VAD false positive (background noise triggers speech detection) | Minimum decoder confidence threshold: discard decoder output if average log-prob < threshold |
| Missing words at boundaries | Segment split mid-word | Decoder trigger at 64 frames (1.28s) is conservative enough to avoid mid-word splits in most cases. Deduplication handles residual overlap |
| Decoder latency spike | Fast speaker produces many tokens per segment | Increase trigger frame count (64 → 96) for fast speech. Measure decoder latency per segment in production |
| Background speech | Other speakers trigger VAD | Silero VAD is single-speaker. Multi-speaker separation is out of scope — document as known limitation |
| Audio buffer overflow | Very long utterance (> 30s continuous speech) | Force decoder trigger at max segment length (30s / 1500 encoder frames) regardless of VAD |

### Output Post-Processing

**Capitalization**: Rule-based. Russian capitalization is mostly sentence-initial + proper nouns. Simple rules handle ~85% of cases. Proper noun dictionary for common names.

**Punctuation**: The model predicts `<COMMA>`, `<PERIOD>`, `<QUESTION>` tokens directly during decoding. These are converted to actual punctuation in post-processing. This produces immediately usable output without a separate punctuation model.

---

## 12. Timeline and Budget

### Timeline

| Phase | Duration | Hardware | Cost | Deliverable |
|-------|----------|----------|------|-------------|
| 0. Setup | 3-5 days | Local 3090 | $0 | Tokenizer, data pipeline, v2.1 fork |
| 0b. PoC tests (T1-T18) | 6-8 days | Local 3090 | $0 | All gates pass, ready for training |
| 1. Validation (v2 + v2.1 Tiny) | 4-6 days | RTX 3090 | $0 | Both Tiny models, initial WER |
| 1b. Evaluate + iterate + ablation | 4-6 days | Local | $0 | HP tuning, ablation results |
| 2a. HP search | 1-2 days | Cloud H100 | ~$50-80 | Best training config |
| 2b. Full training (v2 + v2.1 Small) | 4-6 days | Cloud H100 | ~$200-350 | Both Small models |
| 3. Evaluation + export | 1-2 days | Local | $0 | Benchmarks, ONNX models |
| 4. Mobile testing | 3-5 days | Local + phone | $0 | Working demo |
| **Total** | **~4-6 weeks** | | **~$250-430** | |

### Budget Summary

| Item | Cost |
|------|------|
| Phase 1 (3090, free) | $0 |
| Phase 2a HP search (1× H100, ~1 day) | ~$50-80 |
| Phase 2b full train (1× H100, ~4-6 days for both) | ~$200-350 |
| Phase 3 pseudo-labeling (optional) | ~$150-250 |
| Phase 3 Medium (optional) | ~$250-450 |
| **Total (Small only, both tracks)** | **~$250-430** |
| **Total (with Medium + pseudo-labeling)** | **~$650-1,130** |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| v2.1 doesn't improve over v2 at Tiny scale | Medium | Low — we still ship v2 | Ablation identifies which changes help. Phase 1 is the validation gate |
| WER > 15% on v2 Small | Medium | High | Transfer learning from English Moonshine (Section 6.7), pseudo-labeling, or train Medium |
| v2.1 ONNX export issues | Medium | Medium — can't deploy | Multi-scale + conv adds complexity. Validate export in Phase 1 with Tiny |
| Tokenizer handles Russian morphology poorly | Low | Medium | Use 512 vocab, include English loanwords, train on diverse text. **Resolved in M2**: 3.08 tpw at vocab 512 is acceptable; GigaAM proves 256 works for accuracy. Decoder RTF ~20% slower than planned assumption but TTFT targets unaffected |
| Cache-aware inference doesn't work with v2.1 multi-scale | Low | Medium | Multi-scale requires per-stage caches; fallback to naive inference |
| Not enough data diversity | Medium | Medium | Speed perturbation, SpecAugment, MUSAN noise, RIR, pseudo-labeling |
| Clean training data doesn't generalize to real-world | Medium | Medium | MUSAN noise + RIR augmentation (Section 6.2). Validate on noisy test sets |
| Transfer learning causes instability | Low | Low | Use reduced LR for transferred layers, full LR for new layers. LR warmup |

---

## 14. Deliverables

| # | Deliverable | v2 | v2.1 |
|---|------------|----|------|
| 1 | Russian SentencePiece BPE tokenizer (256 + 512) | Yes | Yes |
| 2 | Tiny model weights (ONNX + PyTorch) | Yes | Yes |
| 3 | Small model weights (ONNX + PyTorch) | Yes | Yes |
| 4 | Cache-aware streaming inference code | Yes | Yes |
| 5 | Speculative decoding (Tiny→Small) | Yes | Yes |
| 6 | Evaluation results on Russian benchmarks (streaming + non-streaming) | Yes | Yes |
| 7 | Training pipeline (reproducible) | Yes (existing tooling) | Yes (custom fork) |
| 8 | Ablation study results | Yes | Yes |
| 9 | (Optional) Medium model weights | Yes | Yes |
| 10 | (Optional) Mobile demo app | Yes | Yes |

All weights released under MIT or Apache 2.0.

---

## Appendix A: Architecture Comparison at a Glance

```
ru-Moonshine v2 (vanilla)                 ru-Moonshine v2.1 (improved)
━━━━━━━━━━━━━━━━━━━━━━━━━━                 ━━━━━━━━━━━━━━━━━━━━━━━━━━

Encoder block:                            Encoder block:
┌──────────────────────┐                  ┌──────────────────────┐
│ Sliding-Window Attn  │                  │ Sliding-Window Attn  │
│        ↓              │                  │        ↓              │
│       FFN             │                  │ Causal DW Conv (k=7) │
│        ↓              │                  │        ↓              │
│      output           │                  │       FFN             │
└──────────────────────┘                  │        ↓              │
                                          │      output           └──────────────────────┘
Stack: 6/10/14 identical blocks           └──────────────────────┘
All at 50Hz                               
                                          Stack: 3 stages (10 layers)
                                          Stage 1: 3 blocks @ 50Hz
                                          Stage 2: 4 blocks @ 25Hz (stride-2)
                                          Stage 3: 3 blocks @ 50Hz (upsample)
                                          + skip connection Stage 1→3
                                          + SSC cross-window context

Training tricks (shared):                 Training tricks (shared):
✓ CTC auxiliary loss                      ✓ CTC auxiliary loss
✓ SpecAugment + speed pert. + noise       ✓ SpecAugment + speed pert. + noise
✓ Dynamic chunk training                  ✓ Dynamic chunk training
✓ Schedule-Free optimizer                 ✓ Schedule-Free optimizer
✓ Label smoothing (ε=0.1)                ✓ Label smoothing (ε=0.1)
✓ Dropout (attention 0.1, FFN 0.1)       ✓ Dropout (attention 0.1, FFN 0.1, drop path 0.1)
✓ Pseudo-labeling pipeline                ✓ Pseudo-labeling pipeline
○ Transfer learning (optional)            ○ Transfer learning (optional)

Inference (shared):                       Inference (shared):
✓ Cache-aware encoder (KV-cache reuse)    ✓ Cache-aware encoder (per-stage cache)
✓ INT8 / INT4 quantization                ✓ INT8 / INT4 quantization
✓ Speculative decoding (Tiny→Small)       ✓ Speculative decoding (Tiny→Small)
✓ ONNX export + operator fusion           ✓ ONNX export + operator fusion
✓ Punctuation token output                ✓ Punctuation token output
✓ Beam search + shallow LM fusion         ✓ Beam search + shallow LM fusion

Tooling:                                  Tooling:
Existing moonshine repo                   Custom fork
```

## Appendix B: References

### Architecture & Techniques
- Moonshine v2: arXiv:2602.12241
- Moonshine v1: arXiv:2410.15608
- Zipformer (multi-scale): ICLR 2024, Povey et al.
- SSCFormer (cross-window): arXiv:2211.11419
- Nemotron cache-aware: arXiv:2312.17279
- SpecASR (speculative decoding): arXiv:2507.18181
- CTC: Graves et al., ICML 2006
- Schedule-Free optimizer: Defazio et al., 2024
- INT4 streaming quantization: arXiv:2604.14493

### Russian ASR
- GigaAM-v3: https://github.com/salute-developers/GigaAM
- T-one: https://huggingface.co/t-tech/T-one
- OpenSTT: https://github.com/snakers4/open_stt
- Golos: https://huggingface.co/datasets/salute-developers/golos
- Russian LibriSpeech: https://huggingface.co/datasets/istupakov/russian_librispeech

### Full streaming ASR review
- See STREAMING_ASR_REVIEW.md for comprehensive analysis of all techniques

### Code
- Moonshine GitHub: https://github.com/moonshine-ai/moonshine (MIT license)

### Review History
- See MOONSHINE_DS_COMMENTS.md for external review comments
- See MOONSHINE_DS_EVALUATION.md for evaluation of those comments
- See MOONSHINE_DS_CRITIC.md for first-pass critical review
- See MOONSHINE_DS_CRITIC_RESPONSE.md for response to that review
- See MOONSHINE_CRITIC_V2.md for second-pass critical review
- See MOONSHINE_GLM_COMMENTS.md for tokenizer analysis
- See MOONSHINE_CRITIC_V3.md for third-pass critical review
- See MOONSHINE_CRITIC_V3_RESPONSE.md for response to V3 review
