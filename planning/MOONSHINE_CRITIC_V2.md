# Second-Pass Critical Evaluation of ru-Moonshine Plan

Gaps not covered by MOONSHINE_DS_CRITIC.md, MOONSHINE_DS_COMMENTS.md, MOONSHINE_DS_EVALUATION.md, or MOONSHINE_GLM_COMMENTS.md.

---

## Critical Gaps

### 1. Transfer learning from English Moonshine — not discussed at all

**This is the biggest miss in the entire plan.** English Moonshine was trained on 300K hours. The plan trains from scratch on 5.4K hours. But the audio preprocessor, most of the encoder (ergodic/position-independent), and the decoder architecture are identical between English and Russian Moonshine. Only the tokenizer + embedding layers change.

Standard practice in multilingual ASR is to initialize from a well-trained model and replace language-specific layers:
- Keep: preprocessor weights, encoder attention/FFN weights, decoder attention/FFN weights
- Replace: tokenizer, encoder input projection (if it exists), decoder embedding matrix, decoder output projection (softmax layer)

**Why it matters**: The encoder has already learned to extract phonetic features, model attention patterns, and handle acoustic variation from 300K hours. Russian and English share many acoustic properties (formant structure, voicing, frication). Transfer learning from English → Russian typically gives +10-30% relative WER improvement vs training from scratch on the same data.

**Counterargument**: The English tokenizer is baked into the decoder's embedding and output projection. Replacing these means the decoder starts from random on the new token space. But the decoder's attention mechanisms and FFN layers still retain useful English-pretrained knowledge.

**The plan should address this explicitly** — either adopt transfer learning or explain why training from scratch is preferred. "We own the weights with no Community License restrictions" (Section 1) is a licensing argument, not an ML one. You can train from scratch AND initialize from English weights.

---

### 2. Decoder streaming mechanics — completely undefined

The plan talks extensively about encoder streaming (sliding window, cache-aware KV) but never explains how the **AED decoder operates in streaming mode**. This is the missing link between "streaming encoder" and "streaming ASR system."

Critical unanswered questions:
- **When does the decoder fire?** Every new encoder frame (50Hz = every 20ms)? Every N frames? After VAD detects a pause?
- **What does the decoder attend to?** In streaming, encoder outputs change every frame. Does the decoder re-cross-attend to all encoder outputs seen so far? Just the latest? A sliding window of encoder outputs?
- **How are partial transcriptions emitted?** Does the decoder emit tokens incrementally, or wait for an end-of-sentence signal? How do you handle mid-word updates?
- **Decoder KV cache in streaming**: The plan describes encoder KV caching but says nothing about decoder KV caching across streaming steps. The decoder's self-attention KV cache grows with each generated token — in streaming, do you carry this cache across frames?

Moonshine v2's original implementation likely handles this, but the plan doesn't describe it. Without this, the streaming latency claims are unsubstantiated.

**Fix**: Add a "Streaming Decoder Mechanics" subsection to Section 3 or 7 describing:
1. Decoder trigger policy (e.g., "decoder fires when VAD detects end of utterance, or every 32 encoder frames, whichever comes first")
2. Cross-attention window (all encoder frames in current utterance, or sliding window)
3. Token emission strategy (incremental with deduplication, or batch at utterance end)
4. Decoder KV cache carry-over across decoder invocations

---

### 3. Capitalization and punctuation — missing from output pipeline

Data preprocessing lowercases everything. Inference pipeline says "post-process: capitalize, punctuate." But there's no plan for how.

Options:
1. **Rule-based**: Russian capitalization is mostly sentence-initial + proper nouns. Simple rules handle ~85% of cases. Punctuation is harder.
2. **Separate punctuation model**: Train a small tagger on punctuated text (Russian Wikipedia). Adds ~5-10M params. Standard approach (e.g., Olivetti et al.).
3. **Include punctuation tokens in ASR output**: Add `<COMMA>`, `<PERIOD>`, `<QUESTION>` tokens to the vocabulary and train the ASR model to predict them. Increases WER slightly but produces usable output.
4. **Leave it for the application layer**: Raw lowercase text is the ASR model's job. The calling application handles formatting.

This matters for user experience. Raw lowercase "сегодня я пошел в магазин и купил хлеб" is significantly less usable than "Сегодня я пошел в магазин и купил хлеб." For a voice assistant or dictation use case, punctuation affects downstream NLU quality.

**Fix**: Choose a strategy. Recommendation: Option 3 (punctuation tokens in vocab) — adds 3-4 tokens to the 256-512 vocab, zero architecture change, and produces immediately usable output. Capitalization via rule-based post-processing.

---

### 4. No noise / real-world robustness strategy

Training data is predominantly clean:
- Common Voice: web-recorded, mostly quiet environments
- MLS: audiobooks (studio quality)
- Golos: device-recorded but controlled
- RuLS: read speech

The target use case (phone/MacBook in real environments) encounters: background TV/music, street noise, café chatter, wind, keyboard clicks. SpecAugment helps but is a weak substitute for real noise.

Missing augmentations that are standard in production ASR:
- **Background noise augmentation**: Mix training audio with noise samples (MUSAN corpus, free). Standard in every production ASR system.
- **Room impulse response (RIR) simulation**: Simulate far-field/reverberant conditions. Critical for phone-based ASR where the mic is 0.5-1m from the speaker.
- **Babble noise**: Overlapping speech from other speakers. The most challenging noise type for ASR.

**Fix**: Add to augmentation strategy:
1. Download MUSAN corpus (noise + music + babble, freely available)
2. Download synthetic RIRs (e.g., from Kaldi's RIR generators)
3. Augment 30% of training batches with: SNR 0-20dB noise + RIR convolution
4. This is standard in ESPnet, Kaldi, NeMo — not exotic.

---

### 5. Code-switching — English loanwords will break

Russian speakers frequently use English words: "скачать файл", "отправить email", "запустить server", "купить iPhone". In tech/professional contexts, code-switching is the norm.

With a 256-512 Russian BPE tokenizer trained on Russian-only text:
- English words fragment into individual characters or very short subwords
- "email" → ["e", "m", "a", "i", "l"] — 5 decoder steps for one word
- The model has never seen these patterns in training data
- Decoding accuracy for English words will be very poor

**Fix**: Include common English loanwords in the tokenizer training corpus. Add the top 1000-2000 most frequent English words (as they appear in Russian text) to the BPE training data. This adds maybe 50-100 useful BPE tokens to the vocabulary — minimal vocab impact, significant robustness gain.

---

### 6. v2.1 encoder layer count inconsistency

v2 Small has **10 encoder layers**. v2.1 Small's multi-scale description says:
> Stage 1: layers 0-3, Stage 2: layers 4-7, Stage 3: layers 8-11

That's **12 encoder layers** (4+4+4). But the parameter table (Section 3.3) says v2.1 Small is ~128-132M, only +5-9M over v2. Two extra encoder layers would add ~8-9M params (matching the estimate), but this means v2.1 Small is NOT the same layer count as v2 Small.

The plan doesn't address this. Either:
- v2.1 Small uses 12 layers (deeper encoder, more compute, slightly more params) — should be stated explicitly
- The multi-scale should be adapted to 10 layers (e.g., 3+4+3) — should be specified

This matters for fair comparison. If v2.1 improves because it has more layers, not because of the architectural changes, the ablation is confounded.

**Fix**: Explicitly state whether v2.1 Small uses 10 or 12 encoder layers. If 12, note that this is a capacity increase alongside the architectural change. Include this in the ablation plan (MOONSHINE_DS_CRITIC.md point #8): add a "v2 with 12 layers" baseline to distinguish capacity effects from architecture effects.

---

### 7. No weight initialization strategy

Training from scratch on 5.4K hours with random initialization is high-risk. The plan doesn't discuss:
- Xavier/Kaiming initialization (standard, but which variant?)
- Pre-norm vs post-norm implications for init
- Whether any layers benefit from pretrained initialization (even from English Moonshine — see point #1)
- Embedding initialization (random vs pretrained word vectors)
- Output projection initialization (small init for stable early training)

This matters because initialization can be the difference between convergence and divergence with limited data.

**Fix**: Add brief note: "Initialize with Kaiming normal for attention/FFN layers. Decoder embedding initialized uniformly. If using transfer learning (point #1), initialize encoder/decoder attention and FFN from English Moonshine checkpoint."

---

### 8. No label smoothing for decoder

Standard regularization for autoregressive decoders. Label smoothing (typically ε=0.1) prevents the decoder from becoming overconfident and improves generalization by 1-3% relative WER.

Not mentioned in the plan. Nearly free to add (`CrossEntropyLoss(label_smoothing=0.1)` in PyTorch 2.0+).

**Fix**: Add to training tricks: "Label smoothing ε=0.1 on decoder cross-entropy loss."

---

### 9. No data versioning or reproducibility

Common Voice releases new versions regularly (CV 16, 17, 18, 19...). Golos has had updates. MLS too. The plan says "Common Voice 19" but doesn't pin the exact release date or commit.

Without version pinning:
- Results are not reproducible 6 months from now
- Comparisons between phases may use slightly different data
- Other researchers can't replicate

**Fix**: Pin dataset versions explicitly:
```
Common Voice: corpus 19.0, released YYYY-MM-DD, sha256: ...
Golos: v1.0, HuggingFace dataset revision abc123
MLS: ru split, v1.0, released YYYY-MM-DD
```

---

### 10. No non-streaming baseline comparison

The plan compares against Vosk, GigaAM, T-one — but doesn't establish a non-streaming WER floor for the same architecture. If you run the same model with full-context (non-streaming) attention, what WER do you get?

This baseline tells you how much accuracy you're losing from streaming constraints. If streaming WER is 12% and non-streaming is 11%, streaming cost is minimal. If streaming is 12% and non-streaming is 8%, the streaming architecture is the bottleneck.

**Fix**: Add a non-streaming evaluation mode (full attention, no sliding window) to Phase 1 evaluation. Report both streaming and non-streaming WER. This is a single inference-mode change, no retraining.

---

### 11. Duration filter splits long audio poorly

The 1-30 second duration filter will split audiobook segments from MLS. But the plan doesn't say how:
- Split at silence boundaries (VAD-based)? Good — preserves utterance integrity.
- Split at arbitrary points? Bad — creates mid-sentence cuts that the model must learn to handle (and will never see at inference if VAD properly segments).

MLS contains long readings (60-300s). Naive splitting at 30s creates artificial boundaries. The preprocessing should split on silence >300ms with a max-segment constraint.

**Fix**: Specify: "Split audio at silence boundaries (energy < threshold for >300ms). Enforce min 1s, max 30s. Discard segments <1s after splitting."

---

### 12. Russian number normalization is underspecified

"Normalize numbers to words" is listed as a preprocessing step. But Russian number normalization is complex:
- Declension: "1 тысяча" / "2 тысячи" / "5 тысяч"
- Gender: "один" / "одна" / "одно"
- Ordinal vs cardinal: "первый" vs "один"
- Compound numbers: "двести тридцать пять"

The plan doesn't specify what library handles this. `num2words` supports Russian but has known edge cases. `ru_num2words` is more complete. Manual rule-based normalization is common.

**Fix**: Specify the normalization tool. Recommend `ru_num2words` + manual review of edge cases on 1000 samples from training data. Add to T12 (tokenizer roundtrip test): include 200 number-containing sentences.

---

## Additional Recommendations

| Item | Priority | Rationale |
|------|----------|-----------|
| Transfer learning from English Moonshine | **High** | Potentially the single biggest accuracy lever not in the plan |
| Decoder streaming mechanics | **High** | Without this, streaming claims are unsubstantiated |
| Punctuation strategy | Medium | Critical for user experience in target use case |
| Noise augmentation (MUSAN + RIR) | Medium | Production robustness — clean-only training won't generalize |
| Code-switching in tokenizer | Medium | English loanwords are ubiquitous in Russian speech |
| Non-streaming baseline WER | Medium | Quantifies streaming accuracy cost |
| Label smoothing | Low | Standard, nearly free, 1-3% improvement |
| Data versioning | Low | Reproducibility best practice |
| v2.1 layer count fix | Medium | Inconsistency in the plan; confounds ablation |

---

## Summary

Previous reviews covered: test set strategy, data mixing, WER expectations, validation during training, checkpoints, pseudo-labeling, mobile latency, ablation, ONNX export, gradient accumulation, evaluation metrics, tokenizer vocab size, implementation difficulty, batch sizes.

**This review finds 3 major structural gaps that previous reviews missed:**

1. **Transfer learning** — the plan trains from scratch on 5.4K hours when 300K-hour English Moonshine weights are available for initialization. This could be worth 10-30% relative WER improvement for free.

2. **Decoder streaming mechanics** — the plan describes encoder streaming in detail but never explains how the AED decoder interacts with streaming encoder outputs. Without this, the entire streaming story is incomplete.

3. **Real-world robustness** — training data is clean, augmentation is weak (SpecAugment only), and there's no noise/babble/RIR augmentation. Production performance will be significantly worse than benchmark WER.

The v2.1 layer count inconsistency (#6) is a concrete error in the plan that should be corrected.
