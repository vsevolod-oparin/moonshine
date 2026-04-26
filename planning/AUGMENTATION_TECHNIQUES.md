# ASR Training Data Augmentation Techniques

Catalog of augmentation techniques used in production ASR frameworks (ESPnet, NeMo, SpeechBrain), organized by where in the pipeline they're applied. Techniques are rated by **relevance to ru-Moonshine** (our edge streaming ASR on ~248h of data).

---

## Pipeline Stages

```
Raw Audio → [Waveform Augmentation] → Feature Extraction → [Feature Augmentation] → Encoder → Decoder
                (CPU, in DataLoader)       (STFT + Mel)        (GPU, in model forward)
```

---

## 1. Waveform-Level Augmentations (CPU, in DataLoader)

Applied to raw audio before feature extraction. These modify the signal directly and are the most realistic augmentations.

### 1.1 Speed Perturbation

**What**: Resamples audio to simulate faster/slower speech. Changes both speed and pitch.

**How**: Pre-create N `torchaudio.transforms.Resample` objects (one per speed). Randomly select one per training sample.

| Param | Typical | Notes |
|-------|---------|-------|
| speeds | [0.9, 1.0, 1.1] | 3x data multiplier. Some use [0.95, 1.0, 1.05] for less distortion |
| orig_freq | 16000 | Must match audio sample rate |

**Source**: All three frameworks. SpeechBrain pre-creates Resample objects (fast). NeMo uses librosa (slow — recommends pre-computing if possible).

**Relevance**: **HIGH** — effectively 3x training data. Already in our plan.

**Gotcha**: Changes audio duration → batch sizes vary if using duration-based batching. Handle via padding in collation.

### 1.2 Time Stretch (Pitch-Preserving)

**What**: Changes speech speed without changing pitch using phase vocoder (STFT → phase manipulation → ISTFT).

| Param | Typical | Notes |
|-------|---------|-------|
| min_rate | 0.9 | |
| max_rate | 1.1 | |
| n_fft | 512 | Doubled for slowdown to reduce artifacts |

**Source**: NeMo (`TimeStretchPerturbation`). Numba-accelerated when available.

**Relevance**: **MEDIUM** — more natural than speed perturbation but computationally expensive. Consider for Phase 2.

### 1.3 Noise Addition (SNR-Controlled)

**What**: Mixes background noise at a random Signal-to-Noise Ratio.

| Param | Typical | Notes |
|-------|---------|-------|
| noise_files | MUSAN corpus | Environmental, music, babble |
| snr_range | [0, 20] dB | Uniform random per sample |
| apply_prob | 0.3 | Probability per sample |

**Source**: All three frameworks. ESPnet handles short noise (wraps/repeats) and long noise (random crop).

**Relevance**: **HIGH** — critical for phone deployment robustness. MUSAN + RIR is the standard combo.

**Variants**:
- **Foreground noise bursts** (NeMo): adds 1-5 short noise bursts at random positions, not full-length mixing. Simulates door slams, keyboard clicks, etc.
- **White noise**: simpler variant — no external files needed. Generate `N(0, σ²)` directly.
- **Pink noise** (SpeechBrain): generates 1/f^α noise via FFT spectral shaping. More realistic than white noise.

### 1.4 Room Impulse Response (RIR) Convolution

**What**: Convolves audio with a Room Impulse Response to simulate reverberation. Models the effect of sound bouncing off walls before reaching the microphone.

| Param | Typical | Notes |
|-------|---------|-------|
| rir_files | Synthetic RIRs or measured | Kaldi RIR generators, OpenSLR RIR dataset |
| apply_prob | 0.3-1.0 | |
| normalize_power | true | Preserve original signal power after convolution |

**Source**: All three frameworks. ESPnet additionally renormalizes mean power. NeMo optionally shifts output to compensate propagation delay.

**Relevance**: **HIGH** — critical for phone-based ASR where mic is 0.5-1m from speaker. Standard in production ASR.

### 1.5 Gain Perturbation (Volume Randomization)

**What**: Randomly scales audio amplitude by a gain in dB.

| Param | Typical | Notes |
|-------|---------|-------|
| min_gain_db | -10 | |
| max_gain_db | +10 | |

**Source**: NeMo (`GainPerturbation`).

**Relevance**: **LOW-MEDIUM** — simple to implement, makes model robust to volume variation. SpeechBrain has similar `RandAmp` (scales to random factor in [0.5, 1.5]).

### 1.6 Codec Degradation

**What**: Applies audio codec compression/decompression to simulate telephone/low-bitrate artifacts.

| Codec | Bitrate | Effect |
|-------|---------|--------|
| G.711 (μ-law) | 64 kbps | Telephone quality |
| AMR-NB | 4.75-12.2 kbps | Mobile phone |
| MP3 | Variable | Music codec artifacts |
| G.722 | 64 kbps | Wideband telephony |

**Source**: NeMo (`TranscodePerturbation` via sox). SpeechBrain (`CodecAugment` via torchaudio).

**Relevance**: **MEDIUM** — useful if deployment includes phone calls. Not needed for clean microphone input.

### 1.7 DropBitResolution

**What**: Quantizes audio to lower bit depth (int16/int8/float16) then converts back, simulating low-resolution ADC degradation.

**Source**: SpeechBrain (`DropBitResolution`).

**Relevance**: **LOW** — niche. Useful for embedded/IoT deployment with cheap microphones.

### 1.8 Audio Clipping Simulation

**What**: Normalizes audio to [-1,1], then clamps to a random lower amplitude range, then restores original scale. Simulates microphone clipping.

**Source**: SpeechBrain (`DoClip`).

**Relevance**: **LOW** — niche, for harsh acoustic environments.

### 1.9 Time Shift (Circular)

**What**: Circular-shifts audio samples with zero-padding for vacated positions. Simulates timing misalignment.

| Param | Typical |
|-------|---------|
| min_shift_ms | -5 |
| max_shift_ms | +5 |

**Source**: NeMo (`ShiftPerturbation`).

**Relevance**: **LOW** — minimal impact for streaming ASR where alignment is handled by the encoder.

### 1.10 Silence Addition

**What**: Prepends/appends random-duration silence to audio.

**Source**: NeMo (`SilencePerturbation`).

**Relevance**: **LOW** — may help with VAD training but not directly useful for ASR accuracy.

---

## 2. Feature-Level Augmentations (GPU, in model forward pass)

Applied to mel spectrograms after feature extraction. These are the most common ASR augmentations.

### 2.1 SpecAugment (Time + Frequency Masking)

**What**: Zeroes out random contiguous blocks in the time and/or frequency axes of the mel spectrogram.

| Param | Typical | Notes |
|-------|---------|-------|
| freq_masks | 2 | Number of frequency bands to mask |
| freq_width | 10-20 | Max width (in mel bins) per mask |
| time_masks | 2 | Number of time bands to mask |
| time_width | 10-40 or 0.05 (adaptive) | Max width in frames, or float = % of sequence length |
| mask_value | 0 | Fill value (0 or mean) |

**Adaptive time masking**: If `time_width` is a float in [0, 1], mask width scales with sequence length. Better for variable-length utterances. Source: NeMo, ESPnet.

**Vectorized implementation**: NeMo's `use_vectorized_code=True` generates all masks in parallel on GPU via batched random tensors. 10-50x faster than per-sample loop.

**Source**: All three frameworks. ESPnet has separate `MaskAlongAxis` and `MaskAlongAxisVariableMaxWidth` classes.

**Relevance**: **HIGH** — the single most important ASR augmentation. Already in our plan.

### 2.2 Time Warping

**What**: Warps the time axis of the spectrogram via bicubic interpolation. Selects a random center point and displaces it within a window.

| Param | Typical | Notes |
|-------|---------|-------|
| window | 5 | Max displacement in frames |
| mode | "bicubic" | Interpolation mode |

**Source**: ESPnet (`time_warp.py`). Original SpecAugment paper includes this, but many frameworks skip it for speed.

**Relevance**: **LOW** — often disabled in practice. Slow (per-sample operation), marginal improvement.

### 2.3 SpecCutout (Rectangle Masking)

**What**: Zeroes out random rectangles (both freq AND time simultaneously) in the spectrogram. Unlike SpecAugment which masks entire rows/columns, this masks 2D patches.

| Param | Typical |
|-------|---------|
| rect_masks | 0-2 |
| rect_freq | 20 |
| rect_time | 5 |

**Source**: NeMo (`SpecCutout`). Applied BEFORE SpecAugment freq/time masking.

**Relevance**: **LOW-MEDIUM** — simpler alternative to full SpecAugment. Not commonly used in practice.

### 2.4 SpectrogramDrop with Multiple Replace Modes

**What**: Masks spectrogram regions but offers multiple fill strategies beyond zeroing.

| Replace Mode | Description |
|-------------|-------------|
| `zeros` | Fill with 0 (standard) |
| `mean` | Fill with global spectrogram mean |
| `rand` | Fill with random values between min/max |
| `cutcat` | Fill with chunks from other batch samples (CutMix-like) |
| `swap` | Fill with temporally-shifted content from same sample |
| `random_selection` | Randomly pick from above each call |

**Source**: SpeechBrain (`SpectrogramDrop`).

**Relevance**: **LOW** — interesting research direction but not standard practice.

### 2.5 Random Frequency Shift

**What**: Circular-shifts the spectrogram along the frequency axis. Forces model to not rely on absolute frequency positions.

**Source**: SpeechBrain (`RandomShift`).

**Relevance**: **LOW** — niche technique.

---

## 3. Feature Extraction Augmentations (Built into preprocessing)

### 3.1 Dithering

**What**: Adds small Gaussian noise (`σ = 1e-5`) to waveform before STFT, training only.

```python
if self.training and self.dither > 0:
    x += self.dither * torch.randn_like(x)
```

**Source**: NeMo (`FilterbankFeatures`).

**Relevance**: **HIGH** — nearly free, prevents edge cases in feature extraction, standard practice. Already in our plan.

### 3.2 Narrowband Augmentation

**What**: Randomly zeroes all frequency bins above a cutoff (e.g., 4kHz) to simulate telephone-quality audio. Applied after STFT but before mel filterbank.

| Param | Typical |
|-------|---------|
| max_freq | 4000 Hz |
| prob | 0.1 per sample |

**Source**: NeMo (`FilterbankFeatures.nb_augmentation_prob`).

**Relevance**: **HIGH** — simple and effective for robustness. Already in our plan.

### 3.3 Filterbank Randomization

**What**: During training, randomly perturbs the center frequencies and bandwidths of mel filterbanks by ±N%. Acts as implicit feature augmentation.

| Param | Typical |
|-------|---------|
| param_rand_factor | 0.0-0.15 | ±15% perturbation |

**Source**: SpeechBrain (`Filterbank.param_rand_factor`).

**Relevance**: **MEDIUM** — interesting technique, makes model robust to speaker frequency variation. Low cost.

### 3.4 Preemphasis

**What**: Applies high-pass filter `y[t] = x[t] - 0.97 * x[t-1]` before STFT. Not augmentation per se — standard signal processing that boosts high frequencies.

**Source**: NeMo (always-on). SpeechBrain and ESPnet have it optional.

**Relevance**: **LOW** — Moonshine's preprocessor already has its own feature pipeline. Don't add preemphasis unless changing the frontend.

---

## 4. Online vs Offline Augmentation

| Category | When | Where | Speed | Flexibility |
|----------|------|-------|-------|-------------|
| Waveform (noise, RIR, speed) | DataLoader `__getitem__` | CPU | Slow (librosa, sox) | Changes every epoch |
| Waveform (noise, RIR) | Collate function | CPU | Medium | Batch-level mixing |
| Feature (SpecAugment) | Model `forward()` | GPU | Fast | Every forward pass |
| Feature (dithering, narrowband) | Preprocessor `forward()` | GPU | Fast | Every forward pass |

**Speed perturbation is notably slow** on CPU. NeMo's docstring explicitly warns: "This is a very slow operation for online augmentation." Consider pre-computing speed-perturbed copies and storing as separate manifest entries if training time is critical.

---

## 5. Augmentation Scheduling

### 5.1 Warmup (Delayed Augmentation)

**What**: Don't apply augmentation for the first N optimizer steps. Let the model learn on clean data first, then gradually introduce augmentation.

| Model | Warmup Steps |
|-------|-------------|
| Conformer Large (LibriSpeech) | 8,000 |
| Conformer Large (Loquacious) | 30,000 |
| Conformer XLarge (Loquacious) | 80,000 |

**Source**: SpeechBrain recipes (`augment_warmup` parameter).

**Relevance**: **HIGH** — simple to implement, potentially important for our limited data regime. Without warmup, aggressive augmentation early in training can prevent the model from learning basic patterns.

### 5.2 Random Augmentation Count

**What**: Instead of always applying all augmentations, randomly select 1-N augmentations per batch. Adds stochastic variety.

**Source**: SpeechBrain Loquacious recipe: `min_augmentations: 1, max_augmentations: 3` for 3 available augmentations.

**Relevance**: **MEDIUM** — easy to add if using SpeechBrain's `Augmenter` pattern.

---

## 6. Recommended Augmentation Stack for ru-Moonshine

### Phase 1 (Tiny, 248h, M9-M10)

Priority: low complexity, proven effectiveness.

| Augmentation | Where | Params | Rationale |
|-------------|-------|--------|-----------|
| Speed perturbation | DataLoader | [0.9, 1.0, 1.1] | 3x data multiplier |
| SpecAugment | Model forward | F=2/W=20, T=2/W=40 | Standard, high impact |
| Dithering | Preprocessor | σ=1e-5 | Prevents edge cases |
| Narrowband | Preprocessor | cutoff=4kHz, p=0.1 | Phone robustness |

### Phase 2 (Small, 5.4K hours, M13-M14)

Add environmental robustness:

| Augmentation | Where | Params | Rationale |
|-------------|-------|--------|-----------|
| + Noise addition | DataLoader | MUSAN, SNR 0-20dB, p=0.3 | Real-world noise robustness |
| + RIR convolution | DataLoader | Synthetic RIRs, p=0.3 | Far-field/phone deployment |
| + Augmentation warmup | Training loop | warmup=5000 steps | Learn clean patterns first |

### Phase 3 (Optional, for deployment robustness)

| Augmentation | Where | Params | Rationale |
|-------------|-------|--------|-----------|
| + Gain perturbation | DataLoader | ±10 dB | Volume robustness |
| + Codec degradation | DataLoader | G.711/AMR-NB | Phone call deployment |
| + Time stretch | DataLoader | 0.9-1.1 (pitch-preserving) | Natural speed variation |
| + Filterbank randomization | Preprocessor | ±10% | Speaker variation |

---

## 7. Key Implementation Tips

1. **Force FP32 for feature extraction** — STFT and mel filterbank under AMP cause NaN in fp16. Wrap in `torch.amp.autocast(enabled=False)`. All three frameworks do this.

2. **Speed perturbation is slow** — it runs on CPU in the DataLoader. If it becomes a bottleneck, pre-compute speed-perturbed copies and add to manifest as separate entries.

3. **SpecAugment should use adaptive time masking** — `time_width` as a float (e.g., 0.05 = 5% of sequence) scales masks to utterance length. Fixed-width masks are too large for short utterances and too small for long ones.

4. **Noise/RIR manifest management** — Keep noise and RIR files in a separate directory with their own CSV/JSONL manifests. The DataLoader loads a random noise/RIR sample per training sample. Don't mix noise files into the training manifest.

5. **Augmentation warmup is underappreciated** — For small datasets, starting with clean data and gradually adding augmentation often improves convergence. Implement as a simple step counter check before applying augmentation.

6. **Vectorized SpecAugment** — NeMo's GPU-batched implementation generates all masks in parallel via `torch.rand` tensors. Much faster than per-sample Python loops. Implement using PyTorch tensor operations, not Python for-loops.

7. **Narrowband augmentation is nearly free** — Just zero out frequency bins above a cutoff after STFT. Zero GPU cost, significant robustness gain for phone deployment.

---

## 8. Framework Comparison

| Technique | ESPnet | NeMo | SpeechBrain | In Our Plan |
|-----------|--------|------|-------------|-------------|
| Speed perturbation | ✅ (torchaudio) | ✅ (librosa, slow) | ✅ (torchaudio, pre-created) | ✅ |
| SpecAugment (freq mask) | ✅ | ✅ (vectorized + Numba) | ✅ | ✅ |
| SpecAugment (time mask) | ✅ | ✅ (vectorized + Numba) | ✅ | ✅ |
| SpecAugment (time warp) | ✅ (bicubic) | ❌ | ✅ (bicubic) | ❌ |
| SpecCutout (rectangle) | ❌ | ✅ | ❌ | ❌ |
| Noise addition | ✅ (SNR) | ✅ (SNR + foreground bursts) | ✅ (SNR + pink noise) | ✅ (Phase 2) |
| RIR convolution | ✅ (power renorm) | ✅ (delay compensation) | ✅ (scale factor) | ✅ (Phase 2) |
| Gain perturbation | ❌ | ✅ | ✅ (RandAmp) | ❌ |
| Codec degradation | ❌ (placeholder) | ✅ (sox: G.711, AMR, OGG) | ✅ (torchaudio: μ-law, MP3, G.722) | ❌ |
| DropBitResolution | ❌ | ❌ | ✅ | ❌ |
| Audio clipping | ❌ | ❌ | ✅ (DoClip) | ❌ |
| Time stretch (pitch-preserving) | ✅ (phase vocoder) | ✅ (Numba accelerated) | ❌ | ❌ |
| Narrowband augmentation | ❌ | ✅ (post-STFT) | ❌ | ✅ |
| Dithering | ❌ | ✅ (1e-5 σ) | ❌ | ✅ |
| Filterbank randomization | ❌ | ❌ | ✅ (±N%) | ❌ |
| DropFreq (notch filter) | ❌ | ❌ | ✅ (1-3 bands) | ❌ |
| DropChunk (temporal) | ❌ | ❌ | ✅ (100-1000 samples) | ❌ |
| Phase corruption | ✅ (STFT phase noise) | ❌ | ❌ | ❌ |
| Bandwidth limitation | ✅ (downsample + upsample) | ❌ | ❌ | ❌ |
| Augmentation warmup | ❌ | ❌ | ✅ (5K-80K steps) | ❌ (should add) |
| Foreground noise bursts | ❌ | ✅ (1-5 per utterance) | ❌ | ❌ |
| Pink noise generation | ❌ | ❌ | ✅ (1/f^α) | ❌ |
