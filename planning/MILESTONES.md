# ru-Moonshine: Milestone Execution Plan

Concrete step-by-step plan. Each milestone is a checkpoint with a gate — do not proceed until the gate passes.

**Legend**:
- Machine: where to run it
- Time: wall-clock estimate
- Cost: cloud spend ($0 = local machine)
- Prerequisites: which milestones must be done first
- Gate: what must be true before moving on
- Deliverables: files/artifacts produced

---

## Dependency Graph

```
M1 ──→ M2 ──→ M3 ──→ M4 ──→ M5 ──→ M6 ──→ M6.5 ──→ M7 ──→ M8
                                                                  │
                                      M9 ←───────────────────────┘
                                       │
                                      M10 ──→ M11 ──→ M12 ──→ M13 ──→ M14
                                                                       │
                                                       M15 ←──────────┘
                                                        │
                                                 M16 ←──┘ ← (optional) M17/M18
```

---

## M1: Environment Setup

**Machine**: Local workstation (any OS, no GPU needed)
**Time**: 0.5-1 day
**Cost**: $0
**Prerequisites**: None

### Actions

1. Clone Moonshine repo: `git clone https://github.com/moonshine-ai/moonshine`
2. Create project repo: init `ru-moonshine` with structure:
   ```
   ru-moonshine/
   ├── configs/          # Training configs (YAML)
   ├── data/             # Data manifests, versions.json
   ├── models/           # Model definitions
   ├── training/         # Training loop, data loader
   ├── inference/        # Streaming inference, ONNX export
   ├── tokenizer/        # SentencePiece training + encoding
   ├── tests/            # PoC tests (T1-T18)
   ├── scripts/          # Utility scripts
   ├── Dockerfile        # Pinned environment
   └── requirements.txt  # Pinned dependencies
   ```
3. Create `Dockerfile` with pinned versions:
   - Python 3.11
   - PyTorch 2.x + CUDA 12.x
   - `sentencepiece`, `schedulefree`, `onnxruntime`, `torchaudio`, `wandb`
4. Create `requirements.txt` with exact versions
5. `wandb login` — authenticate Weights & Biases (free individual account). If W&B is unavailable, TensorBoard works as fallback (no setup needed)
6. Verify: `docker build -t ru-moonshine . && docker run --gpus all ru-moonshine python -c "import torch; print(torch.cuda.is_available())"`
6. Copy Moonshine model definitions from HuggingFace `transformers.models.moonshine` into `models/` as starting point
7. Init git repo, commit baseline

### Self-Check

- [ ] `wandb login` succeeds (skip if using TensorBoard only)
- [ ] Docker container builds and sees GPU
- [ ] `import torch; import sentencepiece; import schedulefree` all succeed inside container
- [ ] Moonshine model code is in `models/` and imports without error
- [ ] Git repo initialized with clean structure

### Gate

Docker builds. GPU detected. All imports work.

### Deliverables

- `Dockerfile`, `requirements.txt` (with `wandb`)
- Project structure with Moonshine model code copied
- W&B project created at `wandb.ai/<your-workspace>/ru-moonshine` (if using W&B)
- Fallback: TensorBoard works with `tensorboard --logdir runs/`
- Initial git commit

---

## M2: Russian BPE Tokenizer

**Machine**: Local workstation (CPU only)
**Time**: 0.5 day
**Cost**: $0
**Prerequisites**: M1

### Actions

1. Download transcript text from Russian speech datasets (text only, no audio):
   - Common Voice 21 ru: `artyomboyko/common_voice_21_0_ru` (validated split, 122K sentences)
     - Download parquet files directly via `huggingface_hub`, extract `sentence` column only
     - CV19 requires license acceptance on their website and isn't on HF; CV21 is the accessible equivalent
   - Russian LibriSpeech: `istupakov/russian_librispeech` (57K sentences, extract `text` column from parquet)
   - **Note**: Golos (`SberDevices/Golos`) is empty on HuggingFace (no data files). Needs manual download from Sber sources in M3. Omit from tokenizer corpus for now.
2. Compile list of top English loanwords (~1100) as they appear in Russian text. Append to corpus.
3. Text normalization for tokenizer training: lowercase, strip punctuation, deduplicate
4. Train SentencePiece BPE tokenizer with Russian-optimized settings:
   ```python
   spm.SentencePieceTrainer.train(
       input='data/tokenizer_corpus.txt',
       model_prefix='data/tokenizer_256',
       vocab_size=256,
       model_type='bpe',
       character_coverage=1.0,
       split_digits=True,                    # standalone digits as separate tokens
       split_by_unicode_script=False,        # prevent Latin/Cyrillic boundary splits for loanwords
       max_sentencepiece_length=24,          # allow longer merge pieces for Russian morphology
       user_defined_symbols=['<blank>', '<sos/eos>', '<pad>', '<COMMA>', '<PERIOD>', '<QUESTION>',
                             '0','1','2','3','4','5','6','7','8','9'],  # explicit digit tokens
   )
   ```
5. Repeat for vocab 512 and 1024 (for later comparison)
6. Run **T12: Tokenizer roundtrip test**:
   - Encode → decode 1000 Russian sentences
   - Test: normal text, numbers, hyphenated words, abbreviations (США, МГУ), dates ("1 мая"), English loanwords
   - All must roundtrip exactly
7. Run **T13: Tokenizer morphology coverage**:
   - Measure avg tokens/word on 10K sentences
   - Target: ≤ 4.0 (vocab 256), ≤ 3.5 (vocab 512), ≤ 3.0 (vocab 1024)
   - Measure fragmentation of top 100 English loanwords
   - **Rationale for adjusted targets**: Russian has 33-letter Cyrillic alphabet + rich morphology. At vocab 256, after 33 letters + 10 digits + 6 specials = 49 base tokens, only ~207 remain for BPE merges. Original English-based targets (≤3.0 at 256, ≤2.0 at 1024) are unachievable for Russian. GigaAM achieves 8.4% WER with 256 vocab, proving accuracy is not limited by vocab size.
8. Build abbreviation dictionary: collect ~300 common Russian abbreviations with spoken forms

### Self-Check

- [x] Tokenizer files exist: `data/tokenizer_256.model`, `data/tokenizer_256.vocab`
- [x] T12: 1475/1475 sentences roundtrip correctly (all vocab sizes)
- [x] T13: avg tokens/word ≤ 4.0 (vocab 256: 3.73), ≤ 3.5 (vocab 512: 3.08), ≤ 3.0 (vocab 1024: 2.55)
- [x] English loanwords fragment into ≤ 4.0 tokens average (256: 4.00, 512: 3.57, 1024: 3.32)
- [x] Abbreviation dictionary exists (318 entries)

### Gate

T12 passes (exact roundtrip). T13 passes (adjusted targets met).

### Deliverables

- `data/tokenizer_256.model`, `data/tokenizer_256.vocab` (and 512, 1024 variants)
- `data/abbreviations.json` (318 abbreviation → spoken form mappings)
- `data/english_loanwords.txt` (1131 loanwords)
- `scripts/build_tokenizer_corpus.py`, `scripts/train_tokenizer.py`
- `tests/test_t12_roundtrip.py`, `tests/test_t13_morphology.py`
- T12 and T13 test results documented

### Actual Results

| Vocab | Tokens/word | Loanword frag | Roundtrip |
|-------|------------|---------------|-----------|
| 256   | 3.73       | 4.00          | 1475/1475 |
| 512   | 3.08       | 3.57          | 1475/1475 |
| 1024  | 2.55       | 3.32          | 1475/1475 |

Corpus: 103K unique sentences (CV21 ru: 122K, RuLS: 57K, deduplicated).

---

## M3: Data Pipeline

**Machine**: Local workstation + 3090 (for VAD/Splitting speed)
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M2

### Actions

1. Download all datasets (see `planning/DATASETS.md` for full availability tracker):
   - Common Voice 21 ru: `artyomboyko/common_voice_21_0_ru` on HuggingFace (validated split has 122K clips; CV19 requires license acceptance on their website and isn't directly on HF — CV21 is the accessible equivalent)
   - Golos: `SberDevices/Golos` is empty on HuggingFace. Download from Sber's original source (e.g. `bond005/sberdevices_golos_*` subsets, or manual from sber.ru). ~1,240 hours when obtained.
   - MLS ru: `facebook/multilingual_librispeech` — **Note**: Russian config doesn't exist in this dataset. Available configs: dutch, french, german, italian, polish, portuguese, spanish. Omit or find alternative.
   - RuLS: `istupakov/russian_librispeech` on HuggingFace (~98 hours)
   - SOVA: from source, pin commit hash
   - **Data stored on external drive**: `data/` is symlinked to `/media/smileijp/5C40E2C140E2A0CE/voice/data` (363GB free)
2. Create `data/versions.json` manifest:
   ```json
   {
     "common_voice": {"version": "21.0", "source": "artyomboyko/common_voice_21_0_ru", "downloaded": "2026-XX-XX"},
     "russian_librispeech": {"source": "istupakov/russian_librispeech"},
     "golos": {"source": "manual", "note": "not on HuggingFace, requires separate download"},
     ...
   }
   ```
3. Preprocess each dataset (save to `data/processed/`):
   - Resample to 16kHz mono WAV
   - Split long audio at silence (>300ms low energy): min 1s, max 30s segments
   - Discard segments < 1s after splitting
   - VAD filter: remove clips with > 50% silence
   - Duration filter: keep 1-30s clips
   - Text normalization:
     - Lowercase
     - Numbers → words via `num2words` (with `lang='ru'`)
     - Abbreviation expansion using dictionary from M2
     - Remove non-speech markers
     - Keep hyphenated words as single tokens
4. Deduplicate across datasets (exact text match + audio duration match)
5. Split: 95% train / 5% validation per dataset, by speaker ID (for CV, RuLS)
6. Create manifests (JSON lines): `data/manifests/train.jsonl`, `val.jsonl`, `test.jsonl`
   - Each line: `{"audio_path": "...", "text": "...", "duration": 5.2, "dataset": "cv21", "speaker_id": "..."}`
7. Download MUSAN corpus (noise + music + babble) and generate synthetic RIRs using Kaldi's RIR generator. Save to `data/augmentation/`
 8. Build PyTorch `Dataset` class that reads manifests, loads audio on-the-fly, applies augmentation (SpecAugment, speed perturbation, MUSAN noise, RIR — configurable via YAML)
    - **Vectorized SpecAugment**: generate all masks in parallel on GPU via batched random tensors, not per-sample Python loops. 10-50x faster. Source: NeMo
    - **Adaptive time masking**: `time_width` as float (e.g., 0.05 = 5% of seq length) instead of fixed frame count. Scales masks to utterance length. Source: NeMo, ESPnet
    - **Augmentation warmup**: skip all augmentation for first N optimizer steps (configurable, default 5000). Only apply when `global_step >= warmup_steps`. Critical for small data regime. Source: SpeechBrain
    - **Balanced augmentation**: when running multiple augmentations in parallel, fix total batch size so augmented data doesn't overwhelm original (`parallel_augment_fixed_bs`). Source: SpeechBrain
 9. Verify data loader: iterate 100 batches, check shapes, no crashes, no NaN

### Self-Check

- [ ] `data/versions.json` exists with all datasets pinned
- [ ] `data/manifests/train.jsonl` has ~5.4K hours of entries
- [ ] No NaN/Inf in audio features from data loader
- [ ] Batch shapes correct: `(B, T)`, `(B, L)` for audio and text
- [ ] Augmentation runs without error on 100 batches
- [ ] Speaker IDs in train and val do not overlap (within each dataset)

### Gate

Data loader produces correct shapes for 100 consecutive batches. Train manifest has expected hours. No speaker overlap within datasets.

### Deliverables

- `data/versions.json`
- `data/manifests/{train,val,test}.jsonl`
- `data/augmentation/` (MUSAN + RIRs)
- `training/dataset.py` (PyTorch Dataset + DataLoader)
- Preprocessing scripts in `scripts/`

---

## M4: Model Architecture (v2 + v2.1)

**Machine**: Local 3090 (GPU needed for forward/backward)
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M1, M2

### Actions

 1. Implement v2 architecture (Tiny variant first, then parameterize for Small/Medium):
    - `models/preprocessor.py` — copy from Moonshine, adapt for standalone use. **Forced FP32 for STFT/mel** (wrap in `torch.amp.autocast(enabled=False)` to prevent NaN under AMP). Source: NeMo, SpeechBrain
    - `models/encoder.py` — sliding-window Transformer encoder with configurable window sizes
    - `models/decoder.py` — causal Transformer decoder with RoPE, cross-attention, SwiGLU
    - `models/adapter.py` — positional embedding + linear projection
    - `models/model.py` — full model: preprocessor → encoder → adapter → decoder + CTC head
    - Configurable via YAML: `d_model`, `num_layers`, `num_heads`, `ffn_dim`, `vocab_size`, `window_size`
    - **Forward contract** (from ESPnet pattern): `forward()` returns `(loss, stats_dict, batch_weight)`. Training loop logs everything from `stats` generically — no model-specific logging code needed
    - **Attention**: use `torch.nn.functional.scaled_dot_product_attention` (PyTorch 2.0+). Auto-selects Flash/memory-efficient kernel. Manual fallback for debugging. Source: ESPnet, NeMo
    - **Subsampling mask propagation**: preprocessor's conv2d subsampling must adjust the attention mask (`mask[:, :, :-2:2]` for stride-2 conv with kernel 3). Missing this is a common bug. Source: ESPnet
    - **Minimum audio length check**: reject or pad audio shorter than subsampling requires (e.g., 7 frames for 4x subsampling). Source: ESPnet `TooShortUttError`
    - **Weight initialization** (from ESPnet pattern): xavier_uniform for dim>1 params, zero for biases, `reset_parameters()` for LayerNorm/Embedding, small init for output projection
    - **QK normalization** (optional): LayerNorm on Q/K before attention scores. Configurable via `qk_norm: true`. Source: ESPnet
 2. Implement v2.1 encoder additions in `models/encoder_v21.py`:
    - CausalDepthwiseConv module (kernel=7, causal padding)
    - Multi-scale U-Net: causal stride-2 downsample (learned conv, kernel=2, stride=2, left-only padding), nearest-neighbor upsample + 1x1 conv
    - Skip connections between stages
    - SSC-style cross-window attention mask generation
 3. Implement stochastic depth (layer drop) in encoder forward:
    - Per-layer drop probability that linearly increases from 0 to `max_drop_rate` (default 0.1)
    - During training: randomly skip layers with probability `p`, scale surviving layers by `1/(1-p)`
    - During inference: use all layers (standard)
    - Source: ESPnet (conformer_encoder.py), NeMo (conformer_encoder.py:511-513)
 4. Implement streaming encoder inference in `inference/streaming_encoder.py`:
    - Circular KV-cache buffer per layer (or per-stage for v2.1)
    - Incremental forward: new query × cached KV
    - **StreamPositionalEncoding**: use `start_idx` offset for correct absolute positions in each chunk (not from 0). Source: ESPnet
    - **Repetition detection**: suppress if last N emitted tokens are identical (N=4). Source: ESPnet
    - **Hold-N mechanism**: hold back N tokens between chunks for revision with more context. Source: ESPnet
    - **Hallucination detection**: 3 pattern detectors (4+ identical, alternating pair, repeating triple). Source: NeMo
4. Run **T1: Forward pass smoke test**:
   - Random audio (16kHz, 5s) → full model → logits
   - Check: no crash, no NaN, logits shape = `(seq_len, vocab_size)` (256 for Tiny, 512 for Small)
   - Run for v2 Tiny, v2.1 Tiny
5. Run **T2: Backward pass test**:
   - Forward + backward on random batch
   - Check: all trainable params have `grad` not None, not NaN
   - Gradient norm finite (< 1e6)
6. Run **T3: Sliding-window mask test**:
   - For position t with window (16,0): verify mask allows t-16..t, blocks t+1..end
   - For (16,4): verify t-16..t+4
7. Run **T4: Feature extraction test**:
   - 5s of 16kHz audio (80,000 samples) → preprocessor → shape (250, enc_dim)
   - No NaN/Inf in features

### Self-Check

- [ ] T1: logits shape correct, no NaN — for both v2 and v2.1 Tiny
- [ ] T2: all grads exist, finite norms — for both v2 and v2.1 Tiny
- [ ] T3: attention masks correct for (16,0) and (16,4)
- [ ] T4: output shape (250, enc_dim), no NaN/Inf

### Gate

**T1 and T2 pass for both v2 and v2.1 Tiny.** If T1/T2 fail = architecture bug. Fix before proceeding.

### Deliverables

- `models/` — full model code for v2 and v2.1
- `inference/streaming_encoder.py` — cache-aware incremental encoder
- `configs/v2_tiny.yaml`, `configs/v21_tiny.yaml` — model configs
- T1-T4 test results

---

## M5: Training Loop

**Machine**: Local 3090
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M3, M4

### Actions

1. Implement `training/train.py`:
    - Main training loop with gradient accumulation (divide loss by `accum_steps` before backward)
    - Joint loss: `L = L_AED + α * L_CTC` (α configurable, default 0.3)
    - Label smoothing ε=0.1 on decoder loss
    - Dropout: attention 0.1, FFN 0.1, drop path 0.1 (v2.1 only)
    - Gradient clipping (norm 5.0)
    - **Non-finite gradient handling**: after `loss.backward()`, check `grad_norm`. If inf/nan: zero gradients, skip optimizer step, log warning. Abort after 5 consecutive bad steps (patience counter). Source: all three major ASR frameworks (ESPnet, NeMo, SpeechBrain)
    - **`optimizer.zero_grad(set_to_none=True)`**: releases gradient memory instead of zeroing. Saves ~20% peak gradient buffer memory. Source: SpeechBrain
 2. Implement optimizer + scheduler:
    - Primary: Schedule-Free (`AdamWScheduleFree`, `optimizer.train()`/`optimizer.eval()` alongside model)
    - Fallback: AdamW with warmup + cosine decay (configurable via `optimizer.name`). Warmup for N steps, cosine decay to 10% of peak. Source: standard in ESPnet, NeMo, SpeechBrain
  3. Implement validation loop:
     - Every 2K steps: CTC greedy decoding on validation set → WER
     - WER aggregation: numerator/denominator pattern (sum edit distances / sum reference word counts, aggregated across batches). Source: NeMo, ESPnet
     - Log: AED loss, CTC loss, total loss, WER, CER, SER, learning rate, gradient norm, clipping indicator, loss scale, skipped steps
     - **ErrorRateStats pattern** (Source: SpeechBrain): per-utterance tracking of insertions/deletions/substitutions, alignment ops. SER = 100 * erroneous_sents / scored_sents. CER via same class with `merge_tokens=False`.
    - Dual-backend logging: W&B (default) or TensorBoard, selected via `logging.backend` in config YAML
    - Thin `Logger` wrapper: same `log(metrics, step)` interface, calls wandb or SummaryWriter underneath
 4. Implement logger (`training/logger.py`):
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
 5. Implement checkpointing (`training/checkpoint.py`):
    - Save every 5K steps (intra-epoch, not just epoch boundaries): model, optimizer, scheduler, step count, best WER
    - Retain top-5 by val WER + latest
    - Auto-resume from latest checkpoint
    - **Checkpoint averaging**: function to average top-N checkpoint state_dicts for final evaluation. ~0.5-1% WER improvement for free. Source: ESPnet, SpeechBrain
    - **Preemption handling**: catch SIGTERM, save checkpoint before exit. Source: NeMo
 6. Implement bucket-shuffle batch sampling (`training/sampler.py`):
    - Sort training samples into N buckets (e.g., 100) by audio duration
    - Shuffle within each bucket
    - Concatenate buckets in order → batch by fixed count
    - Result: ~30-50% less padding waste, stable GPU memory, no OOM from outliers
    - Source: NeMo (semi-sorted batching), SpeechBrain (DynamicBatchSampler)
 7. Implement SentencePiece lazy loading in tokenizer:
    - Defer `SentencePieceProcessor` creation to first `encode()` call
    - Prevents pickle failures with `DataLoader(num_workers > 0)`
    - Source: ESPnet
 8. Implement dynamic chunk training:
    - Per batch: sample `w_left ∈ [8, 16]`, `w_right ∈ [0, 4]`
 9. Create training config YAML schema:
   ```yaml
    model: {name: v2_tiny, d_model: 320, num_layers: 6, ...}
    data: {train_manifest: data/manifests/train.jsonl, ...}
     training: {epochs: 3, batch_size: 16, accum_steps: 8, lr: 2e-3, grad_clip: 5.0, precision: fp16, ...}
     loss: {ctc_weight: 0.3, label_smoothing: 0.1, interctc_layers: [], interctc_weight: 0.2, ctc_zero_infinity: true}
     augmentation: {spec_augment: true, speed_perturbation: true, musan_noise: true, narrowband: {enabled: true, max_freq: 4000, prob: 0.1}, warmup_steps: 5000, adaptive_time_mask: true, min_aug: 1, max_aug: 3, parallel_fixed_bs: true}
    batching: {strategy: bucket_shuffle, num_buckets: 100}

    checkpointing: {save_every: 5000, keep_top: 5, eval_every: 2000, avg_top_n: 5}
    optimizer: {name: schedulefree, lr: 2e-3}
    logging: {backend: wandb, project: ru-moonshine, name: v2-tiny-phase1, log_every: 100, eval_every: 2000}
    ```
10. Run **T7: CTC head sanity**:
    - Random encoder output + real transcript → CTC loss finite, positive, decreases over 50 steps on CTC head alone
11. Run **T8: Joint loss convergence**:
    - Train 500 steps with joint loss on 100 clips
    - Both losses decrease. No NaN.

### Self-Check

- [ ] Training loop runs for 100 steps on random data without error
- [ ] Non-finite gradient handling: inject NaN loss, verify step is skipped, training continues
- [ ] Logger shows metrics (W&B dashboard at wandb.ai, or `tensorboard --logdir runs/`)
- [ ] Switching `logging.backend: tensorboard` works without code changes
- [ ] Checkpoint saves and loads correctly
- [ ] Gradient accumulation produces correct effective batch size
- [ ] Bucket-shuffle sampler groups similar-length samples together
- [ ] Preemption: SIGTERM during training → checkpoint saved → resume succeeds
- [ ] T7: CTC loss finite, positive, decreasing
- [ ] T8: Both AED and CTC losses decrease, no NaN

### Gate

**T7 and T8 pass.** Joint loss converges. If T7 fails = CTC bug. If T8 fails = loss weighting issue.

### Deliverables

- `training/train.py` — full training loop
- `training/validate.py` — WER evaluation
- `training/checkpoint.py` — save/load/resume/average
- `training/logger.py` — dual-backend logger (wandb / tensorboard, config-selected)
- `training/sampler.py` — bucket-shuffle batch sampler
- `configs/train_v2_tiny.yaml` — example training config

---

## M6: Overfit Sanity Check

**Machine**: Local 3090
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M5

### Actions

1. Download a small subset of real Russian audio (10 clips from Common Voice ru). Manually verify transcripts are correct.
2. Run **T5: Overfit 10 samples**:
   - Train v2 Tiny for 500 steps on these 10 clips
   - Loss → near 0. WER → 0 on same clips
   - If this fails: debug loss function, data loading, or gradient flow. Do NOT proceed
3. Download 100 Russian clips. Run **T6: Overfit 100 samples**:
   - Train v2 Tiny for 2000 steps on 100 clips
   - WER < 5% on same clips
   - If this fails: check tokenizer first (re-run T12). If tokenizer passes, increase model capacity or debug convergence
4. Repeat T5 + T6 for v2.1 Tiny
5. Run **T17: GPU memory profiling**:
   - Profile peak VRAM: Tiny batch=32 (3090), Small batch=8 (3090)
   - Record actual numbers. Adjust batch sizes if needed

### Self-Check

- [ ] T5: WER = 0 on 10 overfit clips (v2 and v2.1)
- [ ] T6: WER < 5% on 100 overfit clips (v2 and v2.1)
- [ ] T17: Memory profiled, batch sizes confirmed

### Gate

**T5 passes = model can learn.** If T5 fails, do not proceed — fundamental issue. T6 passes = tokenizer works.

### Deliverables

- Overfit checkpoints (v2 + v2.1 Tiny)
- GPU memory profile results
- T5, T6, T17 test reports

---

## M6.5: Data Quality Assessment (Model-Based)

**Machine**: Local 3090 (need trained model from M6)
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M6

### Rationale

After M6, we have a trained model that can transcribe Russian speech. This model is the best tool for evaluating training data quality — far more reliable than regex heuristics which produce false positives. This milestone uses the model to score its own training data and identify low-quality sources before scaling up in M8.

### Actions

1. **Run inference on all training data** — use the M6 overfitted model to transcribe all 156K train clips
2. **Compute WER per source** — calculate word error rate between model output and ground truth transcript, grouped by dataset (cv21, ruls) and by CV21 speaker
3. **Source-level ranking** — rank sources by (error_rate × size), following Whisper's post-training filtering methodology
4. **Manual inspection** — listen to 20 clips from the worst-ranked sources to determine if errors are from (a) bad transcripts, (b) bad audio, or (c) model weakness
5. **Remove confirmed bad sources** — only remove entire sources after manual verification, never individual samples
 6. **Audio-transcript alignment check** — flag entries where duration vs text length are wildly mismatched (>2σ from mean chars/second)
 7. **Top-K worst utterances** — identify the 50 utterances with highest WER for targeted inspection. Group by speaker to find problematic speakers. Source: SpeechBrain `top_wer_utts` / `top_wer_spks`
 8. **Spot-check 50 random samples** from cleaned data for quality verification

### Gates

- [ ] WER distribution plotted per dataset and per CV21 speaker
- [ ] Bottom 5% of sources manually inspected and labeled (bad data / model weakness)
- [ ] No more than 5% of training data removed
- [ ] 50-sample spot-check: ≤2 samples flagged (≤4% error rate)

### Deliverables

- `scripts/evaluate_training_data.py` — model-based data quality scorer
- `data/quality_report.json` — per-source WER, flags, removal decisions
- Updated `data/manifests/{train,val}.jsonl` with confirmed bad sources removed
- Quality report document in `planning/`

---

## M7: Streaming Correctness + ONNX Export

**Machine**: Local 3090
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M6, M6.5

### Actions

1. Run **T9: Streaming vs non-streaming parity**:
    - Same audio: (a) encode all frames at once, (b) encode chunk-by-chunk (chunk=32 frames) with state carry-over
    - Cosine similarity of encoder outputs > 0.99 on overlapping frames
    - **Dependency matrix inference**: for each input frame, randomize it and check which output frames change. Produces Boolean matrix `[input_frame, output_frame]`. Verifies no accidental future-frame leakage. Source: SpeechBrain `infer_dependency_matrix`
    - Test for both v2 and v2.1
2. Run **T10: TTFT is bounded**:
   - Measure encoder forward time for 1s, 5s, 10s, 30s audio
   - TTFT must be constant (±10%). If it grows linearly, attention is not windowed
3. Run **T11: KV cache correctness**:
   - Encoder with KV cache (incremental) vs without cache (full recomputation)
   - Logits diff < 1e-5 for 5s and 30s audio
 4. Run **T14: ONNX export smoke test**:
    - `torch.onnx.export` encoder and decoder separately (NeMo pattern: split subgraphs)
    - **Module replacement before export**: replace Flash Attention / SDPA with manual attention, any custom ops with standard equivalents. Source: NeMo `export_utils.py`
    - Load in ONNX Runtime, run inference
    - Output diff < 1e-4 vs PyTorch
5. Run **T15: ONNX streaming test**:
   - Export encoder with KV cache inputs/outputs
   - Chunk-by-chunk ONNX inference
   - Compare to PyTorch streaming: diff < 1e-4
6. Run **T18: INT8 quantization sanity** (use overfit checkpoint from M6):
   - Export FP32 and INT8 ONNX
   - Run on 100 test clips
   - WER diff < 1%

### Self-Check

- [ ] T9: cosine similarity > 0.99 (v2 and v2.1)
- [ ] T9: dependency matrix shows no future-frame leakage (upper triangle = all False)
- [ ] T10: TTFT constant ±10% across 1-30s audio
- [ ] T11: logits diff < 1e-5 with and without cache
- [ ] T14: ONNX output diff < 1e-4 vs PyTorch
- [ ] T15: ONNX streaming diff < 1e-4
- [ ] T18: INT8 WER diff < 1%

### Gate

**T9 passes = streaming works.** T14 passes = can export. T9 failure = attention mask bug. T14 failure = fix before deployment.

### Deliverables

- `inference/streaming_inference.py` — working streaming pipeline
- `inference/onnx_export.py` — ONNX export script
- Streaming encoder with KV cache (v2 + v2.1)
- ONNX models (FP32 + INT8) for v2 Tiny and v2.1 Tiny

---

## M8: Scale-Up Readiness

**Machine**: Local 3090
**Time**: 2-3 days
**Cost**: $0
**Prerequisites**: M7

### Actions

1. Prepare Phase 1 data: subset ~700 hours from CV21 ru (500h) + Golos (200h)
2. Create training config `configs/phase1_v2_tiny.yaml`:
   ```yaml
   model: {name: v2_tiny, d_model: 320, num_layers: 6, num_heads: 4, ffn_dim: 1280, vocab_size: 256}
   data: {train_manifest: data/manifests/phase1_train.jsonl, val_manifest: data/manifests/phase1_val.jsonl}
   training: {epochs: 3, batch_size: 16, accum_steps: 8, lr: 2e-3, warmup_steps: 1000}
   loss: {ctc_weight: 0.3, label_smoothing: 0.1}
   augmentation: {spec_augment: true, speed_perturbation: [0.9, 1.0, 1.1], musan_noise: {enabled: true, snr: [0, 20], prob: 0.3}}
   regularization: {attention_dropout: 0.1, ffn_dropout: 0.1}
   checkpointing: {save_every: 2000, keep_top: 3, eval_every: 2000}
   optimizer: {name: schedulefree, lr: 2e-3}
   logging: {backend: wandb, project: ru-moonshine, name: v2-tiny-phase1, log_every: 100, eval_every: 2000}
   ```
3. Run **T16: 100-hour convergence test**:
   - Train 1 epoch on 100h subset of CV21 Russian
   - Loss decreases monotonically
   - WER < 30% after 1 epoch
4. If T16 fails: tune LR, check data quality, reduce CTC weight. Do NOT spend cloud budget
5. If T16 passes: run full Phase 1 data preparation (700h manifests)

### Self-Check

- [ ] T16: loss decreases monotonically over 1 epoch
- [ ] T16: WER < 30% after 1 epoch
- [ ] Phase 1 training config created and validated

### Gate

**T16 passes = pipeline works at real scale.** Ready for Phase 1 full training. If T16 fails = not ready, debug first.

### Deliverables

- `data/manifests/phase1_train.jsonl`, `phase1_val.jsonl` (700h)
- `configs/phase1_v2_tiny.yaml`, `configs/phase1_v21_tiny.yaml`
- T16 convergence report (loss curve, WER curve)
- 100-hour checkpoint

---

## M9: Phase 1 — v2 Tiny Training

**Machine**: Local RTX 3090 (24GB)
**Time**: 2-3 days
**Cost**: $0
**Prerequisites**: M8

### Actions

1. Start training: `python training/train.py --config configs/phase1_v2_tiny.yaml`
2. Monitor via W&B dashboard (wandb.ai/your-workspace/ru-moonshine) or TensorBoard (`tensorboard --logdir runs/`):
   - **Loss curves**: AED loss, CTC loss, total loss (should all decrease)
   - **Validation WER**: every 2K steps (should decrease, target: converge < 25%)
   - **System metrics** (W&B only): GPU utilization, VRAM usage, GPU temperature, power draw
   - **Gradient norm**: should stay < 100, no spikes
 3. Wait for convergence (3 epochs or early stop)
 4. Select best checkpoint by validation WER
 5. **Checkpoint averaging**: average top-5 checkpoints by val WER, evaluate averaged model
 6. Evaluate best + averaged checkpoints:
    - Greedy WER on validation set
    - CTC prefix beam search WER (separate blank/non-blank probabilities). Source: SpeechBrain
    - AED beam search (width 5) WER on validation set
    - Non-streaming WER (full attention)
    - Streaming WER (chunk-by-chunk encoder)
    - Per-dataset WER (CV vs Golos)
    - **Confidence estimation**: word-level max_prob confidence scores on 500 validation samples. Source: NeMo
6. Export to ONNX (FP32 + INT8). Run T14/T15 again on final model
7. Measure streaming TTFT on CPU (MacBook or local)
 8. Run error analysis on 500 validation samples (categorize errors per type from plan Section 10):
    - **Per-token confusion matrix**: build `ClassificationStats` for grapheme-level error analysis (which characters are most confused). Source: SpeechBrain
    - **CTC posterior trimming**: optionally trim encoder frames where CTC blank prob > 0.95 (5-frame tolerance). Measure decoder speedup vs WER impact. Source: ESPnet
 9. Visualize encoder attention patterns on 10 utterances (check for attention sinks)

### Self-Check

- [ ] Training converges (loss plateaus, WER stops improving)
- [ ] Validation WER < 25% (Tiny on 700h is a baseline, anything < 25% is fine)
- [ ] ONNX export works (FP32 + INT8)
- [ ] Streaming TTFT < 300ms on CPU
- [ ] INT8 WER diff < 1% vs FP32

### Gate

**Training converges.** ONNX export works. Streaming works. Specific WER is not gated — this is Tiny on limited data.

### Deliverables

- Best v2 Tiny checkpoint
- v2 Tiny ONNX models (FP32 + INT8)
- Evaluation report: greedy WER, beam WER, streaming WER, non-streaming WER, CER, SER, TTFT
- Error analysis report (v2 Tiny): per-utterance insertions/deletions/substitutions breakdown

---

## M10: Phase 1 — v2.1 Tiny Training

**Machine**: Local RTX 3090 (24GB)
**Time**: 2-3 days
**Cost**: $0
**Prerequisites**: M9

### Actions

1. Start training: `python training/train.py --config configs/phase1_v21_tiny.yaml`
2. Same monitoring as M9 (W&B or TensorBoard — compare runs side-by-side)
3. Select best checkpoint by validation WER
4. Same evaluation as M9
5. Compare v2 vs v2.1:
   - Side-by-side WER comparison
   - Does v2.1 improve over v2? If yes, by how much?
   - Streaming latency comparison
   - FLOPs comparison (measure actual inference time)
6. If v2.1 does NOT improve over v2: note it, proceed anyway — Small scale may show different behavior

### Self-Check

- [ ] Training converges
- [ ] ONNX export works (v2.1 may need custom export for multi-scale)
- [ ] v2 vs v2.1 WER comparison documented

### Gate

**Both v2 and v2.1 Tiny models trained and evaluated.** Can now decide on ablation.

### Deliverables

- Best v2.1 Tiny checkpoint
- v2.1 Tiny ONNX models (FP32 + INT8)
- v2 vs v2.1 comparison report
- Error analysis report (v2.1 Tiny)

---

## M11: Ablation + Iterate

**Machine**: Local RTX 3090 (24GB)
**Time**: 3-5 days
**Cost**: $0
**Prerequisites**: M10

### Actions

1. Run ablation variants (each is 2-3 days on 3090 with 700h data — run sequentially):

   | Run | Config | Purpose |
   |-----|--------|---------|
   | A | v2 baseline (already done — reuse M9) | Control |
   | B | v2 + causal conv only | Isolate conv contribution |
   | C | v2 + multi-scale only | Isolate multi-scale contribution |
   | D | v2 + conv + multi-scale (no SSC) | Combined without SSC |
   | E | v2.1 full (already done — reuse M10) | Full stack |

2. Compare all variants on validation set. Determine:
   - Which changes help? Which don't?
   - Is the improvement from conv, multi-scale, SSC, or all three?
   - If B ≈ E → drop multi-scale and SSC (they add complexity without gain)
3. Tune hyperparameters based on best variant:
   - LR: try 1e-3, 2e-3, 3e-3
   - CTC weight α: try 0.1, 0.3, 0.5
   - Dropout: try 0.0, 0.1, 0.2
   - Run each for 1 epoch only (quick comparison)
4. Run error analysis on best variant: compare error profiles with M9/M10
5. Decide: which architecture (v2 or v2.1, with or without which components) goes to Phase 2

### Self-Check

- [ ] All 5 ablation variants complete
- [ ] WER comparison table populated
- [ ] Best hyperparameters identified
- [ ] Decision made: v2 or v2.1 for Phase 2

### Gate

**Ablation complete. Architecture decision made. HP ranges narrowed.**

### Deliverables

- Ablation results table (WER for each variant)
- HP tuning results
- Architecture decision document (v2 or v2.1, which components, which HP range)
- Recommended config for Phase 2

---

## M12: Phase 2 — Hyperparameter Search

**Machine**: Cloud H100 (e.g., Lambda Labs, RunPod, or GCP)
**Time**: 1-2 days
**Cost**: ~$50-80
**Prerequisites**: M11

### Actions

1. Set up cloud environment:
   - Launch H100 instance with Docker image from M1
   - `wandb login` on cloud instance (if using W&B — same account, all runs in one dashboard)
   - If W&B unavailable on cloud: set `logging.backend: tensorboard` in config, use `ssh -L 6006:localhost:6006` to view
   - Upload code, tokenizer, data manifests (not raw audio — download on cloud)
   - Download raw audio on cloud instance (faster than uploading)
   - Verify data pipeline works on cloud
2. Prepare 1K-hour subset for HP search (speed up iteration)
3. Run 2-3 HP variants (use v2 Small for speed — apply best to v2.1):
   ```
   Run 1: LR=1e-4, epochs=3, ctc_weight=0.3
   Run 2: LR=3e-4, epochs=3, ctc_weight=0.3
   Run 3: LR=3e-4, epochs=5, ctc_weight=0.5
   ```
   Each run: ~4-8 hours on H100 with 1K hours
4. Evaluate each run: validation WER, loss curves, convergence speed
   - W&B: overlay HP runs in dashboard. TensorBoard: `tensorboard --logdir_spec hp1:runs/hp1,hp2:runs/hp2`
5. Select best HP config

### Self-Check

- [ ] Cloud environment works (data loads, training runs, checkpoints save)
- [ ] All HP variants complete without OOM or NaN
- [ ] Best HP identified (lowest validation WER)

### Gate

**Best HP found. Cloud environment validated for full training.**

### Deliverables

- HP search results table
- Best HP config: `configs/phase2_best.yaml`
- Cloud setup documentation (instance type, Docker command, data transfer steps)

---

## M13: Phase 2 — Production Training (v2 Small)

**Machine**: Cloud H100
**Time**: 2-3 days
**Cost**: ~$100-150
**Prerequisites**: M12

### Actions

1. Prepare full 5.4K-hour data on cloud instance
2. Start training: `python training/train.py --config configs/phase2_v2_small.yaml`
   ```
   model: {name: v2_small, d_model: 620, enc_layers: 10, dec_layers: 10, enc_dim: 620, dec_dim: 512, vocab_size: 512}
   data: {train_manifest: data/manifests/train.jsonl}  # full 5.4K hours
   training: {epochs: best_from_M12, batch_size: 16, accum_steps: 8, lr: best_from_M12}
   loss: {ctc_weight: best_from_M12, label_smoothing: 0.1}
   regularization: {attention_dropout: 0.1, ffn_dropout: 0.1}
   augmentation: {spec_augment: true, speed_perturbation: true, musan_noise: true, rir: true, narrowband: true}
   ```
3. Monitor closely for first 2K steps (loss should decrease, no NaN, no explosion)
   - W&B: watch from laptop, no SSH needed. TensorBoard: `ssh -L 6006:localhost:6006`
   - Check: GPU utilization > 90% (if not, data loading is bottleneck)
   - Check: VRAM usage stable (if growing, memory leak)
4. Let run to completion (3-5 epochs)
5. Select best checkpoint by validation WER
6. Full evaluation:
   - Greedy WER on validation + locked test set (CV21 test + Golos test)
   - Beam search + LM fusion WER (train KenLM on Russian Wikipedia, use for shallow fusion)
   - Streaming WER vs non-streaming WER
   - CER, G2P-normalized WER
   - Streaming TTFT on cloud CPU
   - ONNX INT8 model size
7. Export: PyTorch checkpoint + ONNX FP32 + ONNX INT8
8. Download all artifacts to local machine
9. Run error analysis on 500+ test samples

### Self-Check

- [ ] Training converges (val WER plateaus)
- [ ] v2 Small greedy WER on test < 15% (realistic target)
- [ ] Beam+LM WER improves over greedy by 10-15% relative
- [ ] ONNX INT8 < 200MB
- [ ] Streaming TTFT < 100ms on MacBook CPU
- [ ] Test set WER reported (once, not iterated)

### Gate

**v2 Small trained. WER beats Vosk (~14%) or is close enough to proceed.** If WER > 18%: consider transfer learning before v2.1.

### Deliverables

- Best v2 Small checkpoint (PyTorch)
- v2 Small ONNX models (FP32 + INT8)
- KenLM model for Russian
- Full evaluation report (all metrics)
- Error analysis report

---

## M14: Phase 2 — Production Training (v2.1 Small)

**Machine**: Cloud H100
**Time**: 2-3 days
**Cost**: ~$100-150
**Prerequisites**: M13

### Actions

1. Start training with same HP as M13 but v2.1 architecture:
   `python training/train.py --config configs/phase2_v21_small.yaml`
   ```
   regularization: {attention_dropout: 0.1, ffn_dropout: 0.1, drop_path: 0.1}
   ```
2. Same monitoring, checkpoint selection, evaluation as M13
3. Compare v2 vs v2.1 Small:
   - WER on same test set
   - Streaming latency
   - Model size
   - Error profiles
4. Decide winner for release (or release both)

### Self-Check

- [ ] Training converges
- [ ] v2.1 Small WER ≤ v2 Small WER (if not, v2.1 didn't scale — release v2)
- [ ] All evaluation metrics documented
- [ ] Winner selected

### Gate

**Both Small models trained and evaluated. Winner decided.**

### Deliverables

- Best v2.1 Small checkpoint (PyTorch)
- v2.1 Small ONNX models (FP32 + INT8)
- v2 vs v2.1 Small comparison report
- Winner selection (with reasoning)

---

## M15: Final Evaluation + Release Preparation

**Machine**: Local workstation
**Time**: 1-2 days
**Cost**: $0
**Prerequisites**: M13, M14

### Actions

1. Run final evaluation on locked test set (CV21 test + Golos test):
   - Report: greedy WER, beam+LM WER, CER, G2P-normalized WER
   - Report: streaming WER, non-streaming WER (quantify streaming cost)
   - Report: TTFT, RTF, model size
   - For both v2 and v2.1 Small (and Tiny variants from M9/M10)
   - **Report test WER once. Do not iterate.**
2. Run benchmarking protocol:
   ```
   Hardware: MacBook Pro M2, 16GB RAM
   Runtime: ONNX Runtime (latest), CPU execution provider
    Audio: 100 utterances from CV21 test, 5-15s each
   Measurement: avg TTFT over 100 utterances, 10 warmup runs
   ```
   Also benchmark on: iPhone (if available), Android (if available), Raspberry Pi 5 (if available)
3. Run error analysis on full test set (all error categories)
4. Prepare release artifacts:
   - Model weights (PyTorch + ONNX FP32 + ONNX INT8) for Tiny and Small (v2 and v2.1)
   - Tokenizer files (256 + 512 variants)
   - KenLM model
   - Model card (training details, data, hyperparameters, WER)
   - Inference code: streaming + non-streaming, with ONNX Runtime
5. Write model card: architecture, training data, WER results, latency benchmarks, known limitations (dialectal coverage, VAD edge cases)
6. Test release artifacts: fresh clone → load model → run inference → get correct output

### Self-Check

- [ ] All metrics reported for all model variants
- [ ] Latency benchmarked with defined methodology
- [ ] Release artifacts load and run from clean state
- [ ] Model card complete
- [ ] Test WER reported (once)

### Gate

**Release artifacts are complete and reproducible.**

### Deliverables

- Final evaluation report (all models, all metrics)
- Release package: weights + tokenizer + KenLM + inference code + model card
- Latency benchmark results
- Error analysis report

---

## M16: README + Repository Cleanup

**Machine**: Local workstation
**Time**: 0.5 day
**Cost**: $0
**Prerequisites**: M15

### Actions

1. Update README with actual results (replace targets with achieved numbers)
2. Clean up: remove debug scripts, temporary files, stale configs
3. Verify: `git clone` → follow README → works end-to-end
4. Tag release: `git tag v1.0-rc1`

### Deliverables

- Clean repository with updated README
- Release tag

---

## Optional Milestones

### M17: Phase 3 — Pseudo-Labeling

**Machine**: Cloud H100
**Time**: 3-5 days
**Cost**: ~$150-250
**Prerequisites**: M15

1. Download OpenSTT 20K hours
2. Run inference with best Small model → pseudo-labels
3. Filter: confidence threshold + length + KenLM perplexity
4. Spot-check 200 random samples
5. Retrain Small on combined 25K+ hours
6. Re-evaluate on locked test set

### M18: Phase 4 — Knowledge Distillation

**Machine**: Local 3090
**Time**: 2-3 days
**Cost**: $0
**Prerequisites**: M15

1. Implement distillation training loop:
   - Loss = α·L_KL(teacher_logits, student_logits) + β·L_CTC + γ·L_hidden
   - α=0.7, β=0.2, γ=0.1, temperature=2.0
2. Teacher: best Small model. Student: Tiny (same track)
3. Train 50K steps on same data
4. Evaluate: target Tiny WER within 1-2% of Small
5. Export distilled Tiny to ONNX INT8 (~50MB)

---

## Summary: Timeline + Budget

| Milestone | Time | Machine | Cost | Cumulative |
|-----------|------|---------|------|------------|
| M1: Environment | 0.5d | Local | $0 | 0.5d, $0 |
| M2: Tokenizer | 0.5d | Local | $0 | 1d, $0 |
| M3: Data Pipeline | 1-2d | Local + 3090 | $0 | 2-3d, $0 |
| M4: Architecture | 1-2d | 3090 | $0 | 3-5d, $0 |
| M5: Training Loop | 1-2d | 3090 | $0 | 4-7d, $0 |
| M6: Overfit Check | 1-2d | 3090 | $0 | 5-9d, $0 |
| M6.5: Data Quality | 1-2d | 3090 | $0 | 6-11d, $0 |
| M7: Streaming + ONNX | 1-2d | 3090 | $0 | 7-13d, $0 |
| M8: Scale-Up Ready | 2-3d | 3090 | $0 | 8-14d, $0 |
| M9: v2 Tiny Train | 2-3d | 3090 | $0 | 10-17d, $0 |
| M10: v2.1 Tiny Train | 2-3d | 3090 | $0 | 12-20d, $0 |
| M11: Ablation | 3-5d | 3090 | $0 | 15-25d, $0 |
| M12: HP Search | 1-2d | Cloud H100 | ~$60 | 16-27d, ~$60 |
| M13: v2 Small Train | 2-3d | Cloud H100 | ~$125 | 18-30d, ~$185 |
| M14: v2.1 Small Train | 2-3d | Cloud H100 | ~$125 | 20-33d, ~$310 |
| M15: Final Eval | 1-2d | Local | $0 | 21-35d, ~$310 |
| M16: Cleanup | 0.5d | Local | $0 | 22-36d, ~$310 |
| **Total** | **~4-5 weeks** | | **~$310** | |

| Optional | Time | Machine | Cost |
|----------|------|---------|------|
| M17: Pseudo-labeling | 3-5d | Cloud H100 | ~$200 |
| M18: Distillation | 2-3d | Local 3090 | $0 |
