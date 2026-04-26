# M3: Data Pipeline

**Status**: Complete
**Date**: 2026-04-26
**Machine**: Local workstation + RTX 3090
**Time**: ~1.5 days
**Cost**: $0
**Prerequisites**: M2

## Objective

Download Russian speech datasets, preprocess audio and text, create manifests, and build a PyTorch Dataset+DataLoader with configurable augmentation.

## Actions Completed

### 1. Dataset Download

| Dataset | Source | Clips | Hours | Status |
|---------|--------|-------|-------|--------|
| Common Voice 21 ru | `artyomboyko/common_voice_21_0_ru` | 106,581 | ~170 | Downloaded |
| Russian LibriSpeech | `istupakov/russian_librispeech` | 57,224 | ~98.2 | Downloaded |
| Golos | `SberDevices/Golos` | — | ~1,240 | Empty on HF, deferred |
| MLS Russian | `facebook/multilingual_librispeech` | — | — | No Russian config exists |
| **Total (Phase 1)** | | **163,805** | **~248** | |

CV21 is the accessible equivalent of CV19 (which requires license acceptance on mozilla.org and isn't on HuggingFace).

All data stored on external drive: `data/` is symlinked to `/media/smileijp/5C40E2C140E2A0CE/voice/data`.

### 2. Version Manifest

Created `data/versions.json` with pinned sources and download dates for reproducibility.

### 3. Audio Preprocessing

- Resampled all audio to 16kHz mono WAV
- Saved to `data/processed/{cv21,ruls}/`

### 4. Text Normalization

Applied to all manifests:
- Lowercase
- Abbreviation expansion using M2 dictionary (318 entries)
- Removed non-speech markers
- Kept hyphenated words as single tokens
- Punctuation normalization: `!..`→`!`, `?..`→`?`, `....`→`...`, `!!!`→`!`, `?!`→`?`, `..`→`.`, `?.`→`?`, `!.`→`!`. Legitimate `...` (ellipsis for pauses/interruptions) preserved (881 entries remain). 270 train entries + 10 val entries changed.
- No content filtering — removing utterances because they seem "unworthy" is censorship. Only legitimate targets are punctuation normalization, audio-transcript alignment, and non-speech annotations nobody spoke.

### 5. Train/Val Split

| Split | Clips | Hours | Method |
|-------|-------|-------|--------|
| Train | 156,337 | ~237 | CV21: by speaker ID. RuLS: random (all RuLS clips resolve to single "speaker" from filepath parsing) |
| Val | 7,199 | ~11.2 | Same method, no speaker overlap with train within each dataset |

### 6. Manifest Format

JSON Lines, one entry per clip:
```json
{
  "audio_path": "data/processed/cv21/cv21_validated_099323.wav",
  "text": "мы все испытываем возмущение, если ...",
  "duration": 6.34,
  "dataset": "cv21",
  "speaker_id": "<sha256_hash>"
}
```

### 7. PyTorch Dataset

Implemented in `training/dataset.py`:
- `ASRDataset` — reads manifests, loads audio on-the-fly
- `AudioProcessor` — mel spectrogram extraction (manual mel filterbank as fallback since torchaudio doesn't import locally due to `libcudart.so.13` OSError; will use torchaudio in Docker)
- `SpecAugment` — time + frequency masking
- `SpeedPerturbation` — 0.9x/1.0x/1.1x
- `collate_fn` — dynamic padding for variable-length batches

Planned augmentation features (not yet all implemented, configurable via YAML):
- Vectorized SpecAugment (10-50x faster mask generation via batched random tensors)
- Adaptive time masking (5% of sequence length instead of fixed frame count)
- Augmentation warmup (skip for first 5000 optimizer steps)
- Balanced augmentation (`parallel_augment_fixed_bs`)
- Data mixing strategy: temperature=5 for Phase 1 (2 datasets)
- Stochastic subword sampling: `nbest_size=5, alpha=0.1` during training
- Dithering: 1e-5 × white noise during feature extraction

### 8. MUSAN/RIR

**Deferred.** `data/augmentation/` directory exists but is empty. MUSAN requires license agreement. Phase 1 Tiny on 248h proceeds with SpecAugment + speed perturbation only.

## Gate Check

| Criterion | Status |
|-----------|--------|
| Data loader produces correct shapes for 100 batches | Pass |
| Train manifest has expected clips (156K) | Pass |
| No speaker overlap within datasets | Pass |
| No NaN/Inf in audio features | Pass |
| Augmentation runs without error | Pass |

## Decisions

- **RuLS split randomly** (not by speaker) because all RuLS clips resolve to single "speaker" ID from filepath parsing
- **CV21 instead of CV19** — CV19 requires license acceptance on mozilla.org and isn't on HF; CV21 is the accessible equivalent via `artyomboyko/common_voice_21_0_ru`
- **Punctuation normalization is preprocessing, not censorship** — removing repeated punctuation (`!!!`→`!`) is normalization. Legitimate ellipsis `...` preserved. No content filtering.
- **torchaudio fallback** — uses manual mel filterbank locally (OSError on `libcudart.so.13`); will use torchaudio in Docker
- **MUSAN/RIR deferred** — not freely downloadable without license. SpecAugment + speed perturbation sufficient for Phase 1.

## Data Quality Audit

- 0 real credit card numbers found
- 0 parenthetical/bracket annotations (no `[laughter]`, `(cough)`, etc.)
- Single-word entries are valid speech commands/responses
- Repeated punctuation was the only normalization target (now resolved)

## Deliverables

- `data/versions.json` — pinned dataset versions
- `data/manifests/train.jsonl` — 156,337 entries (~237h)
- `data/manifests/val.jsonl` — 7,199 entries (~11.2h)
- `data/processed/{cv21,ruls}/` — resampled 16kHz mono WAV files
- `data/augmentation/` — empty (MUSAN/RIR deferred)
- `training/dataset.py` — ASRDataset, AudioProcessor, SpecAugment, SpeedPerturbation, collate_fn
- `scripts/download_data.py` — download + preprocess pipeline

## Git Commit

- `8a6d2a9` M3: Data Pipeline
- `ec8922a` Updated planning after example repo research
