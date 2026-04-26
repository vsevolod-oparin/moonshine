# M1: Environment Setup

**Status**: Complete
**Date**: 2026-04-26
**Machine**: Local workstation
**Time**: ~0.5 day
**Cost**: $0

## Objective

Set up the project structure, Docker environment with pinned dependencies, and copy Moonshine model definitions as a starting point.

## Actions Completed

### 1. Project Structure

Created `ru-moonshine/` with the planned directory layout:

```
ru-moonshine/
├── configs/          # Training configs (YAML)
├── data/             # Data manifests, versions.json (symlink to external drive)
├── models/           # Model definitions (from transformers)
├── training/         # Training loop, data loader
├── inference/        # Streaming inference, ONNX export
├── tokenizer/        # SentencePiece training + encoding
├── tests/            # PoC tests (T1-T18)
├── scripts/          # Utility scripts
├── planning/         # Planning documents
├── reports/          # Milestone reports
├── Dockerfile        # Pinned environment
└── requirements.txt  # Pinned dependencies
```

### 2. Docker Environment

**Dockerfile** based on `nvidia/cuda:12.4.1-devel-ubuntu22.04`:
- Python 3.11 via deadsnakes PPA (Ubuntu 22.04 doesn't have 3.11 by default)
- PyTorch 2.5.0 + torchaudio 2.5.0 from cu124 index (installed separately from requirements.txt)
- All pinned dependencies via `requirements.txt`

**requirements.txt**:
- `sentencepiece==0.2.1` — BPE tokenizer
- `schedulefree==1.4.1` — Schedule-Free optimizer
- `onnxruntime==1.20.1` + `onnx==1.21.0` — ONNX export/runtime
- `transformers==5.6.2` — for Moonshine model definitions
- `datasets>=3.0` — HuggingFace dataset loading
- `wandb>=0.19` — experiment tracking (primary)
- `tensorboard>=2.18` — experiment tracking (fallback)
- `jiwer>=3.0` — WER computation
- `num2words>=0.5.12` — number-to-words with `lang='ru'` (replaces `ru_num2words` which isn't on PyPI)

### 3. Model Definitions

Copied Moonshine model definitions from HuggingFace `transformers.models.moonshine` into `models/`:
- `models/modeling_moonshine.py` — full model architecture
- `models/configuration_moonshine.py` — model configuration

These files still use `transformers`-relative imports and will be adapted to standalone use in M4.

### 4. Upstream Reference

Cloned Moonshine repo to `../moonshine-upstream` for reference during development.

### 5. Data Storage

`data/` is a git-ignored symlink to `/media/smileijp/5C40E2C140E2A0CE/voice/data` (external drive, 363GB free). Git stores the symlink, not the contents.

### 6. Experiment Tracking

W&B configured as primary tracker. TensorBoard available as zero-config fallback (`tensorboard --logdir runs/`). Logging backend selectable via config YAML.

## Gate Check

| Criterion | Status |
|-----------|--------|
| Docker container builds | Pass |
| GPU detected in container | Pass |
| All imports work (`torch`, `sentencepiece`, `schedulefree`) | Pass |
| Moonshine model code in `models/` | Pass |
| Git repo initialized | Pass |

## Decisions

- **torch/torchaudio installed separately** in Dockerfile from `--index-url https://download.pytorch.org/whl/cu124` — avoids CUDA version mismatch with pip-installed PyTorch
- **`ru_num2words` not used** — not on PyPI; `num2words>=0.5.12` with `lang='ru'` provides equivalent Russian support
- **W&B + TensorBoard dual backend** — thin Logger wrapper with same interface, selectable via config. W&B for cloud runs, TensorBoard for restricted environments

## Deliverables

- `Dockerfile` — CUDA 12.4.1 + Python 3.11 + PyTorch 2.5.0
- `requirements.txt` — pinned deps
- `models/modeling_moonshine.py`, `models/configuration_moonshine.py` — copied from transformers
- Project directory structure
- Initial git commits

## Git Commits

- `67c14fb` Init comment
- `a0f4a7b` readme and license
- `921934c` milestones
- `4ae43aa` Init code commit
- `7bbe1af` Fixing minor issues M1
