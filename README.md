# ru-Moonshine

Russian edge-optimized streaming automatic speech recognition (ASR), based on the [Moonshine v2](https://arxiv.org/abs/2602.12241) architecture.

## What It Does

**ru-Moonshine** provides offline, streaming Russian speech-to-text that runs locally on phones, MacBooks, and edge devices — no network required. It fills the gap left by existing Russian ASR models, which either require server infrastructure (GigaAM, T-one), lack streaming support (Whisper), or use outdated architectures with high error rates (Vosk).

### Key Properties

| Property | Value |
|----------|-------|
| Streaming TTFT | < 100ms on MacBook CPU |
| Offline | Yes — no network needed |
| Edge-ready | Tiny (34M) fits any phone, Small (123M) fits modern devices |
| Russian language | Native support with custom BPE tokenizer |
| Deployment | ONNX Runtime (CPU/GPU/NNAPI/CoreML) |

## Two-Track Architecture

Both tracks share a tokenizer, data pipeline, training tricks, and inference optimizations:

| | **ru-Moonshine v2** (vanilla) | **ru-Moonshine v2.1** (improved) |
|---|---|---|
| Encoder | Standard Moonshine v2 sliding-window attention | + Causal depthwise conv, multi-scale U-Net, SSC cross-window context |
| Decoder | Causal Transformer with RoPE, cross-attention, SwiGLU FFN | Same |
| Tooling | Existing Moonshine export/deploy | Custom fork, same inference API |
| Risk | Low — proven architecture | Medium — new encoder needs validation |

**Strategy**: Train v2 first. If it meets targets, ship it. v2.1 is the backup with architectural improvements.

### Model Sizes

| Variant | Parameters | Encoder Layers | Decoder Layers | ONNX INT8 Size |
|---------|-----------|---------------|---------------|----------------|
| Tiny | 34M | 6 | 6 | ~50MB |
| Small | 123M | 10 | 10 | ~170MB |
| Medium | 245M | 14 | 14 | ~340MB |

## Streaming Inference

```
Audio (16kHz mono) → VAD → Feature Extraction (50Hz) → Encoder (cached KV) → Decoder (autoregressive) → Text
```

- Encoder uses sliding-window attention (16 frames) with KV-cache reuse for bounded latency
- Decoder fires at VAD boundaries or every 64 encoder frames
- Optional speculative decoding: Tiny (34M) drafts for Small/Medium, 3-4x decoder speedup

## Training Data

| Priority | Dataset | Hours | License |
|----------|---------|-------|---------|
| 1 | Common Voice 19 (ru) | ~3,000 | CC-0 |
| 2 | Golos | ~1,240 | CC-BY 4.0 |
| 3 | Multilingual LibriSpeech (ru) | ~1,000 | CC-BY 4.0 |
| 4 | Russian LibriSpeech | ~98 | Public domain |
| 5 | SOVA | ~100 | Open |
| **Total (commercial-safe)** | | **~5,438** | |
| 6 | OpenSTT (pseudo-labeling, Phase 3) | ~14,000 | CC-BY-NC (research only) |

## Training Pipeline

### Training Tricks (v2 + v2.1)

- **CTC auxiliary loss** (`α=0.2-0.3`) — forces monotonic alignments, +5-15% relative WER improvement
- **SpecAugment + speed perturbation + MUSAN noise + RIR** — real-world robustness
- **Dynamic chunk training** — variable attention windows for flexible latency at deployment
- **Schedule-Free optimizer** — no LR schedule tuning needed
- **Label smoothing** (`ε=0.1`) — decoder regularization
- **Pseudo-labeling** (Phase 3) — label unlabeled audio, +10-30% relative WER improvement

### Inference Optimizations

- **Cache-aware encoder** — 8-16x less per-frame compute via KV-cache reuse
- **INT8/INT4 quantization** — 2-4x smaller, 2-3x faster, <0.5% WER increase
- **ONNX export + operator fusion** — 20-40% CPU speedup
- **Speculative decoding** — Tiny drafts for Small/Medium, 3-4x decoder speedup

### Training Phases

| Phase | Duration | Hardware | Cost | Deliverable |
|-------|----------|----------|------|-------------|
| 0. Setup + PoC tests | 1-2 weeks | Local 3090 | $0 | Tokenizer, data pipeline, all gates pass |
| 1. Validation (Tiny) | 1 week | RTX 3090 | $0 | Both v2 and v2.1 Tiny, initial WER |
| 2. Production (Small) | 1 week | Cloud H100 | ~$250-430 | Both Small models |
| 3. Scale + refine | 1-2 weeks | Cloud H100 | ~$250-450 | Medium, pseudo-labeling (optional) |

## Target Platforms

| Platform | Engine | Model | Latency |
|----------|--------|-------|---------|
| MacBook M1/M2/M3 | ONNX Runtime | Small | < 100ms |
| iPhone 14+ | ONNX Runtime / CoreML | Tiny | < 80ms |
| Android (flagship) | ONNX Runtime (NNAPI) | Small | < 150ms |
| Raspberry Pi 5 | ONNX Runtime (CPU) | Tiny | < 300ms |

## WER Targets

| Model | Expected WER | Notes |
|-------|-------------|-------|
| Vosk-model-ru | ~14% | Current edge+streaming baseline |
| **ru-Moonshine v2 Small** | **12-15%** | Phase 2 baseline |
| **ru-Moonshine v2.1 Small** | **10-13%** | Improved architecture |
| T-one | ~8.6% | Telephony (server) |
| GigaAM-v3 | ~8.4% | SOTA (server, 240M params) |

## Project Structure

```
planning/          Architecture plan, reviews, and design decisions
```

Project is in early planning stage. Implementation follows the phased plan in `planning/MOONSHINE_PLAN.md`.

## References

- [Moonshine v2](https://arxiv.org/abs/2602.12241) — base architecture
- [Moonshine v1](https://arxiv.org/abs/2410.15608) — original paper
- [GigaAM](https://github.com/salute-developers/GigaAM) — Russian ASR baseline
- [Golos dataset](https://huggingface.co/datasets/salute-developers/golos) — Russian speech corpus

## License

[MIT](LICENSE)
