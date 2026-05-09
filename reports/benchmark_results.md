# Edge ASR Benchmark: RuMoonshine vs Vosk vs Whisper

**Date:** 2026-05-07
**Hardware:** RTX 3090 (GPU), AMD Ryzen 9 5900X (CPU, 4 threads), 64GB RAM

## Models

| Model | Architecture | Size | Device |
|-------|-------------|------|--------|
| RuMoonshine Tiny v2 | Moonshine v2 (enc 320d/6L, dec 320d/6L) | 26M params | GPU |
| Vosk small-ru-0.22 | TDNN + HCLG decode graph | 45MB | CPU |
| Whisper Tiny | Transformer encoder-decoder | 39M params | GPU |
| Vosk model-ru-0.22 (reference) | Zipformer2 + RNN LM | 1.5GB | CPU |

## Results

### Golos Test Set — Clean Holdout (11,910 samples)

| Model | Size | WER | CER | SER | Speed |
|-------|------|-----|-----|-----|-------|
| Vosk model-ru (server) | 1.5GB | 11.90% | 3.40% | 36.36% | 3.6/s (CPU) |
| **Vosk small-ru** | **45MB** | **13.65%** | **4.24%** | **40.32%** | 2.9/s (CPU) |
| RuMoonshine Tiny | 26M | 60.05% | 27.69% | 88.43% | 52.6/s (GPU) |
| Whisper Tiny | 39M | 66.78% | 28.58% | 91.53% | 23.2/s (GPU) |

### Internal Val Set — Distributionally Contaminated (17,432 samples)

| Model | Size | WER | CER | SER | Speed |
|-------|------|-----|-----|-----|-------|
| Vosk model-ru (server) | 1.5GB | 11.27% | 4.25% | — | 1.3/s (CPU) |
| **RuMoonshine Tiny** | **26M** | **18.03%** | **8.81%** | **47.26%** | 58.3/s (GPU) |
| **Vosk small-ru** | **45MB** | **22.99%** | **8.18%** | **73.62%** | 2.5/s (CPU) |
| Whisper Tiny | 39M | 50.17% | 19.68% | 92.74% | 28.4/s (GPU) |

## Conclusions

### 1. Vosk small is the clear winner on accuracy

The 45MB Vosk small model achieves **13.65% WER on Golos** — a real-world, out-of-domain test set. This is remarkable for a model that fits on a Raspberry Pi. Both RuMoonshine (60%) and Whisper Tiny (67%) are 4-5x worse on the same data. The gap is not a architecture problem — it's a training data and training methodology problem. Vosk small was trained on hundreds of hours of diverse Russian speech with proper Kaldi pipelines, phoneme-based decoding, and an n-gram language model. Our model saw only voice commands and audiobooks.

### 2. Our val set numbers are misleading

RuMoonshine looks competitive on the internal val set (18% vs Vosk small's 23%). But this is an illusion — the val set is drawn from the same distributions the model trained on (87-90% from source train splits). On the clean Golos holdout, RuMoonshine is 4.4x worse. **The val set is not a reliable metric.** Only Golos numbers should be trusted for comparing models.

### 3. Whisper Tiny is not competitive for Russian

Whisper Tiny gets the worst results on both datasets (50-67% WER). This is expected — it's a multilingual model with 99 languages squeezed into 39M params. Russian gets ~1% of the capacity. Both RuMoonshine and Vosk are Russian-specific, which gives them a fundamental advantage regardless of architecture.

### 4. Speed vs accuracy tradeoff is real but secondary

RuMoonshine is 18x faster than Vosk small (GPU vs CPU). But speed only matters if accuracy is acceptable — 60% WER means more than half the words are wrong. A fast wrong answer is still wrong. The speed advantage becomes meaningful only after the accuracy gap is closed.

### 5. The real problem is training data diversity

The core issue is not the Moonshine architecture — it's that our training corpus is narrow (voice commands + audiobooks, ~460 hours). Vosk models train on thousands of hours from diverse sources. To compete:

- **Add diverse Russian data** — Golos, Common Voice, radio, podcasts, calls
- **Add acoustic augmentation** — reverb, noise, farfield simulation
- **Add a language model** — even a simple n-gram LM over the tokenizer would help
- **Use Golos test as the primary metric** — not the contaminated val set

### 6. Architecture comparison: Moonshine vs Whisper

On equal footing (both GPU, both transformer-based, no LM), RuMoonshine edges out Whisper Tiny by ~7pp on both datasets. This validates that the Moonshine architecture (learned frontend, dynamic windowing) has advantages over Whisper's fixed 30-second window for resource-constrained inference. But without good training data, architecture alone cannot compete with a properly-trained traditional pipeline.

## Files

| File | Description |
|------|-------------|
| `reports/rumoonshine_benchmark.json` | RuMoonshine on val set |
| `reports/rumoonshine_golos_benchmark.json` | RuMoonshine on Golos |
| `reports/vosk_small_benchmark.json` | Vosk small on val set |
| `reports/vosk_small_golos_benchmark.json` | Vosk small on Golos |
| `reports/vosk_benchmark.json` | Vosk server on val set |
| `reports/vosk_golos_benchmark.json` | Vosk server on Golos |
| `reports/whisper_tiny_benchmark.json` | Whisper tiny on val set |
| `reports/whisper_tiny_golos_benchmark.json` | Whisper tiny on Golos |
| `benchmarks/eval_vosk.py` | Vosk evaluation script |
| `benchmarks/eval_rumoonshine.py` | RuMoonshine evaluation script |
| `benchmarks/eval_whisper.py` | Whisper evaluation script |
| `/media/smileijp/data/voice/benchmarks/golos_test_manifest.jsonl` | Golos test manifest |
