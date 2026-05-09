# Benchmark Plan: RuMoonshine vs Vosk

## Goal
Compare RuMoonshine Tiny (phase1-v2-tiny-full-v2) against Vosk model-ru-0.22 on identical test sets.

## Models

| Model | Architecture | Size | Notes |
|-------|-------------|------|-------|
| RuMoonshine Tiny (v2) | Moonshine v2 (enc 320d/6L, dec 320d/6L) | ~26M params | Best CTC WER=44.10%, AED WER=18.51% |
| Vosk model-ru-0.22 | Zipformer2 (icefall/k2) | ~1.5GB | Reported 6.1% WER on Common Voice RU |

## Data Layout

All benchmark data lives on the external data volume:

```
/media/smileijp/data/
├── downloads/                          # downloads in progress
│   ├── vosk-model-ru.zip              # Vosk model (~1.5GB)
│   ├── golos_opus.tar.gz              # Golos dataset (~20GB)
│   ├── vosk_download.log
│   └── golos_download.log
├── voice/
│   ├── data/                           # symlinked as project data/
│   │   ├── manifests/
│   │   │   └── val_tokenized.jsonl     # internal val set (17,432 samples)
│   │   └── ...
│   └── benchmarks/                     # benchmark models & extracted data
│       ├── vosk-model-ru/             # extracted Vosk model
│       └── golos/                      # extracted Golos test set
└── (project root symlink: moonshine/data → /media/smileijp/data/voice/data)
```

**Download commands** (already running in background):
```bash
nohup wget -c -O /media/smileijp/data/downloads/vosk-model-ru.zip \
  "https://alphacephei.com/vosk/models/vosk-model-ru-0.22.zip" \
  > /media/smileijp/data/downloads/vosk_download.log 2>&1 &

nohup wget -c -O /media/smileijp/data/downloads/golos_opus.tar.gz \
  "https://openslr.elda.org/resources/114/golos_opus.tar.gz" \
  > /media/smileijp/data/downloads/golos_download.log 2>&1 &
```

**Check progress:** `tail -5 /media/smileijp/data/downloads/*_download.log`

## Test Sets

### 1. Internal Val Set (priority)
- **Path:** `data/manifests/val_tokenized.jsonl` (→ `/media/smileijp/data/voice/data/manifests/val_tokenized.jsonl`)
- **Size:** 17,432 samples
- **Source:** mix of sova_rudevices, sova_audiobooks, etc.
- **Status:** ready
- **Note:** models may have data overlap with training sets, need to verify

### 2. Golos Test Set
- **Source:** OpenSLR SLR114
- **Size:** 11,910 samples, ~12.6 hours (crowd + farfield)
- **Download:** `https://openslr.elda.org/resources/114/golos_opus.tar.gz`
- **Download to:** `/media/smileijp/data/downloads/golos_opus.tar.gz`
- **Extract to:** `/media/smileijp/data/voice/benchmarks/golos/`
- **Status:** downloading (~68min remaining)
- **Manifest:** need to create from Golos test files after extraction

## Evaluation Scripts

### `benchmarks/eval_vosk.py` (done)
- Loads Vosk model, iterates over manifest, computes WER/CER
- Text normalization: lowercase, strip punctuation, collapse whitespace
- Saves results to `reports/vosk_benchmark.json`
- Usage: `python benchmarks/eval_vosk.py --model /media/smileijp/data/voice/benchmarks/vosk-model-ru --manifest data/manifests/val_tokenized.jsonl`

### `benchmarks/eval_rumoonshine.py` (TODO)
- Load RuMoonshine checkpoint, run AED greedy decode on same test set
- Same text normalization as Vosk eval
- Saves results to `reports/rumoonshine_benchmark.json`

## Steps

1. **Wait for downloads**
   - [x] Vosk model: `/media/smileijp/data/downloads/vosk-model-ru.zip`
   - [x] Golos dataset: `/media/smileijp/data/downloads/golos_opus.tar.gz`

2. **Setup Vosk model**
   - [x] Extracted to `/media/smileijp/data/voice/benchmarks/vosk-model-ru-0.22/`
   - [x] Run `eval_vosk.py` on internal val set → **WER=11.27%, CER=4.25%**

3. **Run RuMoonshine on same data**
   - [x] Written `eval_rumoonshine.py`
   - [x] Run on internal val set → **WER=18.03%, CER=8.81%**
   - [x] Run on Golos test set → **WER=60.05%, CER=27.69%**

4. **Run both on Golos test set**
   - [x] Extracted to `/media/smileijp/data/voice/benchmarks/golos/`
   - [x] Created Golos manifest at `/media/smileijp/data/voice/benchmarks/golos_test_manifest.jsonl`
   - [x] Run Vosk eval on Golos → **WER=11.90%, CER=3.40%**
   - [x] Run RuMoonshine eval on Golos → **WER=60.05%, CER=27.69%**

5. **Results comparison table**
   - [x] WER, CER, SER side by side
   - [x] Per-domain breakdown (crowd vs farfield)
   - [x] Speed comparison (samples/sec)
   - [x] Final report: `reports/benchmark_results.md`

## Fairness Considerations

- **Text normalization must be identical** for both models
- **Vosk may have been trained on overlapping data** (Common Voice, Golos) — our model was also trained on some of these
- **RuMoonshine AED decoder** should be compared (not CTC) since Vosk uses full decoding pipeline
- **Golos test set** is the most neutral benchmark since it's a published standard

## Files

| File | Location | Status |
|------|----------|--------|
| `benchmarks/eval_vosk.py` | project | done |
| `benchmarks/eval_rumoonshine.py` | project | done |
| `reports/vosk_benchmark.json` | project | done |
| `reports/rumoonshine_benchmark.json` | project | done |
| `reports/vosk_golos_benchmark.json` | project | done |
| `reports/rumoonshine_golos_benchmark.json` | project | done |
| `reports/benchmark_results.md` | project | done |
| Vosk model download | `/media/smileijp/data/downloads/vosk-model-ru.zip` | done |
| Golos dataset download | `/media/smileijp/data/downloads/golos_opus.tar.gz` | done |
| Vosk model (extracted) | `/media/smileijp/data/voice/benchmarks/vosk-model-ru-0.22/` | done |
| Golos test set (extracted) | `/media/smileijp/data/voice/benchmarks/golos/` | done |
| Golos test manifest | `/media/smileijp/data/voice/benchmarks/golos_test_manifest.jsonl` | done |
| RuMoonshine checkpoint | `checkpoints/phase1-v2-tiny-full-v2/averaged.pt` | ready |
