import argparse
import io
import json
import logging
import re
import time
import wave
from pathlib import Path

import jiwer
import numpy as np
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^а-яёa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def eval_whisper(
    model_size: str = "tiny",
    manifest_path: str = "",
    output_path: str = "",
    max_samples: int = 0,
    device: str = "cuda",
):
    logger.info(f"Loading Whisper {model_size}...")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    logger.info("Model loaded")

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if max_samples > 0:
        records = records[:max_samples]

    total = len(records)
    refs = []
    hyps = []
    t0 = time.time()
    skipped = 0

    for i, rec in enumerate(records):
        try:
            import soundfile as sf
            audio, sr = sf.read(rec["audio_path"])
        except Exception:
            skipped += 1
            continue

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        segments, info = model.transcribe(
            audio, language="ru", beam_size=1, best_of=1,
            condition_on_previous_text=False,
        )
        hyp_text = normalize_text(" ".join(s.text for s in segments))
        ref_text = normalize_text(rec["text"])

        refs.append(ref_text)
        hyps.append(hyp_text)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            sample_wer = jiwer.wer(refs[-500:], hyps[-500:]) * 100
            logger.info(f"[{i+1}/{total}] recent WER={sample_wer:.1f}% rate={rate:.1f}/s ETA={eta:.0f}s skipped={skipped}")

    elapsed = time.time() - t0
    done = len(refs)
    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100
    ser = sum(1 for r, h in zip(refs, hyps) if r != h and len(r) > 0) / max(1, done) * 100

    logger.info(f"Whisper {model_size} Results ({done} samples, {elapsed:.1f}s, skipped={skipped}):")
    logger.info(f"  WER = {wer:.2f}%")
    logger.info(f"  CER = {cer:.2f}%")
    logger.info(f"  SER = {ser:.2f}%")
    logger.info(f"  Rate = {done / elapsed:.1f} samples/s")

    result = {
        "model": f"whisper-{model_size}",
        "samples": done,
        "skipped": skipped,
        "elapsed_sec": round(elapsed, 1),
        "rate_per_sec": round(done / elapsed, 1),
        "wer": round(wer, 2),
        "cer": round(cer, 2),
        "ser": round(ser, 2),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="tiny")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    eval_whisper(args.model_size, args.manifest, args.output, args.max_samples, args.device)
