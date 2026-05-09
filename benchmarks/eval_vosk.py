import argparse
import io
import json
import logging
import struct
import time
import wave
from pathlib import Path

import jiwer
import numpy as np
from vosk import Model, KaldiRecognizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    import re
    text = text.lower().strip()
    text = re.sub(r"[^а-яёa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def eval_vosk(
    model_path: str,
    manifest_path: str,
    output_path: str = "reports/vosk_benchmark.json",
    max_samples: int = 0,
):
    logger.info(f"Loading Vosk model from {model_path}")
    model = Model(model_path)
    logger.info("Model loaded")

    refs = []
    hyps = []

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if max_samples > 0:
        records = records[:max_samples]

    total = len(records)
    t0 = time.time()
    skipped = 0

    for i, rec in enumerate(records):
        audio_path = rec["audio_path"]
        ref_text = normalize_text(rec["text"])

        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
        except Exception:
            skipped += 1
            continue

        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(16000, sr)
            audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)

        wav_bytes = audio_to_wav_bytes(audio, 16000)
        wf = wave.open(io.BytesIO(wav_bytes), "rb")

        rec_obj = KaldiRecognizer(model, wf.getframerate())
        rec_obj.SetWords(True)

        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            rec_obj.AcceptWaveform(data)

        result = json.loads(rec_obj.FinalResult())
        hyp_text = normalize_text(result.get("text", ""))
        wf.close()

        refs.append(ref_text)
        hyps.append(hyp_text)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            sample_wer = jiwer.wer(refs[-500:], hyps[-500:]) * 100
            logger.info(f"[{i+1}/{total}] recent WER={sample_wer:.1f}% rate={rate:.0f}/s ETA={eta:.0f}s skipped={skipped}")

    elapsed = time.time() - t0
    done = len(refs)
    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100
    ser = sum(1 for r, h in zip(refs, hyps) if r != h and len(r) > 0) / max(1, done) * 100

    logger.info(f"Vosk Results ({done} samples, {elapsed:.1f}s, skipped={skipped}):")
    logger.info(f"  WER = {wer:.2f}%")
    logger.info(f"  CER = {cer:.2f}%")
    logger.info(f"  SER = {ser:.2f}%")

    result = {
        "model": "vosk-model-ru-0.22",
        "samples": done,
        "skipped": skipped,
        "elapsed_sec": round(elapsed, 1),
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
    parser.add_argument("--model", default="/media/smileijp/data/voice/benchmarks/vosk-model-ru-0.22")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="reports/vosk_benchmark.json")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()
    eval_vosk(args.model, args.manifest, args.output, args.max_samples)
