import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^а-яёa-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_model(checkpoint_path: str, device: torch.device):
    from models.model import RuMoonshine
    from models.config import ModelConfig

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("_orig_mod.", "")] = v

    config = ModelConfig()
    model = RuMoonshine(config)
    model.load_state_dict(cleaned, strict=False)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded")
    return model


@torch.inference_mode()
def eval_rumoonshine(
    checkpoint_path: str,
    manifest_path: str,
    output_path: str,
    batch_size: int = 16,
    max_samples: int = 0,
    device_name: str = "cuda",
):
    import sentencepiece as spm
    from training.validate import aed_greedy_decode
    from training.dataset import AudioProcessor

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device)
    processor = AudioProcessor()

    tokenizer_path = Path("data/tokenizer_256.model")
    if not tokenizer_path.exists():
        tokenizer_path = Path("data/tokenizer_512.model")
        if not tokenizer_path.exists():
            tokenizer_path = Path("data/tokenizer_1024.model")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))
    logger.info(f"Tokenizer: {tokenizer_path}")

    with open(manifest_path) as f:
        records = [json.loads(line) for line in f if line.strip()]

    if max_samples > 0:
        records = records[:max_samples]

    total = len(records)
    refs = []
    hyps = []
    t0 = time.time()

    batch_audio = []
    batch_refs_raw = []

    def process_batch(audio_batch, ref_texts):
        nonlocal refs, hyps

        lengths = torch.tensor([a.shape[0] for a in audio_batch], dtype=torch.long)
        max_len = lengths.max().item()
        padded = torch.zeros(len(audio_batch), max_len, dtype=torch.float32)
        for i, a in enumerate(audio_batch):
            padded[i, :a.shape[0]] = a
        padded = padded.to(device)
        lengths = lengths.to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
            enc_output, enc_lengths = model.encode(padded, lengths)

        aed_ids = aed_greedy_decode(model, enc_output, enc_lengths, max_len=448)

        for i, ids in enumerate(aed_ids):
            hyp_ids = [t for t in ids if t >= 6]
            hyp_text = tokenizer.decode(hyp_ids) if hyp_ids else ""
            ref_text = normalize_text(ref_texts[i])
            hyp_text = normalize_text(hyp_text)
            refs.append(ref_text)
            hyps.append(hyp_text)

    for i, rec in enumerate(records):
        audio = processor.load_audio(rec["audio_path"])
        audio_tensor = torch.from_numpy(audio).float()
        batch_audio.append(audio_tensor)
        batch_refs_raw.append(rec["text"])

        if len(batch_audio) >= batch_size:
            process_batch(batch_audio, batch_refs_raw)
            batch_audio = []
            batch_refs_raw = []

            done = len(refs)
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (total - done) / rate if rate > 0 else 0
            if done % 500 < batch_size:
                sample_wer = jiwer.wer(refs[-500:], hyps[-500:]) * 100
                logger.info(f"[{done}/{total}] recent WER={sample_wer:.1f}% rate={rate:.1f}/s ETA={eta:.0f}s")

    if batch_audio:
        process_batch(batch_audio, batch_refs_raw)

    elapsed = time.time() - t0
    done = len(refs)

    wer = jiwer.wer(refs, hyps) * 100
    cer = jiwer.cer(refs, hyps) * 100
    ser = sum(1 for r, h in zip(refs, hyps) if r != h and len(r) > 0) / max(1, done) * 100

    logger.info(f"RuMoonshine Results ({done} samples, {elapsed:.1f}s):")
    logger.info(f"  WER = {wer:.2f}%")
    logger.info(f"  CER = {cer:.2f}%")
    logger.info(f"  SER = {ser:.2f}%")
    logger.info(f"  Rate = {done / elapsed:.1f} samples/s")

    result = {
        "model": "rumoonshine-tiny-v2",
        "checkpoint": checkpoint_path,
        "samples": done,
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
    parser.add_argument("--checkpoint", default="checkpoints/phase1-v2-tiny-full-v2/averaged.pt")
    parser.add_argument("--manifest", default="data/manifests/val_tokenized.jsonl")
    parser.add_argument("--output", default="reports/rumoonshine_benchmark.json")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    eval_rumoonshine(
        args.checkpoint, args.manifest, args.output,
        args.batch_size, args.max_samples, args.device,
    )
