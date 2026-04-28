import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jiwer
import numpy as np
import sentencepiece as spm
import torch
from torch.utils.data import DataLoader

from models.config import ModelConfig
from models.model import RuMoonshine
from training.dataset import ASRDataset, collate_fn, load_manifest
from training.validate import ctc_greedy_decode

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config_path: str, device: torch.device):
    import yaml

    with open(config_path) as f:
        data = yaml.safe_load(f)
    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    model_cfg = ModelConfig(**{k: v for k, v in data["model"].items() if k in known})

    model = RuMoonshine(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, model_cfg


@torch.no_grad()
def transcribe_batch(model, audio, audio_lengths, blank_id: int):
    enc_output, enc_lengths = model.encode(audio, audio_lengths)
    ctc_logits = model.ctc_head(enc_output)

    if enc_lengths is None:
        enc_lengths = torch.full(
            (enc_output.size(0),),
            enc_output.size(1),
            dtype=torch.long,
            device=audio.device,
        )

    decoded_ids = ctc_greedy_decode(ctc_logits, enc_lengths, blank_id)
    return decoded_ids


def compute_wer(ref: str, hyp: str) -> float:
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 100.0
    try:
        out = jiwer.process_words(ref, hyp)
        ops = out.insertions + out.deletions + out.substitutions
        return 100.0 * ops / max(len(ref.split()), 1)
    except Exception:
        return 100.0


def run_model_scoring(
    checkpoint_path: str,
    config_path: str,
    manifest_path: str,
    tokenizer_path: str,
    batch_size: int,
    output_path: str,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_model(checkpoint_path, config_path, device)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(tokenizer_path)

    dataset = ASRDataset(
        manifest_path=manifest_path,
        tokenizer_model=tokenizer_path,
        raw_audio=True,
        spec_augment=False,
        speed_perturbation=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    records = load_manifest(manifest_path)

    logger.info(f"Scoring {len(records)} clips with batch_size={batch_size}")

    results = []
    idx = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(loader):
        audio, audio_lengths, tokens, token_lengths = batch
        audio = audio.to(device)
        audio_lengths = audio_lengths.to(device)

        decoded_ids = transcribe_batch(
            model, audio, audio_lengths, model_cfg.blank_token_id
        )

        for i in range(len(decoded_ids)):
            if idx >= len(records):
                break
            rec = records[idx]
            hyp_ids = [t for t in decoded_ids[i] if t >= 6]
            hyp_text = tokenizer.DecodeIds(hyp_ids) if hyp_ids else ""

            ref_text = rec["text"]
            wer = compute_wer(ref_text, hyp_text)

            results.append({
                "idx": idx,
                "audio_path": rec["audio_path"],
                "ref": ref_text,
                "hyp": hyp_text,
                "wer": wer,
                "duration": rec.get("duration", 0.0),
                "dataset": rec.get("dataset", "unknown"),
                "speaker_id": rec.get("speaker_id", "unknown"),
                "chars_per_sec": len(ref_text) / max(rec.get("duration", 0.001), 0.001),
            })
            idx += 1

        if (batch_idx + 1) % 100 == 0:
            elapsed = time.time() - start_time
            rate = idx / elapsed
            eta = (len(records) - idx) / rate
            logger.info(
                f"  [{idx}/{len(records)}] {rate:.0f} clips/s, ETA {eta:.0f}s"
            )

    elapsed = time.time() - start_time
    logger.info(f"Done: {len(results)} clips in {elapsed:.0f}s ({len(results)/elapsed:.0f} clips/s)")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    logger.info(f"Results saved to {output_path}")

    return results


def analyze_results(results_path: str, manifest_path: str, report_path: str):
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    wers = [r["wer"] for r in results]
    cps = [r["chars_per_sec"] for r in results]

    by_dataset = defaultdict(list)
    by_speaker = defaultdict(list)
    for r in results:
        by_dataset[r["dataset"]].append(r["wer"])
        by_speaker[r["speaker_id"]].append(r)

    cps_arr = np.array(cps)
    cps_mean = float(np.mean(cps_arr))
    cps_std = float(np.std(cps_arr))
    cps_lo = cps_mean - 2 * cps_std
    cps_hi = cps_mean + 2 * cps_std

    outliers_cps = [r for r in results if r["chars_per_sec"] < cps_lo or r["chars_per_sec"] > cps_hi]

    sorted_results = sorted(results, key=lambda r: r["wer"], reverse=True)
    top_50 = sorted_results[:50]

    speaker_stats = []
    for sid, entries in by_speaker.items():
        wers_s = [e["wer"] for e in entries]
        speaker_stats.append({
            "speaker_id": sid,
            "dataset": entries[0]["dataset"],
            "n_utterances": len(entries),
            "mean_wer": float(np.mean(wers_s)),
            "median_wer": float(np.median(wers_s)),
        })
    speaker_stats.sort(key=lambda x: x["mean_wer"], reverse=True)
    worst_speakers = speaker_stats[:20]

    report = {
        "summary": {
            "total_clips": len(results),
            "mean_wer": float(np.mean(wers)),
            "median_wer": float(np.median(wers)),
            "p90_wer": float(np.percentile(wers, 90)),
            "p95_wer": float(np.percentile(wers, 95)),
            "p99_wer": float(np.percentile(wers, 99)),
            "wer_0_pct": sum(1 for w in wers if w == 0) / len(wers) * 100,
        },
        "by_dataset": {
            ds: {
                "n": len(ws),
                "mean_wer": float(np.mean(ws)),
                "median_wer": float(np.median(ws)),
                "p95_wer": float(np.percentile(ws, 95)),
            }
            for ds, ws in by_dataset.items()
        },
        "chars_per_sec": {
            "mean": cps_mean,
            "std": cps_std,
            "outlier_threshold_low": cps_lo,
            "outlier_threshold_high": cps_hi,
            "n_outliers": len(outliers_cps),
        },
        "top_50_worst_utterances": [
            {
                "idx": r["idx"],
                "ref": r["ref"],
                "hyp": r["hyp"],
                "wer": r["wer"],
                "duration": r["duration"],
                "dataset": r["dataset"],
                "chars_per_sec": r["chars_per_sec"],
            }
            for r in top_50
        ],
        "worst_20_speakers": worst_speakers,
        "cps_outliers_sample": [
            {
                "idx": r["idx"],
                "ref": r["ref"],
                "duration": r["duration"],
                "chars_per_sec": round(r["chars_per_sec"], 1),
                "wer": r["wer"],
                "dataset": r["dataset"],
            }
            for r in sorted(outliers_cps, key=lambda x: abs(x["chars_per_sec"] - cps_mean), reverse=True)[:30]
        ],
        "wer_distribution": {
            "0": sum(1 for w in wers if w == 0),
            "0-10": sum(1 for w in wers if 0 < w <= 10),
            "10-25": sum(1 for w in wers if 10 < w <= 25),
            "25-50": sum(1 for w in wers if 25 < w <= 50),
            "50-100": sum(1 for w in wers if 50 < w <= 100),
            "100": sum(1 for w in wers if w > 100),
        },
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to {report_path}")

    print("\n" + "=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    s = report["summary"]
    print(f"Total clips: {s['total_clips']:,}")
    print(f"Mean WER: {s['mean_wer']:.1f}%  Median: {s['median_wer']:.1f}%")
    print(f"P90: {s['p90_wer']:.1f}%  P95: {s['p95_wer']:.1f}%  P99: {s['p99_wer']:.1f}%")
    print(f"WER=0%: {s['wer_0_pct']:.1f}% of clips")

    print("\nWER Distribution:")
    for bucket, count in report["wer_distribution"].items():
        pct = count / s["total_clips"] * 100
        print(f"  {bucket:>10s}: {count:6d} ({pct:5.1f}%)")

    print("\nBy Dataset:")
    for ds, stats in report["by_dataset"].items():
        print(f"  {ds}: n={stats['n']:,}, mean_wer={stats['mean_wer']:.1f}%, p95={stats['p95_wer']:.1f}%")

    print(f"\nChars/sec: mean={cps_mean:.1f}, std={cps_std:.1f}")
    print(f"  Outliers (|cps - mean| > 2σ): {len(outliers_cps)} ({len(outliers_cps)/len(results)*100:.1f}%)")

    print(f"\nTop 5 worst utterances:")
    for r in top_50[:5]:
        print(f"  [{r['dataset']}] WER={r['wer']:.0f}% | ref: {r['ref'][:60]}")
        print(f"    {'':>30s} | hyp: {r['hyp'][:60]}")

    print(f"\nTop 5 worst speakers:")
    for sp in worst_speakers[:5]:
        print(f"  [{sp['dataset']}] {sp['speaker_id'][:20]}... n={sp['n_utterances']}, mean_wer={sp['mean_wer']:.1f}%")

    return report


def main():
    parser = argparse.ArgumentParser(description="M6.5: Data Quality Assessment")
    parser.add_argument(
        "command",
        choices=["score", "analyze", "full"],
        help="score=run model inference, analyze=generate report, full=both",
    )
    parser.add_argument("--checkpoint", default="checkpoints/m65-quality-model/latest.pt")
    parser.add_argument("--config", default="configs/m65_quality_model.yaml")
    parser.add_argument("--manifest", default="data/manifests/train.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer_256.model")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", default="data/quality_scores.json")
    parser.add_argument("--report", default="data/quality_report.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command in ("score", "full"):
        run_model_scoring(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            manifest_path=args.manifest,
            tokenizer_path=args.tokenizer,
            batch_size=args.batch_size,
            output_path=args.output,
        )

    if args.command in ("analyze", "full"):
        analyze_results(
            results_path=args.output,
            manifest_path=args.manifest,
            report_path=args.report,
        )


if __name__ == "__main__":
    main()
