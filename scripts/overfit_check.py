import argparse
import gc
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from models.model import RuMoonshine
from training.dataset import ASRDataset, collate_fn, load_manifest
from training.validate import validate

import sentencepiece as spm
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

GPU_TEMP_WARN = 85
GPU_TEMP_CRIT = 90


def gpu_stats():
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        parts = out.strip().split(", ")
        return {
            "temp": float(parts[0]),
            "power": float(parts[1]),
            "util": float(parts[2]),
            "mem_mib": float(parts[3]),
        }
    except Exception:
        return None


def log(msg=""):
    print(msg, flush=True)


def overfit_test(
    config_path: str,
    manifest_path: str,
    tokenizer_path: str,
    max_steps: int,
    target_wer: float,
    eval_every: int = 50,
    batch_size: int = 10,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model_cfg, full_cfg = load_config_for_overfit(config_path)

    model = RuMoonshine(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    warmup = max(1, max_steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: max(0.0, min(1.0, step / warmup) * (1.0 - step / max_steps)),
    )

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
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    records = load_manifest(manifest_path)
    log(f"\n{'='*60}")
    log(f"Overfit test: {len(records)} clips, {max_steps} steps")
    log(f"Model: {model_cfg.version}, target WER < {target_wer}%")
    log(f"Device: {device}, AMP: {use_amp}, batch={batch_size}")
    log(f"{'='*60}")

    best_wer = float("inf")
    final_wer = None
    last_temp_log = 0

    for step in range(1, max_steps + 1):
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0

        for batch in loader:
            audio, audio_lengths, tokens, token_lengths = batch
            audio = audio.to(device)
            audio_lengths = audio_lengths.to(device)
            tokens = tokens.to(device)
            token_lengths = token_lengths.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                loss, stats, weight = model(
                    audio, tokens,
                    audio_lengths=audio_lengths,
                    token_lengths=token_lengths,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += stats["loss"]
            epoch_batches += 1

        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
        else:
            avg_loss = 0.0

        now = time.time()
        do_log_step = step % 10 == 0 or step == 1
        do_eval = step % eval_every == 0 or step == max_steps
        do_temp = (now - last_temp_log) >= 30

        if do_temp and torch.cuda.is_available():
            gs = gpu_stats()
            if gs:
                temp = gs["temp"]
                power = gs["power"]
                if temp >= GPU_TEMP_CRIT:
                    log(f"  !!! GPU CRITICAL: {temp:.0f}C, {power:.0f}W — consider pausing")
                elif temp >= GPU_TEMP_WARN:
                    log(f"  !! GPU HOT: {temp:.0f}C, {power:.0f}W")
            last_temp_log = now

        if do_eval:
            val_metrics = validate(model, loader, sp, device, max_batches=None)
            final_wer = val_metrics["wer"]
            gs = gpu_stats()
            temp_str = f" GPU={gs['temp']:.0f}C/{gs['power']:.0f}W" if gs else ""
            log(
                f"Step {step:4d}/{max_steps}: loss={avg_loss:.4f} "
                f"WER={final_wer:.2f}% SER={val_metrics['ser']:.2f}% "
                f"lr={optimizer.param_groups[0]['lr']:.1e}{temp_str}"
            )
            if final_wer < best_wer:
                best_wer = final_wer
        elif do_log_step:
            log(f"Step {step:4d}/{max_steps}: loss={avg_loss:.4f}")

        scheduler.step()

    passed = final_wer <= target_wer
    gs = gpu_stats()
    temp_str = f" (GPU {gs['temp']:.0f}C)" if gs else ""
    log(f"\n{'PASS' if passed else 'FAIL'}: Final WER={final_wer:.2f}% (target<{target_wer}%){temp_str}")
    if passed and final_wer == 0.0:
        log("WER = 0%: model perfectly memorized the data.")

    del model, optimizer, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"wer": final_wer, "best_wer": best_wer, "passed": passed}


def profile_memory(
    config_path: str,
    tokenizer_path: str,
    manifest_path: str,
    batch_size: int = 32,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        log("CUDA not available, skipping memory profile")
        return

    model_cfg, _ = load_config_for_overfit(config_path)
    model = RuMoonshine(model_cfg).to(device)

    dataset = ASRDataset(
        manifest_path=manifest_path,
        tokenizer_model=tokenizer_path,
        raw_audio=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.train()
    batch = next(iter(loader))
    audio, audio_lengths, tokens, token_lengths = batch
    audio = audio.to(device)
    audio_lengths = audio_lengths.to(device)
    tokens = tokens.to(device)
    token_lengths = token_lengths.to(device)

    with torch.amp.autocast(device_type="cuda", enabled=True):
        loss, stats, _ = model(
            audio, tokens,
            audio_lengths=audio_lengths,
            token_lengths=token_lengths,
        )
    loss.backward()

    mem_peak = torch.cuda.max_memory_allocated() / 1024**2
    mem_after = torch.cuda.memory_allocated() / 1024**2
    total_params = sum(p.numel() for p in model.parameters())
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2

    gs = gpu_stats()
    temp_str = f" GPU={gs['temp']:.0f}C/{gs['power']:.0f}W" if gs else ""

    log(f"\n{'='*60}")
    log(f"GPU Memory Profile: {model_cfg.version} Tiny, batch={batch_size}{temp_str}")
    log(f"{'='*60}")
    log(f"Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    log(f"Model size: {model_mb:.1f} MB")
    log(f"Batch audio shape: {audio.shape}")
    log(f"Peak VRAM: {mem_peak:.0f} MB")
    log(f"Current VRAM: {mem_after:.0f} MB")
    log(f"GPU total: {total_mb:.0f} MB")
    log(f"Headroom: {total_mb - mem_peak:.0f} MB")

    del model, batch
    gc.collect()
    torch.cuda.empty_cache()

    return {"peak_mb": mem_peak, "params": total_params, "batch_size": batch_size}


def load_config_for_overfit(path: str):
    import yaml
    from models.config import ModelConfig

    with open(path) as f:
        data = yaml.safe_load(f)

    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    model_data = data.get("model", {})
    model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in known})

    return model_cfg, data


def main():
    parser = argparse.ArgumentParser(description="M6: Overfit Sanity Check")
    parser.add_argument(
        "command",
        choices=["t5-v2", "t5-v21", "t6-v2", "t6-v21", "t17", "all"],
        help="Which test to run",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tokenizer = "data/tokenizer_256.model"
    overfit_10 = "data/manifests/overfit_10.jsonl"
    overfit_100 = "data/manifests/overfit_100.jsonl"
    results = {}

    gs = gpu_stats()
    if gs:
        log(f"GPU initial: {gs['temp']:.0f}C, {gs['power']:.0f}W, {gs['mem_mib']:.0f}MB used")

    if args.command in ("t5-v2", "all"):
        log("\n" + "=" * 60)
        log("T5: Overfit 10 samples — v2 Tiny")
        log("=" * 60)
        results["t5-v2"] = overfit_test(
            "configs/overfit_v2_tiny_10.yaml",
            overfit_10, tokenizer,
            max_steps=800, target_wer=0.0,
            eval_every=100, batch_size=10,
        )

    if args.command in ("t5-v21", "all"):
        log("\n" + "=" * 60)
        log("T5: Overfit 10 samples — v2.1 Tiny")
        log("=" * 60)
        results["t5-v21"] = overfit_test(
            "configs/overfit_v21_tiny_10.yaml",
            overfit_10, tokenizer,
            max_steps=800, target_wer=0.0,
            eval_every=100, batch_size=10,
        )

    if args.command in ("t6-v2", "all"):
        log("\n" + "=" * 60)
        log("T6: Overfit 100 samples — v2 Tiny")
        log("=" * 60)
        results["t6-v2"] = overfit_test(
            "configs/overfit_v2_tiny_100.yaml",
            overfit_100, tokenizer,
            max_steps=1500, target_wer=5.0,
            eval_every=150, batch_size=32,
        )

    if args.command in ("t6-v21", "all"):
        log("\n" + "=" * 60)
        log("T6: Overfit 100 samples — v2.1 Tiny")
        log("=" * 60)
        results["t6-v21"] = overfit_test(
            "configs/overfit_v21_tiny_100.yaml",
            overfit_100, tokenizer,
            max_steps=1500, target_wer=5.0,
            eval_every=150, batch_size=32,
        )

    if args.command in ("t17", "all"):
        log("\n" + "=" * 60)
        log("T17: GPU Memory Profiling")
        log("=" * 60)
        for cfg, label in [
            ("configs/overfit_v2_tiny_10.yaml", "v2 Tiny"),
            ("configs/overfit_v21_tiny_10.yaml", "v2.1 Tiny"),
        ]:
            results[f"t17-{label}"] = profile_memory(
                cfg, tokenizer, overfit_100, batch_size=32,
            )

    log("\n" + "=" * 60)
    log("M6 SUMMARY")
    log("=" * 60)
    all_pass = True
    for name, r in results.items():
        if isinstance(r, dict) and "passed" in r:
            status = "PASS" if r["passed"] else "FAIL"
            log(f"  {name}: WER={r['wer']:.2f}% — {status}")
            if not r["passed"]:
                all_pass = False
        elif isinstance(r, dict) and "peak_mb" in r:
            log(f"  {name}: peak={r['peak_mb']:.0f}MB, params={r['params']/1e6:.1f}M")

    gs = gpu_stats()
    if gs:
        log(f"  GPU final: {gs['temp']:.0f}C, {gs['power']:.0f}W")

    log(f"\nOverall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
