import argparse
import faulthandler
import logging
import math
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from models.config import ModelConfig
from models.model import RuMoonshine
from training.checkpoint import CheckpointManager, average_checkpoints
from training.dataset import ASRDataset, collate_fn, load_manifest
from training.logger import TrainLogger
from training.sampler import BucketShuffleSampler, DynamicBatchSampler
from training.validate import validate

logger = logging.getLogger(__name__)

_GPU_TEMP_WARN = 85
_GPU_TEMP_CRIT = 90
_last_gpu_log = 0.0
_peak_vram_mib = 0.0


def _gpu_stats():
    if not torch.cuda.is_available():
        return None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total",
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
            "mem_total_mib": float(parts[4]),
        }
    except Exception:
        return None


def _gpu_mem_pytorch():
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


def _log_gpu_temp(step, force=False):
    global _last_gpu_log, _peak_vram_mib
    now = time.time()
    if not force and (now - _last_gpu_log) < 60:
        return
    gs = _gpu_stats()
    if gs is None:
        return
    _last_gpu_log = now
    mem_pct = gs["mem_mib"] / gs["mem_total_mib"] * 100
    if gs["mem_mib"] > _peak_vram_mib:
        _peak_vram_mib = gs["mem_mib"]
    temp = gs["temp"]
    if temp >= _GPU_TEMP_CRIT:
        logger.warning(
            f"[step {step}] GPU CRITICAL: {temp:.0f}C, {gs['power']:.0f}W, "
            f"VRAM {mem_pct:.0f}% ({gs['mem_mib']:.0f}/{gs['mem_total_mib']:.0f}MB)"
        )
    elif temp >= _GPU_TEMP_WARN:
        logger.warning(
            f"[step {step}] GPU HOT: {temp:.0f}C, {gs['power']:.0f}W, "
            f"VRAM {mem_pct:.0f}%"
        )
    return gs


def setup_optimizer(model, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("name", "schedulefree").lower()
    lr = cfg.get("lr", 1e-3)
    weight_decay = cfg.get("weight_decay", 0.01)

    if name == "schedulefree":
        from schedulefree import AdamWScheduleFree

        return AdamWScheduleFree(
            model.parameters(), lr=lr, weight_decay=weight_decay, warmup_steps=0
        )
    elif name == "adamw":
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, fused=fused
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def setup_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    name = cfg.get("name", "schedulefree").lower()
    if name == "schedulefree":
        return None

    warmup_steps = cfg.get("warmup_steps", 2000)
    max_steps = cfg.get("max_steps", 50000)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_full_config(path: str) -> tuple[ModelConfig, dict]:
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    known = set(ModelConfig.__dataclass_fields__.keys())
    model_data = data.get("model", {})
    model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in known})

    return model_cfg, data


class _StepTimer:
    __slots__ = ("_events", "_cpu_t0", "_labels")

    def __init__(self):
        self._events = []
        self._labels = []
        self._cpu_t0 = None

    def start(self):
        self._events.clear()
        self._labels.clear()
        self._cpu_t0 = time.perf_counter()
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._events.append(e)

    def mark(self, label: str):
        self._labels.append(label)
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._events.append(e)

    def results_ms(self):
        if len(self._events) < 2:
            return {}
        torch.cuda.synchronize()
        out = {}
        for i in range(len(self._events) - 1):
            ms = self._events[i].elapsed_time(self._events[i + 1])
            out[self._labels[i]] = ms
        total_cpu = (time.perf_counter() - self._cpu_t0) * 1000
        out["step_ms"] = total_cpu
        return out


def train(config_path: str, resume: bool = True, seed: int = 42):
    global _peak_vram_mib
    setup_seed(seed)

    model_cfg, full_cfg = load_full_config(config_path)
    train_cfg = full_cfg.get("training", {})

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.set_float32_matmul_precision("high")
    data_cfg = full_cfg.get("data", {})
    log_cfg = full_cfg.get("logging", {})
    opt_cfg = train_cfg.get("optimizer", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = RuMoonshine(model_cfg).to(device)

    aug_cfg = train_cfg.get("augmentation", {})
    model.spec_augment = aug_cfg.get("spec_augment", False)
    if model.spec_augment:
        logger.info("SpecAugment enabled (freq_mask=15, time_mask=50)")

    if train_cfg.get("compile", True) and device.type == "cuda":
        logger.info("Compiling encoder with torch.compile (mode=reduce-overhead)")
        model.encoder = torch.compile(model.encoder, mode="reduce-overhead")

    batch_size = train_cfg.get("batch_size", 16)
    accum_steps = train_cfg.get("accum_steps", 4)
    max_steps = train_cfg.get("max_steps", 50000)
    grad_clip = train_cfg.get("grad_clip", 5.0)

    precision = train_cfg.get("precision", "fp16")
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(precision == "fp16"))

    optimizer = setup_optimizer(model, opt_cfg)
    is_schedulefree = "schedulefree" in opt_cfg.get("name", "").lower()

    train_manifest = data_cfg.get("train_manifest", "data/manifests/train.jsonl")
    val_manifest = data_cfg.get("val_manifest", "data/manifests/val.jsonl")
    tokenizer_model = data_cfg.get("tokenizer_model", "data/tokenizer_256.model")

    train_dataset = ASRDataset(
        manifest_path=train_manifest,
        tokenizer_model=tokenizer_model,
        raw_audio=True,
        spec_augment=False,
        speed_perturbation=aug_cfg.get("speed_perturbation", False),
    )

    val_dataset = ASRDataset(
        manifest_path=val_manifest,
        tokenizer_model=tokenizer_model,
        raw_audio=True,
    )

    records = load_manifest(train_manifest)
    durations = [r.get("duration", 1.0) for r in records]

    batching_cfg = train_cfg.get("batching", {})
    max_tokens = batching_cfg.get("max_tokens", None)

    if max_tokens is not None:
        sampler = DynamicBatchSampler(
            lengths=durations,
            max_tokens=max_tokens,
            frames_per_sec=batching_cfg.get("frames_per_sec", 41.0),
            max_batch_size=batching_cfg.get("max_batch_size", 512),
            min_batch_size=batching_cfg.get("min_batch_size", 4),
            num_buckets=train_cfg.get("num_buckets", 100),
            shuffle=True,
            drop_last=True,
        )
        dl_kwargs = {
            "batch_sampler": sampler,
            "num_workers": train_cfg.get("num_workers", 4),
            "collate_fn": collate_fn,
            "pin_memory": True,
        }
        nw = dl_kwargs["num_workers"]
        if nw > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)
        logger.info(
            f"Dynamic batching: max_tokens={max_tokens}, "
            f"{len(sampler)} batches, "
            f"avg batch_size={len(durations)/len(sampler):.1f}"
        )
    else:
        sampler = BucketShuffleSampler(
            lengths=durations,
            num_buckets=train_cfg.get("num_buckets", 100),
            batch_size=batch_size,
            shuffle=True,
        )
        dl_kwargs = {
            "batch_size": batch_size,
            "sampler": sampler,
            "num_workers": train_cfg.get("num_workers", 4),
            "collate_fn": collate_fn,
            "pin_memory": True,
            "drop_last": True,
        }
        nw = dl_kwargs["num_workers"]
        if nw > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)

    train_loader = torch.utils.data.DataLoader(train_dataset, **dl_kwargs)

    val_batch_size = train_cfg.get("val_batch_size", batch_size)

    val_kwargs = {
        "batch_size": val_batch_size,
        "shuffle": False,
        "num_workers": 2,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if val_kwargs["num_workers"] > 0:
        val_kwargs["persistent_workers"] = True
        val_kwargs["prefetch_factor"] = train_cfg.get("prefetch_factor", 2)

    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)

    steps_per_epoch = len(train_loader)
    scheduler = setup_scheduler(optimizer, opt_cfg, steps_per_epoch)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_model)

    ckpt_cfg = train_cfg.get("checkpointing", {})
    run_name = log_cfg.get("name", "run")
    ckpt_dir = train_cfg.get("checkpoint_dir", f"checkpoints/{run_name}")

    ckpt_mgr = CheckpointManager(
        save_dir=ckpt_dir,
        keep_top_k=ckpt_cfg.get("save_top_k", 5),
    )

    train_logger = TrainLogger(
        backend=log_cfg.get("backend", "tensorboard"),
        project=log_cfg.get("project", "ru-moonshine"),
        name=run_name,
        config=full_cfg,
    )

    start_step = 0
    best_wer = float("inf")
    if resume:
        resumed_step = ckpt_mgr.load_latest(
            model, optimizer, scheduler, scaler, map_location=device
        )
        if resumed_step is not None:
            start_step = resumed_step
            best_metric = ckpt_mgr.best_metric()
            if best_metric is not None:
                best_wer = best_metric
            logger.info(f"Resumed from step {start_step}, best WER={best_wer:.2f}%")

    ckpt_mgr.install_preemption_handler(
        lambda: ckpt_mgr.save_latest(model, optimizer, scheduler, global_step, scaler)
    )

    nonfinite_patience = 5
    nonfinite_count = 0
    escape_wer_patience = train_cfg.get("validation", {}).get("escape_wer_patience", 0)
    escape_wer_min_steps = train_cfg.get("validation", {}).get("escape_wer_min_steps", 5000)
    escape_wer_counter = 0
    last_val_wer = float("inf")
    escape_wer_stopped = False
    log_every = train_cfg.get("log_every", 100)
    val_every = train_cfg.get("validation", {}).get("every_n_steps", 2000)
    ckpt_every = ckpt_cfg.get("every_n_steps", 2000)
    val_max_batches = train_cfg.get("validation", {}).get("max_batches", 50)

    timer = _StepTimer()
    accum_loss_sum = 0.0
    accum_stats_buf = {"loss": 0.0, "loss_aed": 0.0, "loss_ctc": 0.0, "acc": 0.0}

    global_step = start_step
    epoch = 0
    model.train()
    if is_schedulefree:
        optimizer.train()

    logger.info(
        f"Starting training: {max_steps} steps, batch={batch_size}, "
        f"accum={accum_steps}, amp={use_amp}, device={device}"
        + (f", max_tokens={max_tokens}" if max_tokens else "")
    )

    while global_step < max_steps:
        epoch += 1
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            timer.start()

            audio, audio_lengths, tokens, token_lengths = batch
            audio = audio.to(device, non_blocking=True)
            audio_lengths = audio_lengths.to(device)
            tokens = tokens.to(device, non_blocking=True)
            token_lengths = token_lengths.to(device)

            timer.mark("h2d")

            with autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                loss, stats, weight = model(
                    audio,
                    tokens,
                    audio_lengths=audio_lengths,
                    token_lengths=token_lengths,
                )
                loss = loss / accum_steps

            timer.mark("forward")

            for k, v in stats.items():
                if k in accum_stats_buf:
                    accum_stats_buf[k] += v.item() if torch.is_tensor(v) else v

            scaler.scale(loss).backward()

            timer.mark("backward")

            if (batch_idx + 1) % accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip
                )

                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    nonfinite_count += 1
                    logger.warning(
                        f"Non-finite grad norm at step {global_step} "
                        f"({nonfinite_count}/{nonfinite_patience})"
                    )
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if nonfinite_count >= nonfinite_patience:
                        logger.error(
                            f"Aborting: {nonfinite_patience} consecutive non-finite gradients"
                        )
                        ckpt_mgr.save_latest(
                            model, optimizer, scheduler, global_step, scaler
                        )
                        train_logger.close()
                        return
                    continue

                nonfinite_count = 0
                scaler.step(optimizer)
                scaler.update()

                if is_schedulefree:
                    pass
                elif scheduler is not None:
                    scheduler.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                timer.mark("optimizer")

                step_loss = accum_stats_buf["loss"] / accum_steps
                epoch_loss += step_loss
                epoch_batches += 1
                accum_loss_sum += step_loss

                _log_gpu_temp(global_step)

                if global_step % log_every == 0:
                    timings = timer.results_ms()

                    loss_val = accum_stats_buf["loss"] / accum_steps
                    loss_aed_val = accum_stats_buf["loss_aed"] / accum_steps
                    loss_ctc_val = accum_stats_buf["loss_ctc"] / accum_steps
                    acc_val = accum_stats_buf["acc"] / accum_steps

                    log_metrics = {
                        "train/loss": loss_val,
                        "train/loss_aed": loss_aed_val,
                        "train/loss_ctc": loss_ctc_val,
                        "train/acc": acc_val,
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": optimizer.param_groups[0].get("lr", opt_cfg.get("lr", 1e-3)),
                        "train/step": global_step,
                        "train/epoch": epoch,
                        "train/batch_size": audio.shape[0],
                    }

                    for phase in ("h2d", "forward", "backward", "optimizer"):
                        if phase in timings:
                            log_metrics[f"timing/{phase}_ms"] = timings[phase]
                    if "step_ms" in timings:
                        log_metrics["timing/step_ms"] = timings["step_ms"]
                        data_load_ms = timings["step_ms"] - sum(
                            timings.get(p, 0) for p in ("h2d", "forward", "backward", "optimizer")
                        )
                        log_metrics["timing/data_load_ms"] = max(0, data_load_ms)
                        total_compute = timings.get("forward", 0) + timings.get("backward", 0) + timings.get("optimizer", 0)
                        if timings["step_ms"] > 0:
                            log_metrics["timing/gpu_active_pct"] = total_compute / timings["step_ms"] * 100

                    gs = _log_gpu_temp(global_step, force=True)
                    if gs:
                        log_metrics["sys/gpu_temp"] = gs["temp"]
                        log_metrics["sys/gpu_power"] = gs["power"]
                        log_metrics["sys/gpu_util"] = gs["util"]
                        log_metrics["sys/gpu_mem_pct"] = gs["mem_mib"] / gs["mem_total_mib"] * 100
                        log_metrics["sys/gpu_mem_mib"] = gs["mem_mib"]

                    pm = _gpu_mem_pytorch()
                    if pm:
                        log_metrics["sys/gpu_mem_pytorch_mb"] = pm["allocated_mb"]
                        log_metrics["sys/gpu_mem_peak_mb"] = pm["peak_allocated_mb"]
                        log_metrics["sys/gpu_mem_reserved_mb"] = pm["reserved_mb"]
                        torch.cuda.reset_peak_memory_stats()

                    train_logger.log(log_metrics, global_step)

                    timing_str = ""
                    if "timing/data_load_ms" in log_metrics:
                        timing_str = f" data={log_metrics['timing/data_load_ms']:.0f}ms fwd={log_metrics.get('timing/forward_ms', 0):.0f}ms"
                    bs_str = f" bs={audio.shape[0]}" if max_tokens else ""
                    logger.info(
                        f"[step {global_step}] loss={loss_val:.4f} "
                        f"aed={loss_aed_val:.4f} ctc={loss_ctc_val:.4f} "
                        f"acc={acc_val:.3f} grad={grad_norm.item():.2f}"
                        f"{bs_str}{timing_str}"
                    )

                accum_stats_buf = {"loss": 0.0, "loss_aed": 0.0, "loss_ctc": 0.0, "acc": 0.0}

                if global_step % val_every == 0:
                    val_metrics = validate(
                        model, val_loader, sp, device,
                        max_batches=val_max_batches, precision=precision,
                    )
                    val_wer = val_metrics["wer"]

                    gs = _gpu_stats()
                    vram_str = ""
                    if gs:
                        vram_str = f" VRAM={gs['mem_mib'] / gs['mem_total_mib'] * 100:.0f}%"

                    train_logger.log(
                        {
                            "val/loss": val_metrics["val_loss"],
                            "val/wer": val_wer,
                            "val/ser": val_metrics["ser"],
                            "val/cer": val_metrics.get("cer", 0.0),
                        },
                        global_step,
                    )
                    logger.info(
                        f"[step {global_step}] val_loss={val_metrics['val_loss']:.4f} "
                        f"WER={val_wer:.2f}% SER={val_metrics['ser']:.2f}% "
                        f"CER={val_metrics.get('cer', 0.0):.1f}%{vram_str}"
                    )

                    if val_wer < best_wer:
                        best_wer = val_wer
                        ckpt_mgr.save(
                            model, optimizer, scheduler, global_step, val_wer, scaler
                        )
                        train_logger.log_summary({"best_wer": best_wer})

                    ckpt_mgr.save_latest(
                        model, optimizer, scheduler, global_step, scaler
                    )
                    model.train()
                    if is_schedulefree:
                        optimizer.train()

                    if escape_wer_patience > 0 and global_step >= escape_wer_min_steps:
                        if val_wer > last_val_wer:
                            escape_wer_counter += 1
                            logger.info(
                                f"WER increased ({escape_wer_counter}/{escape_wer_patience}): "
                                f"{last_val_wer:.2f}% -> {val_wer:.2f}%"
                            )
                            if escape_wer_counter >= escape_wer_patience:
                                logger.warning(
                                    f"Early stopping: WER increased for {escape_wer_patience} "
                                    f"consecutive validations. Best WER={best_wer:.2f}% at step {global_step}"
                                )
                                escape_wer_stopped = True
                                break
                        else:
                            escape_wer_counter = 0
                    last_val_wer = val_wer

                if global_step % ckpt_every == 0 and global_step % val_every != 0:
                    ckpt_mgr.save_latest(
                        model, optimizer, scheduler, global_step, scaler
                    )

        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            logger.info(
                f"Epoch {epoch} done: avg_loss={avg_loss:.4f}, step={global_step}"
            )

        if escape_wer_stopped:
            ckpt_mgr.save_latest(model, optimizer, scheduler, global_step, scaler)
            break

    stopped_msg = " (early stopped)" if escape_wer_stopped else ""
    gs = _gpu_stats()
    peak_str = f"Peak VRAM: {_peak_vram_mib:.0f}MB"
    if gs:
        peak_str += f" ({_peak_vram_mib / gs['mem_total_mib'] * 100:.0f}% of {gs['mem_total_mib']:.0f}MB)"
    logger.info(f"Training complete{stopped_msg}. Best WER: {best_wer:.2f}%. {peak_str}")

    if ckpt_mgr.checkpoint_paths:
        avg_path = str(Path(ckpt_dir) / "averaged.pt")
        top_n = min(len(ckpt_mgr.checkpoint_paths), 5)
        average_checkpoints(ckpt_mgr.checkpoint_paths[:top_n], avg_path)
        logger.info(f"Averaged top-{top_n} checkpoints → {avg_path}")

    train_logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train RuMoonshine")
    parser.add_argument("config", help="Path to training config YAML")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    faulthandler.enable()

    try:
        train(args.config, resume=not args.no_resume, seed=args.seed)
    except Exception as e:
        logging.getLogger(__name__).error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
