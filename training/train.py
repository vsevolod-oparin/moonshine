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
import psutil
import torch
from torch.amp import GradScaler, autocast

from models.config import ModelConfig
from models.model import RuMoonshine
from training.checkpoint import CheckpointManager, average_checkpoints
from training.config import FullConfig, OptimizerConfig, ThermalConfig
from training.dataset import ASRDataset, collate_fn, load_manifest
from training.logger import TrainLogger
from training.sampler import BucketShuffleSampler, DynamicBatchSampler
from training.validate import validate

logger = logging.getLogger(__name__)


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


def _cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        for name in ("k10temp", "coretemp", "cpu_thermal"):
            if name in temps:
                for entry in temps[name]:
                    if entry.label in ("Tctl", "Package id 0", "") or "core" in entry.label.lower():
                        return entry.current
                return temps[name][0].current
    except Exception:
        pass
    return None


def _temp_str():
    gpu = _gpu_stats()
    cpu = _cpu_temp()
    parts = []
    if cpu is not None:
        parts.append(f"CPU={cpu:.0f}C")
    if gpu:
        parts.append(f"GPU={gpu['temp']:.0f}C")
    return " ".join(parts) if parts else ""


def _gpu_mem_pytorch():
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }


class _ThermalMonitor:
    def __init__(self, cfg: ThermalConfig):
        self.cfg = cfg
        self._last_gpu_log = 0.0
        self._peak_vram_mib = 0.0

    def _max_temp(self):
        gpu = _gpu_stats()
        cpu = _cpu_temp()
        temps = []
        if cpu is not None:
            temps.append(cpu)
        if gpu:
            temps.append(gpu["temp"])
        return max(temps) if temps else None

    def wait_if_hot(self, step):
        t = self._max_temp()
        if t is None or t < self.cfg.cpu_temp_warn:
            return
        ts = _temp_str()
        logger.warning(f"[step {step}] THERMAL PAUSE: {ts} — waiting to cool below {self.cfg.resume_temp:.0f}C")
        while True:
            time.sleep(self.cfg.poll_interval_sec)
            t = self._max_temp()
            ts = _temp_str()
            logger.info(f"[step {step}] Thermal check: {ts}")
            if t is None or t < self.cfg.resume_temp:
                logger.info(f"[step {step}] Resuming training: {ts}")
                break

    def log(self, step, force=False):
        now = time.time()
        if not force and (now - self._last_gpu_log) < 60:
            return None, None
        self._last_gpu_log = now
        gs = _gpu_stats()
        cpu = _cpu_temp()
        if gs and gs["mem_mib"] > self._peak_vram_mib:
            self._peak_vram_mib = gs["mem_mib"]
        ts = _temp_str()
        if gs:
            gpu_t = gs["temp"]
            mem_pct = gs["mem_mib"] / gs["mem_total_mib"] * 100
            if gpu_t >= self.cfg.gpu_temp_crit or (cpu is not None and cpu >= self.cfg.cpu_temp_crit):
                logger.warning(
                    f"[step {step}] THERMAL CRITICAL: {ts}, "
                    f"{gs['power']:.0f}W, VRAM {mem_pct:.0f}%"
                )
            elif gpu_t >= self.cfg.gpu_temp_warn or (cpu is not None and cpu >= self.cfg.cpu_temp_warn):
                logger.warning(
                    f"[step {step}] THERMAL WARNING: {ts}, "
                    f"{gs['power']:.0f}W, VRAM {mem_pct:.0f}%"
                )
        return gs, cpu


def setup_optimizer(model, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    name = cfg.name.lower()
    if name == "schedulefree":
        from schedulefree import AdamWScheduleFree

        return AdamWScheduleFree(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, warmup_steps=0
        )
    elif name == "adamw":
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=fused
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def setup_scheduler(optimizer, cfg: OptimizerConfig, max_steps: int):
    name = cfg.name.lower()
    if name == "schedulefree":
        return None

    schedule = cfg.lr_schedule.lower()
    warmup_steps = cfg.warmup_steps
    min_lr_ratio = cfg.min_lr_ratio

    if schedule == "wsd":
        decay_start_step = cfg.decay_start_step
        decay_steps = cfg.decay_steps
        plateau_start_step = cfg.plateau_start_step
        plateau_steps = cfg.plateau_steps
        post_decay_steps = cfg.post_decay_steps
        final_lr = cfg.final_lr
        final_decay_start_step = cfg.final_decay_start_step
        final_decay_steps = cfg.final_decay_steps

        def _wsd_lr(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < decay_start_step:
                return 1.0
            progress = (step - decay_start_step) / max(1, decay_steps)
            progress = min(1.0, progress)
            decay_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, decay_mult)

        def lr_lambda(step):
            lr_mult = _wsd_lr(step)
            if plateau_steps > 0 and step >= plateau_start_step:
                plateau_end = plateau_start_step + plateau_steps
                if step < plateau_end:
                    return _wsd_lr(plateau_start_step)
                plateau_lr = _wsd_lr(plateau_start_step)
                post_steps = max(1, post_decay_steps) if post_decay_steps > 0 else max(1, max_steps - plateau_end)
                progress = (step - plateau_end) / post_steps
                progress = min(1.0, progress)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr_mult = min_lr_ratio + (plateau_lr - min_lr_ratio) * cosine
            if final_decay_steps > 0 and step >= final_decay_start_step:
                final_lr_mult = final_lr / cfg.lr
                fprogress = (step - final_decay_start_step) / max(1, final_decay_steps)
                fprogress = min(1.0, fprogress)
                return min_lr_ratio + (final_lr_mult - min_lr_ratio) * fprogress
            return lr_mult

    elif schedule == "noam":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return math.sqrt(warmup_steps) / math.sqrt(max(1, step))

    else:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    cfg = FullConfig.from_yaml(config_path)
    tc = cfg.training
    opt = tc.optimizer
    dc = cfg.data
    lc = cfg.logging
    ac = tc.augmentation
    bc = tc.batching
    vc = tc.validation
    cc = tc.checkpointing

    seed = tc.seed
    setup_seed(seed)
    torch.set_num_threads(tc.num_threads)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.set_float32_matmul_precision("high")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    known = set(ModelConfig.__dataclass_fields__.keys())
    model_cfg = ModelConfig(**{k: v for k, v in cfg.model.items() if k in known})
    model = RuMoonshine(model_cfg).to(device)

    model.spec_augment = ac.spec_augment
    if model.spec_augment:
        logger.info("SpecAugment enabled (freq_mask=15, time_mask=50)")

    if tc.compile and device.type == "cuda":
        logger.info("Compiling encoder with torch.compile (mode=default)")
        model.encoder = torch.compile(model.encoder, mode="default")

    max_steps = tc.max_steps
    batch_size = tc.batch_size
    accum_steps = tc.accum_steps
    grad_clip = tc.grad_clip

    precision = tc.precision
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(precision == "fp16"))

    optimizer = setup_optimizer(model, opt)
    is_schedulefree = "schedulefree" in opt.name.lower()

    train_dataset = ASRDataset(
        manifest_path=dc.train_manifest,
        tokenizer_model=dc.tokenizer_model,
        raw_audio=True,
        spec_augment=False,
        speed_perturbation=ac.speed_perturbation,
    )

    val_dataset = ASRDataset(
        manifest_path=dc.val_manifest,
        tokenizer_model=dc.tokenizer_model,
        raw_audio=True,
    )

    records = load_manifest(dc.train_manifest)
    durations = [r.get("duration", 1.0) for r in records]

    thermal = _ThermalMonitor(tc.thermal)

    if bc.max_tokens is not None:
        sampler = DynamicBatchSampler(
            lengths=durations,
            max_tokens=bc.max_tokens,
            frames_per_sec=bc.frames_per_sec,
            max_batch_size=bc.max_batch_size,
            min_batch_size=bc.min_batch_size,
            num_buckets=tc.num_buckets,
            shuffle=True,
            drop_last=True,
        )
        dl_kwargs = {
            "batch_sampler": sampler,
            "num_workers": tc.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": True,
        }
        if tc.num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = tc.prefetch_factor
        logger.info(
            f"Dynamic batching: max_tokens={bc.max_tokens}, "
            f"{len(sampler)} batches, "
            f"avg batch_size={len(durations)/len(sampler):.1f}"
        )
    else:
        sampler = BucketShuffleSampler(
            lengths=durations,
            num_buckets=tc.num_buckets,
            batch_size=batch_size,
            shuffle=True,
        )
        dl_kwargs = {
            "batch_size": batch_size,
            "sampler": sampler,
            "num_workers": tc.num_workers,
            "collate_fn": collate_fn,
            "pin_memory": True,
            "drop_last": True,
        }
        if tc.num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = tc.prefetch_factor

    train_loader = torch.utils.data.DataLoader(train_dataset, **dl_kwargs)

    val_batch_size = tc.get_val_batch_size()
    val_kwargs = {
        "batch_size": val_batch_size,
        "shuffle": False,
        "num_workers": tc.val_num_workers,
        "collate_fn": collate_fn,
        "pin_memory": True,
    }
    if tc.val_num_workers > 0:
        val_kwargs["persistent_workers"] = True
        val_kwargs["prefetch_factor"] = tc.prefetch_factor

    val_loader = torch.utils.data.DataLoader(val_dataset, **val_kwargs)

    scheduler = setup_scheduler(optimizer, opt, max_steps)

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(dc.tokenizer_model)

    run_name = lc.name
    ckpt_dir = f"checkpoints/{run_name}"

    ckpt_mgr = CheckpointManager(
        save_dir=ckpt_dir,
        keep_top_k=cc.save_top_k,
    )

    train_logger = TrainLogger(
        backend=lc.backend,
        project=lc.project,
        name=run_name,
        config=cfg.model_dump(),
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

    nonfinite_patience = tc.nonfinite_patience
    nonfinite_count = 0
    escape_wer_patience = vc.escape_wer_patience
    escape_wer_min_steps = vc.escape_wer_min_steps
    escape_wer_counter = 0
    last_val_wer = float("inf")
    escape_wer_stopped = False
    log_every = tc.log_every
    val_every = vc.every_n_steps
    ckpt_every = cc.every_n_steps
    val_max_batches = vc.max_batches

    timer = _StepTimer()
    accum_stats_buf = {"loss": 0.0, "loss_aed": 0.0, "loss_ctc": 0.0, "acc": 0.0}

    global_step = start_step
    epoch = 0
    model.train()
    if is_schedulefree:
        optimizer.train()

    logger.info(
        f"Starting training: {max_steps} steps, batch={batch_size}, "
        f"accum={accum_steps}, amp={use_amp}, device={device}"
        + (f", max_tokens={bc.max_tokens}" if bc.max_tokens else "")
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

                thermal.log(global_step)
                thermal.wait_if_hot(global_step)

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
                        "train/lr": optimizer.param_groups[0].get("lr", opt.lr),
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

                    gs, cpu = thermal.log(global_step, force=True)
                    if gs:
                        log_metrics["sys/gpu_temp"] = gs["temp"]
                        log_metrics["sys/gpu_power"] = gs["power"]
                        log_metrics["sys/gpu_util"] = gs["util"]
                        log_metrics["sys/gpu_mem_pct"] = gs["mem_mib"] / gs["mem_total_mib"] * 100
                        log_metrics["sys/gpu_mem_mib"] = gs["mem_mib"]
                    if cpu is not None:
                        log_metrics["sys/cpu_temp"] = cpu

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
                    ts = _temp_str()
                    temp_str = f" {ts}" if ts else ""
                    bs_str = f" bs={audio.shape[0]}" if bc.max_tokens else ""
                    logger.info(
                        f"[step {global_step}] loss={loss_val:.4f} "
                        f"aed={loss_aed_val:.4f} ctc={loss_ctc_val:.4f} "
                        f"acc={acc_val:.3f} grad={grad_norm.item():.2f}"
                        f"{bs_str}{timing_str}{temp_str}"
                    )

                accum_stats_buf = {"loss": 0.0, "loss_aed": 0.0, "loss_ctc": 0.0, "acc": 0.0}

                if global_step % val_every == 0:
                    val_metrics = validate(
                        model, val_loader, sp, device,
                        max_batches=val_max_batches, precision=precision,
                    )
                    val_wer = val_metrics["wer"]
                    val_aed_wer = val_metrics.get("wer_aed", -1)

                    gs = _gpu_stats()
                    cpu = _cpu_temp()
                    sys_parts = []
                    if cpu is not None:
                        sys_parts.append(f"CPU={cpu:.0f}C")
                    if gs:
                        sys_parts.append(f"GPU={gs['temp']:.0f}C")
                        sys_parts.append(f"VRAM={gs['mem_mib'] / gs['mem_total_mib'] * 100:.0f}%")
                    sys_str = f" {' '.join(sys_parts)}" if sys_parts else ""

                    log_vals = {
                        "val/loss": val_metrics["val_loss"],
                        "val/wer": val_wer,
                        "val/ser": val_metrics["ser"],
                        "val/cer": val_metrics.get("cer", 0.0),
                    }
                    aed_str = ""
                    if val_aed_wer >= 0:
                        log_vals["val/wer_aed"] = val_aed_wer
                        log_vals["val/ser_aed"] = val_metrics["ser_aed"]
                        log_vals["val/cer_aed"] = val_metrics.get("cer_aed", 0.0)
                        aed_str = f" AED_WER={val_aed_wer:.2f}%"
                    train_logger.log(log_vals, global_step)

                    logger.info(
                        f"[step {global_step}] val_loss={val_metrics['val_loss']:.4f} "
                        f"WER={val_wer:.2f}% SER={val_metrics['ser']:.2f}% "
                        f"CER={val_metrics.get('cer', 0.0):.1f}%{aed_str}{sys_str}"
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
    peak_str = f"Peak VRAM: {thermal._peak_vram_mib:.0f}MB"
    if gs:
        peak_str += f" ({thermal._peak_vram_mib / gs['mem_total_mib'] * 100:.0f}% of {gs['mem_total_mib']:.0f}MB)"
    logger.info(f"Training complete{stopped_msg}. Best WER: {best_wer:.2f}%. {peak_str}")

    if ckpt_mgr.checkpoint_paths:
        avg_path = str(Path(ckpt_dir) / "averaged.pt")
        top_n = min(len(ckpt_mgr.checkpoint_paths), tc.average_top_n)
        average_checkpoints(ckpt_mgr.checkpoint_paths[:top_n], avg_path)
        logger.info(f"Averaged top-{top_n} checkpoints → {avg_path}")

    train_logger.close()


def main():
    parser = argparse.ArgumentParser(description="Train RuMoonshine")
    parser.add_argument("config", help="Path to training config YAML")
    parser.add_argument("--no-resume", action="store_true", help="Start from scratch")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    faulthandler.enable()

    try:
        train(args.config, resume=not args.no_resume, seed=args.seed or 42)
    except Exception as e:
        logging.getLogger(__name__).error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
