import argparse
import logging
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from models.config import ModelConfig
from models.model import RuMoonshine
from training.checkpoint import CheckpointManager, average_checkpoints
from training.dataset import ASRDataset, collate_fn, load_manifest
from training.logger import TrainLogger
from training.sampler import BucketShuffleSampler
from training.validate import validate

logger = logging.getLogger(__name__)


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
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
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

    known = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    model_data = data.get("model", {})
    model_cfg = ModelConfig(**{k: v for k, v in model_data.items() if k in known})

    return model_cfg, data


def sample_dynamic_window(model_cfg: ModelConfig):
    w_left = random.randint(8, 16)
    w_right_fl = random.randint(0, 4)
    w_right_mid = random.choice([0, w_right_fl]) if random.random() < 0.5 else 0

    orig = (model_cfg.window_left, model_cfg.window_right_first_last, model_cfg.window_right_middle)
    model_cfg.window_left = w_left
    model_cfg.window_right_first_last = w_right_fl
    model_cfg.window_right_middle = w_right_mid
    return orig


def restore_window(model_cfg: ModelConfig, orig: tuple):
    model_cfg.window_left, model_cfg.window_right_first_last, model_cfg.window_right_middle = orig


def train(config_path: str, resume: bool = True, seed: int = 42):
    setup_seed(seed)

    model_cfg, full_cfg = load_full_config(config_path)
    train_cfg = full_cfg.get("training", {})
    data_cfg = full_cfg.get("data", {})
    log_cfg = full_cfg.get("logging", {})
    opt_cfg = train_cfg.get("optimizer", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = RuMoonshine(model_cfg).to(device)

    batch_size = train_cfg.get("batch_size", 16)
    accum_steps = train_cfg.get("accum_steps", 4)
    max_steps = train_cfg.get("max_steps", 50000)
    grad_clip = train_cfg.get("grad_clip", 5.0)
    dynamic_window = train_cfg.get("dynamic_window", False)

    use_amp = train_cfg.get("precision", "fp16") == "fp16" and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    optimizer = setup_optimizer(model, opt_cfg)
    is_schedulefree = "schedulefree" in opt_cfg.get("name", "").lower()

    train_manifest = data_cfg.get("train_manifest", "data/manifests/train.jsonl")
    val_manifest = data_cfg.get("val_manifest", "data/manifests/val.jsonl")
    tokenizer_model = data_cfg.get("tokenizer_model", "data/tokenizer_256.model")

    aug_cfg = train_cfg.get("augmentation", {})
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
    sampler = BucketShuffleSampler(
        lengths=durations,
        num_buckets=train_cfg.get("num_buckets", 100),
        batch_size=batch_size,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=train_cfg.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )

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
    log_every = train_cfg.get("log_every", 100)
    val_every = train_cfg.get("validation", {}).get("every_n_steps", 2000)
    ckpt_every = ckpt_cfg.get("every_n_steps", 2000)
    val_max_batches = train_cfg.get("validation", {}).get("max_batches", 50)

    global_step = start_step
    epoch = 0
    model.train()
    if is_schedulefree:
        optimizer.train()

    logger.info(
        f"Starting training: {max_steps} steps, batch={batch_size}, "
        f"accum={accum_steps}, amp={use_amp}, device={device}"
    )

    while global_step < max_steps:
        epoch += 1
        epoch_loss = 0.0
        epoch_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if global_step >= max_steps:
                break

            audio, audio_lengths, tokens, token_lengths = batch
            audio = audio.to(device, non_blocking=True)
            audio_lengths = audio_lengths.to(device)
            tokens = tokens.to(device, non_blocking=True)
            token_lengths = token_lengths.to(device)

            orig_window = None
            if dynamic_window:
                orig_window = sample_dynamic_window(model_cfg)

            with autocast(device_type="cuda", enabled=use_amp):
                loss, stats, weight = model(
                    audio,
                    tokens,
                    audio_lengths=audio_lengths,
                    token_lengths=token_lengths,
                )
                loss = loss / accum_steps

            if orig_window is not None:
                restore_window(model_cfg, orig_window)

            scaler.scale(loss).backward()

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
                    scaler.unscale_(optimizer)
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

                epoch_loss += stats["loss"]
                epoch_batches += 1

                if global_step % log_every == 0:
                    log_metrics = {
                        "train/loss": stats["loss"],
                        "train/loss_aed": stats["loss_aed"],
                        "train/loss_ctc": stats["loss_ctc"],
                        "train/acc": stats["acc"],
                        "train/grad_norm": grad_norm.item(),
                        "train/lr": optimizer.param_groups[0].get("lr", opt_cfg.get("lr", 1e-3)),
                        "train/step": global_step,
                        "train/epoch": epoch,
                    }
                    train_logger.log(log_metrics, global_step)
                    logger.info(
                        f"[step {global_step}] loss={stats['loss']:.4f} "
                        f"aed={stats['loss_aed']:.4f} ctc={stats['loss_ctc']:.4f} "
                        f"acc={stats['acc']:.3f} grad={grad_norm.item():.2f}"
                    )

                if global_step % val_every == 0:
                    val_metrics = validate(
                        model, val_loader, sp, device, max_batches=val_max_batches
                    )
                    val_wer = val_metrics["wer"]
                    train_logger.log(
                        {
                            "val/loss": val_metrics["val_loss"],
                            "val/wer": val_wer,
                            "val/ser": val_metrics["ser"],
                        },
                        global_step,
                    )
                    logger.info(
                        f"[step {global_step}] val_loss={val_metrics['val_loss']:.4f} "
                        f"WER={val_wer:.2f}% SER={val_metrics['ser']:.2f}%"
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

                if global_step % ckpt_every == 0 and global_step % val_every != 0:
                    ckpt_mgr.save_latest(
                        model, optimizer, scheduler, global_step, scaler
                    )

        if epoch_batches > 0:
            avg_loss = epoch_loss / epoch_batches
            logger.info(
                f"Epoch {epoch} done: avg_loss={avg_loss:.4f}, step={global_step}"
            )

    logger.info(f"Training complete. Best WER: {best_wer:.2f}%")

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
    )

    train(args.config, resume=not args.no_resume, seed=args.seed)


if __name__ == "__main__":
    main()
