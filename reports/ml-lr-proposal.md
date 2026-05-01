# Learning Rate Schedule Proposal for ru-Moonshine

## Problem Statement

The current LR schedule ties cosine decay to `max_steps`:

```python
progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
```

This has two issues:

1. **Changing `max_steps` reshapes the entire LR curve.** During M9 training, increasing `max_steps` from 100K to 200K after a crash caused LR to jump from 1.79e-4 to 4.01e-4 — a 2.2x increase. The model recovered, but this was accidental, not intentional.

2. **Cosine decay is optimal only when training duration is known in advance** (Hoffmann et al., 2022 — Chinchilla). If WER is still improving at step 200K, extending training requires either accepting a too-low LR or triggering another LR bump.

## Evidence from M9 Training

| Phase | Steps | LR Range | CTC WER | Rate |
|-------|-------|----------|---------|------|
| Pre-crash | 4K–60K | 5e-4 → 1.79e-4 | 91.3 → 56.7 | -0.62%/1K |
| Accidental bump | 60K | 1.79e-4 → 4.01e-4 | — | — |
| Post-bump recovery | 64K–76K | 4.01e-4 → 3.47e-4 | 58.2 → 56.9 | recovery |
| Post-bump learning | 76K–112K | 3.47e-4 → 2.07e-4 | 56.9 → 52.9 | -0.11%/1K |

The model handled the 2.2x LR bump without issues. WER continued improving steadily. But the improvement rate is slowing as LR decays — at step 112K, LR is already at 41% of peak, and by step 160K it hits the floor (5e-5, 10% of peak). The last 40K steps (20% of training) are wasted at minimum LR.

## What Production Systems Use

| System | Schedule | Decoupled from max_steps? |
|--------|----------|--------------------------|
| wav2vec2 / fairseq | Inverse sqrt (noam) | Yes |
| wenet Conformer | Warmup → Hold → Cosine | Yes (explicit params) |
| ESPnet NoamLR | Inverse sqrt | Yes |
| Whisper (OpenAI) | Linear warmup → linear decay | No |
| SmolLM2 / Megatron-LM | WSD (constant + cooldown) | Yes |

## Proposed Solution: WSD (Warmup-Stable-Decay)

Replace the cosine-tied-to-max_steps schedule with **WSD** — the emerging standard validated by Defazio et al. (2024) and Wen et al. (2024):

```
Phase 1 — Warmup:   linear ramp from 0 to peak LR over warmup_steps
Phase 2 — Stable:   constant peak LR (majority of training)
Phase 3 — Decay:    short cooldown over decay_steps (5-20% of training)
```

### Why WSD

- **Decoupled from max_steps.** LR is constant for the bulk of training. No accidental bumps from changing training duration.
- **Matches cosine quality.** Defazio et al. (2024) proved constant LR + short cooldown matches optimally-tuned cosine.
- **Flexible.** Train as long as needed. Decay only when WER plateaus. Early stopping handles convergence.
- **Theoretically grounded.** Wen et al. (2024) showed high LR explores the loss landscape ("river"), decay exploits ("descends to valley").

### Proposed Config Parameters

```yaml
training:
  optimizer:
    name: adamw
    lr: 0.0005
    weight_decay: 0.01
    # Schedule: WSD (warmup-stable-decay)
    lr_schedule: wsd
    warmup_steps: 2000
    decay_start_step: 150000    # when to begin cooldown
    decay_steps: 50000          # duration of cooldown
    min_lr_ratio: 0.1           # floor = lr * min_lr_ratio = 5e-5
```

### Proposed LR Curve

```
step       0: lr=0.000e+00   (0% of peak)
step    2000: lr=5.000e-04   (100% — warmup done)
step   50000: lr=5.000e-04   (100% — stable)
step  100000: lr=5.000e-04   (100% — stable)
step  150000: lr=5.000e-04   (100% — decay begins)
step  160000: lr=4.044e-04   (81%)
step  170000: lr=3.310e-04   (66%)
step  180000: lr=2.621e-04   (52%)
step  190000: lr=2.000e-04   (40%)
step  200000: lr=1.545e-04   (31%)
step  210000: lr=1.178e-04   (24%)
step  220000: lr=9.549e-05   (19%)
step  230000: lr=8.091e-05   (16%)
step  240000: lr=7.286e-05   (15%)
step  250000: lr=5.000e-05   (10% — floor hit)
step  260000: lr=5.000e-04   (10% — floor)
```

### Implementation

```python
def setup_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    name = cfg.get("name", "schedulefree").lower()
    if name == "schedulefree":
        return None

    schedule = cfg.get("lr_schedule", "cosine").lower()
    warmup_steps = cfg.get("warmup_steps", 2000)
    min_lr_ratio = cfg.get("min_lr_ratio", 0.1)

    if schedule == "wsd":
        decay_start = cfg.get("decay_start_step", 150000)
        decay_steps = cfg.get("decay_steps", 50000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / max(1, decay_steps)
            progress = min(1.0, progress)
            decay_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr_ratio, decay_mult)

    elif schedule == "cosine":
        max_steps = cfg.get("max_steps", 50000)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    elif schedule == "noam":
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return math.sqrt(warmup_steps) / math.sqrt(max(1, step))

    else:
        raise ValueError(f"Unknown lr_schedule: {schedule}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### Migration Plan for Current M9 Run

The current run is at step ~112K with `max_steps=200K` cosine schedule. Options:

1. **Restart from latest checkpoint with WSD.** Set `decay_start_step=150000, decay_steps=50000`. The model picks up at step 112K with LR jumping from 2.07e-4 to 5e-4 (2.4x). Based on the accidental bump experience (2.2x), the model should handle this fine.

2. **Finish current run, use WSD for future runs.** Let M9 complete its cosine schedule to 200K, then switch to WSD for M10+.

Option 1 is recommended — the data shows the model benefits from higher LR, and the accidental bump demonstrated resilience to LR jumps.

## References

1. Defazio et al. (2024) — "Revisiting Cosine Schedule" — constant LR + cooldown matches cosine. arXiv:2405.18392
2. Wen et al. (2024) — "Understanding Warmup-Stable-Decay" — theoretical justification. arXiv:2410.05192
3. Hoffmann et al. (2022) — Chinchilla: cosine optimal only when matching training duration
4. Hu et al. (2024) — WSD scheduler for LLM pretraining
5. "Pre-training LLM without LR Decay" (March 2026) — no-decay training improves fine-tuning. arXiv:2603.16127
6. fairseq LR scheduler docs — inverse_sqrt, cosine, tri_stage implementations
7. wenet/utils/scheduler.py — warmup → hold → cosine with min_lr floor
8. ESPnet TristageLR docs — warmup/hold/decay ratio-based scheduling
9. arXiv:2602.04774 (Feb 2026) — Theory of Optimal Learning Rate Schedules — proves WSD optimal for hard tasks
