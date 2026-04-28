# Training Performance: Research & Optimization Plan

## Problem Statement

Training runs showed GPU underutilization (~130W/480W, 6% util at BS=16) and CPU overload. The user observed ~23GB VRAM usage on a 24GB GPU during BS=96 training.

## Diagnosis

### Benchmark Results (April 28, 2026)

**Data loading speed** (4 workers, no GPU work):
| BS | clips/s | ms/batch |
|----|---------|----------|
| 16 | 1,239 | 13 |
| 32 | 1,096 | 29 |
| 64 | 858 | 75 |

**GPU throughput** (forward+backward+optimizer, no bucketing):
| BS | clips/s | GPU W | Util% | VRAM |
|----|---------|-------|-------|------|
| 16 | 14 | 151 | 6% | 4.5GB |
| 32 | 24 | 148 | 6% | 10GB |
| 64 | 44 | 250 | 18% | 20.8GB |
| 96 | 57 | 221 | 75% | 14GB |
| 128 | 66 | 269 | 100% | 21.3GB |

**GPU throughput** (BS=96, WITH BucketShuffleSampler):
| Metric | Value |
|--------|-------|
| clips/s | 225 |
| Peak VRAM (PyTorch tensors) | 2,080MB |
| nvidia-smi VRAM (actual) | ~23GB |

### Root Causes

1. **Batch size too small** — 22M-param model can't saturate a 3090 at BS=16. Each kernel finishes instantly, GPU idles between kernels. Fixed by increasing BS to 96.

2. **VRAM discrepancy** — `torch.cuda.max_memory_allocated()` showed 2GB but nvidia-smi showed 23GB. The gap comes from: optimizer states (AdamW stores 2x model params as momentum+variance = ~175MB), cuDNN workspace, CUDA context (~500MB), gradient scaler buffers, memory fragmentation, and variable-length audio padding in non-uniform buckets.

3. **Hidden CPU-GPU sync points** — `loss.item()` called 4x per step forces CPU to wait for GPU completion, breaking async execution pipeline.

4. **Per-sample SentencePiece tokenization** — `spm.encode()` called in every `__getitem__()`. Not cached.

## Monitoring Plan

### Step Timing Breakdown

Measure each phase using `torch.cuda.Event` for GPU-accurate timing:

```
data_load_ms  — time waiting for next(batch) from DataLoader
h2d_ms        — time for .to(device) host-to-device transfer
forward_ms    — forward pass (including autocast)
backward_ms   — loss.backward()
optimizer_ms  — optimizer.step() + zero_grad()
step_ms       — total wall time per step
gpu_idle_pct  — (data_load_ms + h2d_ms) / step_ms × 100
```

Logged to tensorboard/wandb every `log_every` steps.

### VRAM Monitoring

Three layers of memory tracking:
- **nvidia-smi** `memory.used` / `memory.total` → real process VRAM including all overhead
- **`torch.cuda.memory_allocated()`** → PyTorch tensor memory only
- **`torch.cuda.max_memory_allocated()`** → peak tensor memory (reset after each log)

Logged as `sys/gpu_mem_pct`, `sys/gpu_mem_pytorch_mb`, `sys/gpu_mem_peak_mb`.

Run-wide peak tracker printed at training end.

### Sync Point Elimination

Defer `loss.item()` to log intervals only. Store detached tensors between logs:

```python
# Before (4 syncs per step):
stats = {"loss": loss.item(), "loss_aed": loss_aed.item(), ...}

# After (0 syncs per step, N syncs per log_every):
stats = {"loss": loss.detach(), "loss_aed": loss_aed.detach(), ...}
# .item() called only when logging
```

## Optimization Plan

### Phase 1: Easy Wins (minimal code changes)

#### 1. Fused AdamW
`torch.optim.AdamW(..., fused=True)` combines the optimizer's elementwise ops into a single CUDA kernel. Reduces kernel launch overhead.

PyTorch docs: "fused (bool) – whether the fused implementation is used. Currently, fused=True is only supported on CUDA."

Expected: **5-15% speedup** on optimizer step.

#### 2. cuDNN Benchmark Mode
`torch.backends.cudnn.benchmark = True` auto-selects fastest cuDNN convolution algorithm per input shape.

Caveat: re-benchmarks when input sizes change. Our variable-length audio means constant re-benchmarking. HOWEVER, with BucketShuffleSampler, clips within a batch have similar length → similar tensor shapes → cuDNN caches per bucket. Safe to enable.

Expected: **marginal for our Conv1d encoder** (cuDNN auto-tuning mainly helps Conv2d).

#### 3. Pre-tokenize Manifests
Add `token_ids` field to each manifest line (pre-computed SentencePiece encoding). Eliminates `spm.encode()` call in every `__getitem__()`.

Script: read manifest → for each line, tokenize text → write back with `token_ids` field.

### Phase 2: torch.compile (needs testing)

`torch.compile(model)` uses TorchDynamo + TorchInductor to JIT-compile the model into optimized Triton kernels. PyTorch 2.0 paper reports **1.41x training speedup** across 180+ models.

Risks for our model:
- Variable-length inputs (dynamic shapes) → potential graph breaks
- RoPE implementation uses complex numbers → historically caused issues
- `torch.amp.autocast("cuda", enabled=False)` in encoder → may interfere with compilation
- SDPA attention → should compile cleanly (well-supported)

Test plan:
1. Try `torch.compile(model, mode="reduce-overdue")` on a few forward+backward passes
2. Check for graph breaks with `TORCH_LOGS="dynamo"` env var
3. If graph breaks found, isolate which module causes them
4. Consider compiling only decoder (static shapes per batch) while leaving encoder in eager mode

### Phase 3: RAM Audio Cache (conditional)

Only if monitoring shows `data_load_ms` is a significant fraction of `step_ms`.

Total audio size: 156K clips × ~80KB avg ≈ 12GB. Fits in system RAM.

Options:
- Load all WAV files into a dict at dataset init (simplest, ~12GB RAM)
- Use `torch.multiprocessing.shared_memory` for worker-safe access
- Use WebDataset / TFRecord format for sequential reads (more complex, overkill for our scale)

### Phase 4: Not Applicable

- **CUDA graphs**: Requires fixed-shape inputs. Our variable-length audio makes this impossible without padding all sequences to the same length.
- **Channel-last format**: Image-specific optimization.
- **Gradient checkpointing**: Memory-saving technique, not speed. Not needed at Tiny scale (22M params, 2GB VRAM).

## Sources

- PyTorch 2.0 paper: 1.41x training speedup with torch.compile (ASPLOS 2024)
- "Speed Up PyTorch Training by 3.2x with NVIDIA Nsight": step-by-step profiling methodology, loss.item() sync point identification, cuDNN benchmark, torch.compile, AMP
- PyTorch DataLoader docs: persistent_workers, prefetch_factor, pin_memory, non_blocking behavior
- StackOverflow #72961448: multiprocessing serialization overhead for large numpy arrays limits DataLoader scaling
- FFCV library: up to 52% training cost reduction via optimized data pipeline (relevant but we'll use simpler approaches first)
- IBM/PyTorch FSDP + torch.compile: 10-23% MFU improvement from compile, data loader bottleneck at 6T tokens
