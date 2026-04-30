# Knowledge Base
Last updated: 2026-04-30T14:30:06.151533

## [con-20260430143006-33885e]
Category: context
Tags: m8, scaleup, phase1
Changed: 2026-04-30T14:30:06.151505

M8 COMPLETE: Data expanded to 595h (CV21+RuLS+SOVA RuDevices+SOVA Audiobooks). T16 convergence test: best WER 64.4% on 100h subset (50K steps, AdamW lr=5e-4, bf16). Key fixes: bf16 instead of fp16 (gradient overflow on variable batches), cuDNN SDPA disabled (segfault). Download script supports --raw/--merge modes. DataLoader works with num_workers=4+dynamic batching. Phase 1 full config ready: configs/phase1_v2_tiny_full.yaml (595h, 100K steps).

