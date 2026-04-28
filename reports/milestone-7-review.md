# Milestone 7 Code Review

**Reviewer**: Automated review  
**Date**: 2026-04-28  
**Scope**: `inference/streaming_encoder.py`, `tests/test_m7_streaming_onnx.py`, `tests/test_m4_model.py`  
**Status**: 14 findings (3 medium, 11 low)

---

## Findings

### F1. Dropout active during streaming inference if model is in training mode
- **Severity**: MEDIUM
- **File**: `inference/streaming_encoder.py:82-128` (StreamingEncoderV2.process_chunk)
- **Description**: `process_chunk` has `@torch.no_grad()` which prevents gradient computation, but does NOT set the encoder to eval mode. If the parent model is in training mode (e.g., user forgot `.eval()`), dropout layers in the encoder are active, producing non-deterministic output. The same chunk processed twice yields different results.
- **Evidence**: Tested with `model.train()` + `process_chunk()`: outputs differ between calls with same seed. Dropout rate is 0.1 on attention and FFN.
- **Fix**: Add `self.encoder.eval()` in `StreamingEncoderV2.__init__()` or wrap `process_chunk` with a context manager that temporarily sets eval mode.
- **Impact**: Silent correctness bug in production if caller forgets `.eval()`. Streaming output would be noisy and non-reproducible.

### F2. Decoder has no KV cache — O(N^2) autoregressive generation
- **Severity**: MEDIUM
- **File**: `inference/streaming_encoder.py:227-263` (StreamingASR.decode_buffer)
- **Description**: `decode_buffer` generates tokens autoregressively by re-running the full decoder on the growing `generated` sequence at each step. No decoder self-attention KV cache is maintained. This is O(N^2) in token count. Measured at 423ms for 50 tokens on v2 Tiny.
- **Evidence**: Line 248: `generated = torch.cat([generated, torch.tensor([[token]], ...)], dim=1)` grows each step. Line 251: `model.decoder(generated, enc_output)` reprocesses all tokens.
- **Fix**: Add decoder KV cache (similar to encoder cache). The `MultiHeadAttention` class already supports `past_key`/`past_value`/`use_cache` — just need to thread it through the decoder layers.
- **Impact**: Significant latency in real-time streaming. At ~8ms per decoder step (measured), 100-token output takes 800ms. With KV cache: ~8ms total for 100 tokens.

### F3. KV cache memory allocation pattern is wasteful
- **Severity**: MEDIUM
- **File**: `inference/streaming_encoder.py:42-53` (CircularKVCache.update)
- **Description**: Every `update()` call does `torch.cat([old, new], dim=2)` then slices to `window_size`. After warmup, each update allocates a tensor of size `(old + new)` then immediately discards it, keeping only `window_size` frames. For a 21-frame window and 32-frame chunk: allocates 53 frames, keeps 21. New `data_ptr` every call confirms no reuse.
- **Evidence**: Benchmarked: data_ptr changes every update. 2 allocations per update (cat + slice).
- **Fix**: Pre-allocate a circular buffer tensor of size `(1, num_heads, window_size + chunk_size, head_dim)` and use index assignment with wrapping. Alternatively, use `torch.roll` + slice.
- **Impact**: ~2.5x memory churn per streaming step. Not critical for batch_size=1 inference but adds GC pressure in production.

### F4. Dead code in StreamingASR
- **Severity**: LOW
- **File**: `inference/streaming_encoder.py:197-201`
- **Description**: `prev_text` (line 199), `hold_buffer` (line 200), and `hold_n` (line 201) are set in `__init__` and cleared in `reset()` but never otherwise used. These are placeholder fields for segment deduplication and hold-N mechanism described in the plan but not implemented.
- **Fix**: Either implement the hold-N mechanism (hold back N tokens between chunks for revision) or remove the dead fields until needed. If kept as placeholders, add comments explaining the intended behavior.

### F5. Unused import in test file
- **Severity**: LOW
- **File**: `tests/test_m7_streaming_onnx.py:10`
- **Description**: `CircularKVCache` is imported but never used in any test. The import was likely from an earlier iteration when tests directly tested cache behavior.
- **Fix**: Remove `CircularKVCache` from the import line.

### F6. Unused method in TestT14ONNXExport
- **Severity**: LOW
- **File**: `tests/test_m7_streaming_onnx.py:280-285`
- **Description**: `_get_onnx_eps()` is defined but never called by any test method. It was likely intended to be used as a setup/skip check but the tests import `onnxruntime` inline and would naturally fail if unavailable.
- **Fix**: Remove the method, or convert it to a module-level `pytest.importorskip("onnxruntime")` check.

### F7. Streaming encoder not exported for ONNX streaming inference
- **Severity**: LOW
- **File**: `inference/streaming_encoder.py` (architectural)
- **Description**: T15 "ONNX streaming test" in the milestone plan says "Export encoder with KV cache inputs/outputs. Chunk-by-chunk ONNX inference. Compare to PyTorch streaming." The current implementation does NOT export the streaming encoder to ONNX. T15 tests non-streaming ONNX encoder/decoder parity, not streaming ONNX. The `StreamingEncoderV2` is not a `nn.Module` and cannot be exported by `torch.onnx.export`.
- **Evidence**: Attempted `torch.onnx.export(streaming.process_chunk, ...)` — fails with `AttributeError: 'function' object has no attribute 'modules'`.
- **Impact**: T15 gate is partially met (ONNX components work individually) but the stated goal of "chunk-by-chunk ONNX inference with state carry-over" is not validated. This needs a separate streaming ONNX wrapper module.
- **Deferred**: Full streaming ONNX export with KV cache requires a dedicated export wrapper. Recommend M9 scope.

### F8. `_make_streaming_chunk_mask` allocates tensors every call
- **Severity**: LOW
- **File**: `inference/streaming_encoder.py:10-27`
- **Description**: `torch.tensor(0.0, ...)` and `torch.tensor(_MASK_NEG, ...)` on lines 24-25 create new tensors inside `torch.where` on every call. Benchmarked at 0.028ms per call. Called 6× per chunk (once per layer), 7× per audio = 42 calls. Not a bottleneck but creates unnecessary small tensor allocations.
- **Fix**: Pre-allocate scalar constants as class attributes or module-level constants.

### F9. `StreamingASR.add_audio_chunk` splits frames one-by-one into list
- **Severity**: LOW
- **File**: `inference/streaming_encoder.py:207-208`
- **Description**: `for t in range(frames.size(1)): self.frame_buffer.append(frames[:, t : t + 1, :])` splits the preprocessor output into individual frame tensors and stores them in a list. Then immediately reassembles with `torch.cat(self.frame_buffer[:self.chunk_size], dim=1)`. This is wasteful: split → list → cat for every audio chunk.
- **Fix**: Track a frame count index into the full preprocessor output tensor instead of splitting into individual tensors. Or process the preprocessor output directly in chunks without intermediate list storage.

### F10. `encoder_buffer` stores individual frame tensors
- **Severity**: LOW
- **File**: `inference/streaming_encoder.py:214-215, 224-225`
- **Description**: Encoder output is stored as individual frame tensors: `for t in range(enc_out.size(1)): self.encoder_buffer.append(enc_out[:, t, :].squeeze(0))`. Then `decode_buffer` reassembles with `torch.stack(self.encoder_buffer, dim=1)`. Creates N individual tensors where N can be hundreds.
- **Fix**: Append the full chunk output as a single tensor and concatenate on decode. Change `encoder_buffer: list[torch.Tensor]` where each entry is `(chunk_len, d_model)`.

### F11. T10 TTFT test measures first-chunk only, not actual TTFT
- **Severity**: LOW
- **File**: `tests/test_m7_streaming_onnx.py:149-177`
- **Description**: `test_ttft_constant` processes only the first chunk (32 frames) for each audio length. The test title says "TTFT is bounded" but it doesn't measure the actual time-to-first-token, which would include preprocessor + first chunk + decoder step. It measures only `process_chunk` latency for the first chunk, which is always the same size regardless of audio length — trivially constant.
- **Evidence**: `first_chunk = frames[:, :chunk_size, :]` — always 32 frames. The test verifies that processing 32 frames takes the same time regardless of how long the total audio is, which is expected but doesn't validate that TTFT doesn't grow.
- **Fix**: For a more meaningful TTFT test, measure the full pipeline: preprocessor + first chunk encoder + first decoder step. Or at minimum, verify that the preprocessor output time is proportional to audio length (expected) while the encoder first-chunk time is constant.

### F12. `StreamingASR.decode_buffer` re-runs full decoder + lm_head every token
- **Severity**: LOW (overlaps with F2)
- **File**: `inference/streaming_encoder.py:248-253`
- **Description**: Each token generation step does `torch.cat` to grow the sequence, then runs the full decoder + lm_head. This is the same issue as F2 but from the tensor allocation perspective: `torch.cat` allocates a new tensor each step, and `model.decoder(generated, enc_output)` processes all tokens including already-processed ones.
- **Fix**: Same as F2 — add decoder KV cache.

### F13. ONNX TracerWarnings for mask generation
- **Severity**: LOW
- **File**: `models/masks.py:17-18`
- **Description**: `torch.tensor(0.0, ...)` and `torch.tensor(_MASK_NEG, ...)` inside `torch.where` trigger TracerWarnings during ONNX export because they're registered as constants. This means the exported ONNX graph has the mask values baked in for the specific `seq_len` used during export.
- **Impact**: Non-streaming export works because the mask is rebuilt by ONNX Runtime for the actual input shape (dynamic axes handle this). But this would break streaming ONNX export where mask depends on runtime state.
- **Fix**: For streaming ONNX, implement mask generation using only ONNX-compatible ops (arange, comparison, where with broadcastable scalars).

### F14. v2.1 streaming not implemented
- **Severity**: LOW
- **File**: `inference/streaming_encoder.py:188-189`
- **Description**: `StreamingASR` raises `NotImplementedError` for v2.1 encoder. The v2.1 encoder has multi-scale U-Net with downsampling/upsampling stages, causal depthwise conv, and skip connections. Streaming for v2.1 requires per-stage KV caches and handling of the frame rate changes.
- **Impact**: v2.1 training and inference will proceed without streaming until this is implemented. Plan mentions M10 scope for v2.1 Tiny training.

---

## Summary

| Severity | Count | IDs |
|----------|-------|-----|
| HIGH | 0 | — |
| MEDIUM | 3 | F1 (dropout), F2 (decoder cache), F3 (memory alloc) |
| LOW | 11 | F4-F14 |
| **Total** | **14** | |

**Recommended fixes before M8**:
- F1 (dropout): Add eval-mode enforcement — 2 lines, prevents silent correctness bug
- F5 (unused import): Cleanup
- F6 (unused method): Cleanup

**Recommended fixes before M9 (Phase 1 training)**:
- F2 (decoder KV cache): Critical for streaming latency in production
- F4 (dead code): Remove or document placeholders

**Can defer**:
- F3 (memory alloc), F7 (streaming ONNX), F8 (mask allocation), F9/F10 (buffer optimization), F13 (TracerWarnings) — performance optimizations, not correctness issues
- F11 (TTFT test accuracy) — test quality, not code bug
- F14 (v2.1 streaming) — explicitly deferred
