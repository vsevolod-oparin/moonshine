import os
import time

import pytest
import torch
import torch.nn.functional as F

from models.config import PRESETS
from models.model import RuMoonshine
from inference.streaming_encoder import StreamingEncoderV2, CircularKVCache


def _encode_non_streaming(model: RuMoonshine, audio: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        enc_output, _ = model.encode(audio.unsqueeze(0) if audio.dim() == 1 else audio)
    return enc_output.squeeze(0)


def _encode_streaming(model: RuMoonshine, audio: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    model.eval()
    config = model.config
    streaming = StreamingEncoderV2(model.encoder, config)
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=False):
            frames, _ = model.preprocessor(audio.unsqueeze(0) if audio.dim() == 1 else audio, None)
    seq_len = frames.size(1)
    outputs = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk = frames[:, start:end, :]
        with torch.no_grad():
            enc_out = streaming.process_chunk(chunk)
        outputs.append(enc_out.squeeze(0))
    return torch.cat(outputs, dim=0)


class TestT9StreamingParity:
    @pytest.fixture
    def v2_tiny(self):
        model = RuMoonshine(PRESETS["v2_tiny"])
        model.eval()
        return model

    def test_cosine_similarity(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(80000)

        non_stream = _encode_non_streaming(v2_tiny, audio)
        stream = _encode_streaming(v2_tiny, audio, chunk_size=32)

        assert non_stream.shape == stream.shape, (
            f"Shape mismatch: non_stream={non_stream.shape}, stream={stream.shape}"
        )

        cos_sims = F.cosine_similarity(non_stream, stream, dim=-1)
        mean_sim = cos_sims.mean().item()
        min_sim = cos_sims.min().item()

        assert mean_sim > 0.99, f"Mean cosine sim {mean_sim:.6f} < 0.99"
        assert min_sim > 0.97, f"Min cosine sim {min_sim:.6f} < 0.97 (boundary frames)"

    def test_cosine_similarity_large_chunk(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(80000)

        non_stream = _encode_non_streaming(v2_tiny, audio)
        stream = _encode_streaming(v2_tiny, audio, chunk_size=128)

        cos_sims = F.cosine_similarity(non_stream, stream, dim=-1)
        mean_sim = cos_sims.mean().item()
        min_sim = cos_sims.min().item()

        assert mean_sim > 0.999, f"Large chunk mean cosine sim {mean_sim:.6f} < 0.999"
        assert min_sim > 0.99, f"Large chunk min cosine sim {min_sim:.6f} < 0.99"

    def test_dependency_matrix(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(80000)

        config = v2_tiny.config
        chunk_size = 32
        max_wr = config.window_right_first_last

        streaming = StreamingEncoderV2(v2_tiny.encoder, config)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                frames, _ = v2_tiny.preprocessor(audio.unsqueeze(0), None)

        seq_len = frames.size(1)

        original_outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            with torch.no_grad():
                enc_out = streaming.process_chunk(frames[:, start:end, :])
            original_outputs.append(enc_out.clone())
        original = torch.cat(original_outputs, dim=1)

        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(n_chunks - 1):
            out_chunk_start = chunk_idx * chunk_size
            out_chunk_end = min(out_chunk_start + chunk_size, seq_len)
            next_chunk_start = (chunk_idx + 1) * chunk_size
            next_chunk_end = min(next_chunk_start + chunk_size, seq_len)

            sampled_inp = list(range(next_chunk_start, next_chunk_end, 4))
            if sampled_inp[-1] != next_chunk_end - 1:
                sampled_inp.append(next_chunk_end - 1)

            for inp_frame in sampled_inp:
                test_streaming = StreamingEncoderV2(v2_tiny.encoder, config)
                test_frames = frames.clone()
                test_frames[0, inp_frame, :] = torch.randn_like(test_frames[0, inp_frame, :])

                outputs = []
                for start in range(0, seq_len, chunk_size):
                    end = min(start + chunk_size, seq_len)
                    with torch.no_grad():
                        enc_out = test_streaming.process_chunk(test_frames[:, start:end, :])
                    outputs.append(enc_out)
                modified = torch.cat(outputs, dim=1)

                for out_frame in range(out_chunk_start, min(out_chunk_start + chunk_size - max_wr, out_chunk_end)):
                    diff = (original[0, out_frame] - modified[0, out_frame]).abs().sum()
                    assert diff < 1e-3, (
                        f"Future leakage: input frame {inp_frame} (next chunk) "
                        f"affects output frame {out_frame} (previous chunk), diff={diff:.6f}"
                    )

    def test_various_chunk_sizes(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(80000)
        non_stream = _encode_non_streaming(v2_tiny, audio)

        for cs in [32, 64, 128]:
            stream = _encode_streaming(v2_tiny, audio, chunk_size=cs)
            cos_sim = F.cosine_similarity(non_stream, stream, dim=-1).mean().item()
            assert cos_sim > 0.99, f"Chunk size {cs}: cosine sim {cos_sim:.6f} < 0.99"


class TestT10TTFTBounded:
    @pytest.fixture
    def v2_tiny(self):
        model = RuMoonshine(PRESETS["v2_tiny"])
        model.eval()
        return model

    @pytest.mark.parametrize("duration_s", [1, 5, 10, 30])
    def test_ttft_constant(self, v2_tiny, duration_s):
        torch.manual_seed(42)
        sr = 16000
        audio = torch.randn(sr * duration_s)

        config = v2_tiny.config
        streaming = StreamingEncoderV2(v2_tiny.encoder, config)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                frames, _ = v2_tiny.preprocessor(audio.unsqueeze(0), None)

        chunk_size = 32
        first_chunk = frames[:, :chunk_size, :]

        times = []
        for _ in range(5):
            streaming.reset()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                streaming.process_chunk(first_chunk)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_time = sum(times) / len(times)
        assert avg_time < 1.0, f"TTFT {avg_time*1000:.1f}ms too high for {duration_s}s audio"

    def test_ttft_does_not_grow_with_length(self, v2_tiny):
        torch.manual_seed(42)
        sr = 16000
        times = {}

        for duration_s in [1, 5, 10, 30]:
            audio = torch.randn(sr * duration_s)
            config = v2_tiny.config
            streaming = StreamingEncoderV2(v2_tiny.encoder, config)

            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=False):
                    frames, _ = v2_tiny.preprocessor(audio.unsqueeze(0), None)

            chunk_size = 32
            first_chunk = frames[:, :chunk_size, :]

            run_times = []
            for _ in range(5):
                streaming.reset()
                t0 = time.perf_counter()
                with torch.no_grad():
                    streaming.process_chunk(first_chunk)
                t1 = time.perf_counter()
                run_times.append(t1 - t0)
            times[duration_s] = sum(run_times) / len(run_times)

        ratio_30_1 = times[30] / times[1] if times[1] > 0 else float("inf")
        assert ratio_30_1 < 1.5, (
            f"TTFT grows with audio length: 30s={times[30]*1000:.1f}ms, 1s={times[1]*1000:.1f}ms, ratio={ratio_30_1:.2f}"
        )


class TestT11KVCacheCorrectness:
    @pytest.fixture
    def v2_tiny(self):
        model = RuMoonshine(PRESETS["v2_tiny"])
        model.eval()
        return model

    def test_incremental_vs_full(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(5 * 16000, device="cpu")

        non_stream = _encode_non_streaming(v2_tiny, audio)
        stream = _encode_streaming(v2_tiny, audio, chunk_size=128)

        cos_sim = F.cosine_similarity(non_stream, stream, dim=-1)
        mean_sim = cos_sim.mean().item()
        min_sim = cos_sim.min().item()

        assert mean_sim > 0.999, f"Mean cosine sim {mean_sim:.6f} < 0.999"
        assert min_sim > 0.98, f"Min cosine sim {min_sim:.6f} < 0.98"

    def test_chunk32_stable_frames(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(5 * 16000, device="cpu")

        non_stream = _encode_non_streaming(v2_tiny, audio)
        stream = _encode_streaming(v2_tiny, audio, chunk_size=32)

        chunk_size = 32
        wr = v2_tiny.config.window_right_first_last
        seq_len = non_stream.size(0)
        stable_mask = torch.ones(seq_len, dtype=torch.bool)
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            if end < seq_len:
                stable_mask[max(0, end - wr):end] = False

        stable_cos = F.cosine_similarity(non_stream, stream, dim=-1)[stable_mask]
        assert stable_cos.mean().item() > 0.99, f"Stable mean cos sim {stable_cos.mean():.6f} < 0.99"

    def test_long_audio_cache(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(30 * 16000, device="cpu")

        non_stream = _encode_non_streaming(v2_tiny, audio)
        stream = _encode_streaming(v2_tiny, audio, chunk_size=128)

        cos_sim = F.cosine_similarity(non_stream, stream, dim=-1)
        assert cos_sim.mean().item() > 0.998, f"30s audio mean cos sim {cos_sim.mean():.6f} < 0.998"

    def test_cache_reset(self, v2_tiny):
        torch.manual_seed(42)
        audio = torch.randn(80000)

        out1 = _encode_streaming(v2_tiny, audio, chunk_size=32)
        out2 = _encode_streaming(v2_tiny, audio, chunk_size=32)

        max_diff = (out1 - out2).abs().max().item()
        assert max_diff < 1e-6, f"Reset produces different results: diff {max_diff:.6e}"


class TestT14ONNXExport:
    @pytest.fixture
    def v2_tiny(self):
        model = RuMoonshine(PRESETS["v2_tiny"])
        model.eval()
        return model

    def _get_onnx_eps(self):
        try:
            import onnxruntime
            return onnxruntime.get_available_providers()
        except ImportError:
            pytest.skip("onnxruntime not available")

    def test_encoder_export(self, v2_tiny, tmp_path):
        import onnxruntime as ort
        torch.manual_seed(42)

        audio = torch.randn(1, 80000)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                enc_frames, _ = v2_tiny.preprocessor(audio, None)
            pt_out = v2_tiny.encoder(enc_frames)

        onnx_path = str(tmp_path / "encoder.onnx")
        torch.onnx.export(
            v2_tiny.encoder,
            (enc_frames,),
            onnx_path,
            input_names=["enc_frames"],
            output_names=["enc_output"],
            dynamic_axes={
                "enc_frames": {1: "seq_len"},
                "enc_output": {1: "seq_len"},
            },
            opset_version=17,
        )

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"enc_frames": enc_frames.numpy()})[0]

        max_diff = abs(pt_out.numpy() - ort_out).max()
        assert max_diff < 1e-4, f"Encoder ONNX diff {max_diff:.6e} > 1e-4"

    def test_decoder_export(self, v2_tiny, tmp_path):
        import onnxruntime as ort
        torch.manual_seed(42)

        audio = torch.randn(1, 80000)
        with torch.no_grad():
            enc_output, _ = v2_tiny.encode(audio)
            adapted = v2_tiny.adapter(enc_output)

        tokens = torch.tensor([[v2_tiny.config.sos_eos_token_id]])
        pt_out = v2_tiny.decoder(tokens, adapted)
        pt_logits = v2_tiny.lm_head(pt_out)

        class DecoderWrapper(torch.nn.Module):
            def __init__(self, decoder, lm_head):
                super().__init__()
                self.decoder = decoder
                self.lm_head = lm_head

            def forward(self, tokens, enc_output):
                dec_out = self.decoder(tokens, enc_output)
                return self.lm_head(dec_out)

        wrapper = DecoderWrapper(v2_tiny.decoder, v2_tiny.lm_head)
        wrapper.eval()

        onnx_path = str(tmp_path / "decoder.onnx")
        torch.onnx.export(
            wrapper,
            (tokens, adapted),
            onnx_path,
            input_names=["tokens", "enc_output"],
            output_names=["logits"],
            dynamic_axes={
                "tokens": {1: "tok_len"},
                "enc_output": {1: "enc_len"},
                "logits": {1: "tok_len"},
            },
            opset_version=17,
        )

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            None,
            {
                "tokens": tokens.numpy(),
                "enc_output": adapted.numpy(),
            },
        )[0]

        max_diff = abs(pt_logits.detach().numpy() - ort_out).max()
        assert max_diff < 1e-4, f"Decoder ONNX diff {max_diff:.6e} > 1e-4"

    def test_full_pipeline_export(self, v2_tiny, tmp_path):
        import onnxruntime as ort
        torch.manual_seed(42)

        audio = torch.randn(1, 80000)
        with torch.no_grad():
            pt_logits, _, _ = v2_tiny(audio, torch.tensor([[v2_tiny.config.sos_eos_token_id]]))

        class FullModel(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, audio, tokens):
                enc_output, _ = self.model.encode(audio)
                logits = self.model.decode(tokens, enc_output)
                return logits

        wrapper = FullModel(v2_tiny)
        wrapper.eval()

        onnx_path = str(tmp_path / "full.onnx")
        tokens = torch.tensor([[v2_tiny.config.sos_eos_token_id]])
        torch.onnx.export(
            wrapper,
            (audio, tokens),
            onnx_path,
            input_names=["audio", "tokens"],
            output_names=["logits"],
            dynamic_axes={
                "audio": {0: "batch", 1: "audio_len"},
                "tokens": {0: "batch", 1: "tok_len"},
                "logits": {0: "batch", 1: "tok_len"},
            },
            opset_version=17,
        )

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_out = sess.run(
            None,
            {"audio": audio.numpy(), "tokens": tokens.numpy()},
        )[0]

        max_diff = abs(pt_logits.detach().numpy() - ort_out).max()
        assert max_diff < 1e-4, f"Full pipeline ONNX diff {max_diff:.6e} > 1e-4"


class TestT15ONNXStreaming:
    @pytest.fixture
    def v2_tiny(self):
        model = RuMoonshine(PRESETS["v2_tiny"])
        model.eval()
        return model

    def test_encoder_full_onnx_matches_pytorch(self, v2_tiny, tmp_path):
        import onnxruntime as ort
        torch.manual_seed(42)

        audio = torch.randn(80000)
        with torch.no_grad():
            enc_output, _ = v2_tiny.encode(audio.unsqueeze(0))
            with torch.amp.autocast("cuda", enabled=False):
                enc_frames, _ = v2_tiny.preprocessor(audio.unsqueeze(0), None)

        onnx_path = str(tmp_path / "encoder.onnx")
        torch.onnx.export(
            v2_tiny.encoder,
            (enc_frames,),
            onnx_path,
            input_names=["enc_frames"],
            output_names=["enc_output"],
            dynamic_axes={
                "enc_frames": {1: "seq_len"},
                "enc_output": {1: "seq_len"},
            },
            opset_version=17,
        )

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_full = sess.run(None, {"enc_frames": enc_frames.numpy()})[0]

        max_diff = abs(enc_output.detach().numpy() - ort_full).max()
        assert max_diff < 1e-4, f"ONNX full vs PyTorch diff {max_diff:.6e} > 1e-4"

    def test_decoder_onnx_chunked(self, v2_tiny, tmp_path):
        import onnxruntime as ort
        torch.manual_seed(42)

        audio = torch.randn(80000)
        with torch.no_grad():
            enc_output, _ = v2_tiny.encode(audio.unsqueeze(0))
            adapted = v2_tiny.adapter(enc_output)

        class DecoderWrapper(torch.nn.Module):
            def __init__(self, decoder, lm_head):
                super().__init__()
                self.decoder = decoder
                self.lm_head = lm_head

            def forward(self, tokens, enc_output):
                dec_out = self.decoder(tokens, enc_output)
                return self.lm_head(dec_out)

        wrapper = DecoderWrapper(v2_tiny.decoder, v2_tiny.lm_head)
        wrapper.eval()

        onnx_path = str(tmp_path / "decoder.onnx")
        tokens = torch.tensor([[v2_tiny.config.sos_eos_token_id]])
        torch.onnx.export(
            wrapper,
            (tokens, adapted),
            onnx_path,
            input_names=["tokens", "enc_output"],
            output_names=["logits"],
            dynamic_axes={
                "tokens": {0: "batch", 1: "tok_len"},
                "enc_output": {0: "batch", 1: "enc_len"},
                "logits": {0: "batch", 1: "tok_len"},
            },
            opset_version=17,
        )

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

        with torch.no_grad():
            pt_logits = wrapper(tokens, adapted)

        ort_logits = sess.run(
            None,
            {
                "tokens": tokens.numpy(),
                "enc_output": adapted.numpy(),
            },
        )[0]

        max_diff = abs(pt_logits.detach().numpy() - ort_logits).max()
        assert max_diff < 1e-4, f"Decoder ONNX diff {max_diff:.6e} > 1e-4"


class TestT18INT8Quantization:
    @pytest.fixture
    def v2_tiny(self):
        model = RuMoonshine(PRESETS["v2_tiny"])
        model.eval()
        return model

    def test_int8_encoder_quality(self, v2_tiny, tmp_path):
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnxruntime as ort
        import numpy as np

        torch.manual_seed(42)
        fp32_path = str(tmp_path / "encoder_fp32.onnx")
        int8_path = str(tmp_path / "encoder_int8.onnx")

        audio = torch.randn(1, 80000)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                enc_frames, _ = v2_tiny.preprocessor(audio, None)

        torch.onnx.export(
            v2_tiny.encoder,
            (enc_frames,),
            fp32_path,
            input_names=["enc_frames"],
            output_names=["enc_output"],
            dynamic_axes={
                "enc_frames": {1: "seq_len"},
                "enc_output": {1: "seq_len"},
            },
            opset_version=17,
        )

        quantize_dynamic(
            fp32_path,
            int8_path,
            weight_type=QuantType.QInt8,
        )

        fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
        int8_sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

        inputs = enc_frames.numpy()
        fp32_out = fp32_sess.run(None, {"enc_frames": inputs})[0]
        int8_out = int8_sess.run(None, {"enc_frames": inputs})[0]

        cos_sim = np.sum(fp32_out * int8_out, axis=-1) / (
            np.linalg.norm(fp32_out, axis=-1) * np.linalg.norm(int8_out, axis=-1) + 1e-8
        )
        assert cos_sim.mean() > 0.95, f"INT8 mean cosine sim {cos_sim.mean():.4f} < 0.95"

        fp32_size = os.path.getsize(fp32_path)
        int8_size = os.path.getsize(int8_path)
        assert int8_size < fp32_size, f"INT8 ({int8_size}B) not smaller than FP32 ({fp32_size}B)"
        assert int8_size < fp32_size * 0.5, f"INT8 ({int8_size}B) not < 50% of FP32 ({fp32_size}B)"
