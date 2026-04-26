import torch
import pytest

from models.config import PRESETS
from models.model import RuMoonshine


@pytest.fixture
def v2_tiny():
    return RuMoonshine(PRESETS["v2_tiny"])


@pytest.fixture
def v21_tiny():
    return RuMoonshine(PRESETS["v21_tiny"])


class TestT1ForwardPass:
    def test_v2_tiny_forward(self, v2_tiny):
        audio = torch.randn(2, 80000)
        tokens = torch.randint(0, 256, (2, 20))
        v2_tiny.eval()
        with torch.no_grad():
            logits, stats, weight = v2_tiny(audio, tokens)
        assert logits.shape[0] == 2
        assert logits.shape[2] == 256
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"

    def test_v21_tiny_forward(self, v21_tiny):
        audio = torch.randn(2, 80000)
        tokens = torch.randint(0, 256, (2, 20))
        v21_tiny.eval()
        with torch.no_grad():
            logits, stats, weight = v21_tiny(audio, tokens)
        assert logits.shape[0] == 2
        assert logits.shape[2] == 256
        assert not torch.isnan(logits).any(), "NaN in logits"
        assert not torch.isinf(logits).any(), "Inf in logits"


class TestT2BackwardPass:
    def test_v2_tiny_backward(self, v2_tiny):
        audio = torch.randn(2, 80000)
        tokens = torch.randint(0, 256, (2, 20))
        loss, stats, weight = v2_tiny(audio, tokens)
        loss.backward()
        for name, p in v2_tiny.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
        grad_norm = torch.nn.utils.clip_grad_norm_(v2_tiny.parameters(), 1e6)
        assert grad_norm.isfinite(), f"Non-finite grad norm: {grad_norm}"
        assert grad_norm < 1e6, f"Exploding grad norm: {grad_norm}"

    def test_v21_tiny_backward(self, v21_tiny):
        v21_tiny.encoder.stochastic_depth = False
        audio = torch.randn(2, 80000)
        tokens = torch.randint(0, 256, (2, 20))
        loss, stats, weight = v21_tiny(audio, tokens)
        loss.backward()
        for name, p in v21_tiny.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN grad for {name}"
        grad_norm = torch.nn.utils.clip_grad_norm_(v21_tiny.parameters(), 1e6)
        assert grad_norm.isfinite(), f"Non-finite grad norm: {grad_norm}"


class TestT3SlidingWindowMask:
    def test_window_16_0(self):
        from models.masks import make_sliding_window_mask
        mask = make_sliding_window_mask(20, 16, 0, torch.device("cpu"))
        assert mask.shape == (1, 1, 20, 20)
        for t in range(20):
            allowed = mask[0, 0, t]
            assert allowed[t] == 0.0, f"Position {t} should see itself"
            left_boundary = max(0, t - 16)
            assert allowed[left_boundary] == 0.0, f"Position {t} should see left boundary {left_boundary}"
            if left_boundary > 0:
                assert allowed[left_boundary - 1] == float("-inf"), f"Position {t} should NOT see {left_boundary - 1}"
            if t < 19:
                assert allowed[t + 1] == float("-inf"), f"Position {t} should NOT see future (t+1)"

    def test_window_16_4(self):
        from models.masks import make_sliding_window_mask
        mask = make_sliding_window_mask(20, 16, 4, torch.device("cpu"))
        for t in range(20):
            allowed = mask[0, 0, t]
            assert allowed[t] == 0.0
            right_edge = min(20, t + 5)
            for f in range(t + 1, right_edge):
                assert allowed[f] == 0.0, f"Position {t} should see future {f} (within right window)"
            if right_edge < 20:
                assert allowed[right_edge] == float("-inf"), f"Position {t} should NOT see {right_edge}"


class TestT4FeatureExtraction:
    def test_preprocessor_output(self):
        from models.preprocessor import Preprocessor
        from models.config import PRESETS
        config = PRESETS["v2_tiny"]
        prep = Preprocessor(config)
        audio = torch.randn(1, 80000)
        out, lengths = prep(audio)
        assert out.dim() == 3, f"Expected 3D output, got {out.dim()}D"
        assert out.shape[0] == 1
        assert out.shape[2] == config.enc_dim
        assert not torch.isnan(out).any(), "NaN in features"
        assert not torch.isinf(out).any(), "Inf in features"

    def test_preprocessor_output_length(self):
        from models.preprocessor import Preprocessor
        from models.config import PRESETS
        config = PRESETS["v2_tiny"]
        prep = Preprocessor(config)
        length = prep.output_length(80000)
        assert length > 0
        audio = torch.randn(1, 80000)
        out, _ = prep(audio)
        assert out.shape[1] == length, f"Expected {length} frames, got {out.shape[1]}"
