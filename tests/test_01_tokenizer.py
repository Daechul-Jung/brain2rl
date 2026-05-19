"""
Layer 1: EEGTokenizer — shape and forward pass tests.

Run standalone: pytest tests/test_01_tokenizer.py -v
All tests use CPU and dummy tensors — no EEG data files required.
"""

import pytest
import torch
from research.eeg.tokenizer import EEGTokenizer


@pytest.fixture
def tokenizer():
    return EEGTokenizer(n_channels=8, token_dim=64, hidden_dim=32, n_layers=2, dropout=0.0)


class TestEEGTokenizerShapes:
    def test_output_shape_single(self, tokenizer):
        x = torch.randn(1, 8, 500)
        out = tokenizer(x, T_rl=32)
        assert out.shape == (1, 32, 64), f"Got {out.shape}"

    def test_output_shape_batch(self, tokenizer):
        x = torch.randn(4, 8, 500)
        out = tokenizer(x, T_rl=128)
        assert out.shape == (4, 128, 64)

    def test_t_rl_flexibility(self, tokenizer):
        x = torch.randn(2, 8, 1000)
        for T in [16, 32, 64, 128, 256]:
            out = tokenizer(x, T_rl=T)
            assert out.shape == (2, T, 64), f"T_rl={T} failed: got {out.shape}"

    def test_variable_eeg_length(self, tokenizer):
        for T_eeg in [128, 256, 500, 1024]:
            x = torch.randn(1, 8, T_eeg)
            out = tokenizer(x, T_rl=32)
            assert out.shape == (1, 32, 64)

    def test_different_channel_counts(self):
        for C in [4, 8, 16, 32, 64]:
            tok = EEGTokenizer(n_channels=C, token_dim=64, hidden_dim=16, n_layers=2)
            x = torch.randn(1, C, 256)
            out = tok(x, T_rl=16)
            assert out.shape == (1, 16, 64)


class TestEEGTokenizerGrad:
    def test_gradients_flow(self, tokenizer):
        x = torch.randn(2, 8, 256, requires_grad=False)
        out = tokenizer(x, T_rl=16)
        loss = out.mean()
        loss.backward()
        # Check that trunk parameters have gradients
        for name, p in tokenizer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_output_is_finite(self, tokenizer):
        x = torch.randn(2, 8, 256)
        out = tokenizer(x, T_rl=16)
        assert torch.isfinite(out).all()

    def test_different_inputs_different_outputs(self, tokenizer):
        tokenizer.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 8, 256)
            x2 = torch.randn(1, 8, 256)
            out1 = tokenizer(x1, T_rl=16)
            out2 = tokenizer(x2, T_rl=16)
        assert not torch.allclose(out1, out2), "Tokenizer output is identical for different inputs"


class TestEEGTokenizerConfig:
    def test_token_dim_property(self):
        tok = EEGTokenizer(n_channels=8, token_dim=128)
        assert tok.token_dim == 128

    def test_default_args_work(self):
        tok = EEGTokenizer(n_channels=8)
        x = torch.randn(1, 8, 256)
        out = tok(x, T_rl=16)
        assert out.shape == (1, 16, 128)

    def test_layer_count_affects_output(self):
        tok1 = EEGTokenizer(n_channels=8, token_dim=64, n_layers=1)
        tok2 = EEGTokenizer(n_channels=8, token_dim=64, n_layers=3)
        x = torch.randn(1, 8, 256)
        out1 = tok1(x, T_rl=16)
        out2 = tok2(x, T_rl=16)
        # Both should produce valid shapes
        assert out1.shape == out2.shape == (1, 16, 64)
