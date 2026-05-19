"""
Layer 2: BrainConditioner interface + TransformerDelta implementation.

Run standalone: pytest tests/test_02_brain.py -v
All tests use CPU and dummy tensors — no EEG data required.
"""

import pytest
import torch

from research.brain.base import BrainConditioner
from research.brain.transformer_delta import TransformerDelta
from research.brain.diffusion_conditioned import DiffusionConditioned


TOKEN_DIM = 64
ACTION_DIM = 6
B = 4


@pytest.fixture
def brain():
    return TransformerDelta(
        token_dim=TOKEN_DIM,
        action_dim=ACTION_DIM,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.0,
        max_seq_len=256,
    )


def make_tokens(B: int, t: int, token_dim: int) -> torch.Tensor:
    return torch.randn(B, t + 1, token_dim)


class TestBrainConditionerInterface:
    def test_is_subclass_of_nn_module(self, brain):
        assert isinstance(brain, torch.nn.Module)

    def test_is_subclass_of_brain_conditioner(self, brain):
        assert isinstance(brain, BrainConditioner)

    def test_token_dim_property(self, brain):
        assert brain.token_dim == TOKEN_DIM

    def test_action_dim_property(self, brain):
        assert brain.action_dim == ACTION_DIM

    def test_forward_returns_dict(self, brain):
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        out = brain(tokens, t=5)
        assert isinstance(out, dict)

    def test_output_has_delta_action(self, brain):
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        out = brain(tokens, t=5)
        assert "delta_action" in out

    def test_output_has_alpha(self, brain):
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        out = brain(tokens, t=5)
        assert "alpha" in out

    def test_check_output_passes(self, brain):
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        out = brain(tokens, t=5)
        brain.check_output(out, B)  # should not raise


class TestTransformerDeltaShapes:
    def test_delta_shape(self, brain):
        tokens = make_tokens(B, t=0, token_dim=TOKEN_DIM)
        out = brain(tokens, t=0)
        assert out["delta_action"].shape == (B, ACTION_DIM)

    def test_alpha_shape(self, brain):
        tokens = make_tokens(B, t=0, token_dim=TOKEN_DIM)
        out = brain(tokens, t=0)
        assert out["alpha"].shape == (B, 1)

    def test_alpha_range(self, brain):
        brain.eval()
        with torch.no_grad():
            tokens = make_tokens(B, t=20, token_dim=TOKEN_DIM)
            out = brain(tokens, t=20)
        alpha = out["alpha"]
        assert (alpha >= 0.0).all() and (alpha <= 1.0).all(), f"alpha out of [0,1]: {alpha}"

    def test_various_timesteps(self, brain):
        for t in [0, 1, 5, 31, 63, 127]:
            tokens = make_tokens(B, t=t, token_dim=TOKEN_DIM)
            out = brain(tokens, t=t)
            assert out["delta_action"].shape == (B, ACTION_DIM)
            assert out["alpha"].shape == (B, 1)

    def test_causal_mask_single_token(self, brain):
        tokens = make_tokens(B, t=0, token_dim=TOKEN_DIM)
        out = brain(tokens, t=0)
        assert torch.isfinite(out["delta_action"]).all()


class TestTransformerDeltaGrad:
    def test_gradients_flow_through_brain(self, brain):
        tokens = make_tokens(B, t=10, token_dim=TOKEN_DIM)
        tokens.requires_grad_(True)
        out = brain(tokens, t=10)
        loss = out["delta_action"].mean() + out["alpha"].mean()
        loss.backward()
        assert tokens.grad is not None

    def test_output_is_finite(self, brain):
        tokens = make_tokens(B, t=10, token_dim=TOKEN_DIM)
        out = brain(tokens, t=10)
        assert torch.isfinite(out["delta_action"]).all()
        assert torch.isfinite(out["alpha"]).all()


class TestTransformerDeltaStochastic:
    def test_stochastic_has_log_std(self):
        brain_stoch = TransformerDelta(
            token_dim=TOKEN_DIM, action_dim=ACTION_DIM,
            d_model=64, n_heads=4, n_layers=1, stochastic=True
        )
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        out = brain_stoch(tokens, t=5)
        assert "log_std" in out
        assert out["log_std"].shape == (B, ACTION_DIM)

    def test_non_stochastic_no_log_std(self, brain):
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        out = brain(tokens, t=5)
        assert "log_std" not in out


class TestDiffusionConditionedPlaceholder:
    def test_placeholder_raises_not_implemented(self):
        dc = DiffusionConditioned(token_dim=TOKEN_DIM, action_dim=ACTION_DIM)
        tokens = make_tokens(B, t=5, token_dim=TOKEN_DIM)
        with pytest.raises(NotImplementedError):
            dc(tokens, t=5)

    def test_placeholder_has_correct_properties(self):
        dc = DiffusionConditioned(token_dim=TOKEN_DIM, action_dim=ACTION_DIM)
        assert dc.token_dim == TOKEN_DIM
        assert dc.action_dim == ACTION_DIM
