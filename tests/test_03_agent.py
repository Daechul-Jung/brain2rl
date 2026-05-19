"""
Layer 3: EEGRePPOAgent — initialization, action computation, save/load.

Run standalone: pytest tests/test_03_agent.py -v
Uses CPU + dummy EEG data; does NOT step a real environment.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from models.rl.agents.reppo import RePPOAgent
from models.rl.utils.reppo_network import EmpiricalNormalizer
from research.eeg.tokenizer import EEGTokenizer
from research.brain.transformer_delta import TransformerDelta
from research.agents.eeg_reppo import EEGRePPOAgent


OBS_DIM = 17
ACTION_DIM = 6
N_CHANNELS = 8
T_EEG = 256
TOKEN_DIM = 64
T_RL = 16
N_SEGS = 10  # dummy EEG dataset size
DEVICE = "cpu"


def make_reppo() -> RePPOAgent:
    norm = EmpiricalNormalizer(OBS_DIM, device=DEVICE)
    return RePPOAgent(
        observation_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        num_atoms=51,
        vmin=-100.0,
        vmax=100.0,
        device=DEVICE,
        lr=3e-4,
        obs_normalizer=norm,
        critic_obs_normalizer=norm,
    )


def make_dummy_eeg():
    segs = np.random.randn(N_SEGS, N_CHANNELS, T_EEG).astype(np.float32)
    labels = np.array([i % 4 for i in range(N_SEGS)], dtype=np.int64)
    return segs, labels


@pytest.fixture
def agent():
    reppo = make_reppo()
    tokenizer = EEGTokenizer(
        n_channels=N_CHANNELS, token_dim=TOKEN_DIM,
        hidden_dim=32, n_layers=2, dropout=0.0
    )
    brain = TransformerDelta(
        token_dim=TOKEN_DIM, action_dim=ACTION_DIM,
        d_model=64, n_heads=4, n_layers=1, max_seq_len=T_RL + 4
    )
    segs, labels = make_dummy_eeg()
    return EEGRePPOAgent(
        reppo=reppo, tokenizer=tokenizer, brain=brain,
        eeg_segments=segs, eeg_labels=labels,
        T_rl=T_RL, brain_lr=3e-4,
    )


class TestAgentInit:
    def test_agent_creates_without_error(self, agent):
        assert agent is not None

    def test_device_is_cpu(self, agent):
        assert agent.device == torch.device("cpu")

    def test_brain_on_correct_device(self, agent):
        p = next(agent.brain.parameters())
        assert p.device == torch.device("cpu")

    def test_tokenizer_on_correct_device(self, agent):
        p = next(agent.tokenizer.parameters())
        assert p.device == torch.device("cpu")


class TestEEGSampling:
    def test_sample_eeg_segment_shape(self, agent):
        seg = agent.sample_eeg_segment(label=0)
        assert seg.shape == (1, N_CHANNELS, T_EEG)

    def test_sample_different_labels(self, agent):
        for lbl in [0, 1, 2, 3]:
            seg = agent.sample_eeg_segment(label=lbl)
            assert seg.shape == (1, N_CHANNELS, T_EEG)

    def test_invalid_label_raises(self, agent):
        with pytest.raises(ValueError):
            agent.sample_eeg_segment(label=99)

    def test_tokenize_shape(self, agent):
        seg = agent.sample_eeg_segment(label=0)
        tokens = agent.tokenize(seg)
        assert tokens.shape == (1, T_RL, TOKEN_DIM)


class TestConditionedAction:
    def test_conditioned_action_shape(self, agent):
        obs = torch.randn(1, OBS_DIM)
        seg = agent.sample_eeg_segment(label=0)
        token_seq = agent.tokenize(seg)

        action, log_prob, pi = agent._conditioned_action(obs, token_seq, t=0)
        assert action.shape == (1, ACTION_DIM)
        assert log_prob.shape == (1,)

    def test_action_in_tanh_range(self, agent):
        agent.reppo.actor.eval()
        agent.brain.eval()
        with torch.no_grad():
            obs = torch.randn(2, OBS_DIM)
            seg = agent.sample_eeg_segment(label=0).expand(2, -1, -1)
            token_seq = agent.tokenize(seg)
            action, _, _ = agent._conditioned_action(obs, token_seq, t=0)
        assert (action > -1.0).all() and (action < 1.0).all()

    def test_action_is_finite(self, agent):
        obs = torch.randn(1, OBS_DIM)
        seg = agent.sample_eeg_segment(label=0)
        token_seq = agent.tokenize(seg)
        action, log_prob, _ = agent._conditioned_action(obs, token_seq, t=0)
        assert torch.isfinite(action).all()
        assert torch.isfinite(log_prob).all()


class TestSaveLoad:
    def test_save_and_load_roundtrip(self, agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pth")
            agent.save(path, step=100)
            assert os.path.exists(path)

            # Load into a fresh agent
            reppo2 = make_reppo()
            tokenizer2 = EEGTokenizer(
                n_channels=N_CHANNELS, token_dim=TOKEN_DIM,
                hidden_dim=32, n_layers=2
            )
            brain2 = TransformerDelta(
                token_dim=TOKEN_DIM, action_dim=ACTION_DIM,
                d_model=64, n_heads=4, n_layers=1, max_seq_len=T_RL + 4
            )
            segs, labels = make_dummy_eeg()
            agent2 = EEGRePPOAgent(
                reppo=reppo2, tokenizer=tokenizer2, brain=brain2,
                eeg_segments=segs, eeg_labels=labels, T_rl=T_RL,
            )
            step = agent2.load(path)
            assert step == 100

    def test_loaded_weights_match(self, agent):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "ckpt.pth")
            agent.save(path)

            reppo2 = make_reppo()
            tokenizer2 = EEGTokenizer(n_channels=N_CHANNELS, token_dim=TOKEN_DIM, hidden_dim=32, n_layers=2)
            brain2 = TransformerDelta(token_dim=TOKEN_DIM, action_dim=ACTION_DIM, d_model=64, n_heads=4, n_layers=1, max_seq_len=T_RL + 4)
            segs, labels = make_dummy_eeg()
            agent2 = EEGRePPOAgent(reppo=reppo2, tokenizer=tokenizer2, brain=brain2, eeg_segments=segs, eeg_labels=labels, T_rl=T_RL)
            agent2.load(path)

            for (n1, p1), (n2, p2) in zip(
                agent.brain.named_parameters(), agent2.brain.named_parameters()
            ):
                assert torch.allclose(p1, p2), f"Brain param mismatch at {n1}"
