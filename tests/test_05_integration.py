"""
Layer 5: Full pipeline integration — mini-rollout with dummy data.

Verifies that EEGTokenizer → TransformerDelta → EEGRePPOAgent → Gymnasium env
all wire together correctly. Runs on CPU with CartPole so no GPU or real EEG is required.

Run standalone: pytest tests/test_05_integration.py -v
"""

import numpy as np
import pytest
import torch

from models.rl.agents.reppo import RePPOAgent
from models.rl.utils.reppo_network import EmpiricalNormalizer
from research.eeg.tokenizer import EEGTokenizer
from research.brain.transformer_delta import TransformerDelta
from research.agents.eeg_reppo import EEGRePPOAgent
from research.envs.registry import make_env


DEVICE = "cpu"
N_CHANNELS = 4
T_EEG = 128
TOKEN_DIM = 32
T_RL = 8        # very short rollout for fast test
N_SEGS = 6


def _make_agent(obs_dim: int, action_dim: int) -> EEGRePPOAgent:
    norm = EmpiricalNormalizer(obs_dim, device=DEVICE)
    reppo = RePPOAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        num_atoms=51,
        vmin=-100.0,
        vmax=100.0,
        device=DEVICE,
        lr=3e-4,
        obs_normalizer=norm,
        critic_obs_normalizer=norm,
    )
    tokenizer = EEGTokenizer(
        n_channels=N_CHANNELS, token_dim=TOKEN_DIM,
        hidden_dim=16, n_layers=1, dropout=0.0
    )
    brain = TransformerDelta(
        token_dim=TOKEN_DIM, action_dim=action_dim,
        d_model=32, n_heads=4, n_layers=1, max_seq_len=T_RL + 4
    )
    segs = np.random.randn(N_SEGS, N_CHANNELS, T_EEG).astype(np.float32)
    labels = np.array([i % 2 for i in range(N_SEGS)], dtype=np.int64)
    return EEGRePPOAgent(
        reppo=reppo, tokenizer=tokenizer, brain=brain,
        eeg_segments=segs, eeg_labels=labels,
        T_rl=T_RL, brain_lr=3e-4,
    )


class TestMiniRollout:
    """Full rollout on Pendulum-v1 (continuous action space, like robot envs)."""

    def test_collect_returns_trajectory(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        traj, obs, cobs, infos, eeg_tokens = agent.collect(
            env, observation=None, critic_observation=None,
            task_label=0, num_steps=T_RL,
        )
        env.close()

        assert traj.shape[0] == T_RL, f"Expected {T_RL} steps, got {traj.shape[0]}"

    def test_trajectory_has_required_keys(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        traj, *_ = agent.collect(env, None, None, task_label=0, num_steps=T_RL)
        env.close()

        required_keys = [
            "observation", "critic_observation", "actions",
            "rewards", "dones", "truncations",
            "eeg_tokens", "eeg_timestep",
        ]
        for key in required_keys:
            assert key in traj.keys(), f"Missing key: {key}"

    def test_actions_in_valid_range(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        traj, *_ = agent.collect(env, None, None, task_label=0, num_steps=T_RL)
        env.close()

        actions = traj["actions"]
        assert (actions > -1.0).all() and (actions < 1.0).all(), "Actions outside (-1, 1)"

    def test_rewards_are_finite(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        traj, *_ = agent.collect(env, None, None, task_label=0, num_steps=T_RL)
        env.close()

        assert torch.isfinite(traj["rewards"]).all()

    def test_eeg_tokens_shape(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        _, _, _, _, eeg_tokens = agent.collect(env, None, None, task_label=0, num_steps=T_RL)
        env.close()

        assert eeg_tokens.shape == (1, T_RL, TOKEN_DIM)


class TestGradientFlow:
    """Verify RL loss backpropagates through brain + tokenizer."""

    def test_brain_params_receive_gradients_after_update(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        traj, *_, eeg_tokens = agent.collect(env, None, None, task_label=0, num_steps=T_RL)
        env.close()

        eeg_seg = agent.sample_eeg_segment(label=0)

        # Zero all grads before update
        agent.brain_optimizer.zero_grad()
        for p in agent.brain.parameters():
            p.grad = None
        for p in agent.tokenizer.parameters():
            p.grad = None

        metrics = agent.update(traj, eeg_seg)

        brain_has_grad = any(
            p.grad is not None for p in agent.brain.parameters() if p.requires_grad
        )
        tokenizer_has_grad = any(
            p.grad is not None for p in agent.tokenizer.parameters() if p.requires_grad
        )

        assert brain_has_grad, "Brain conditioner received no gradients from RL loss"
        assert tokenizer_has_grad, "EEGTokenizer received no gradients from RL loss"

    def test_update_returns_metrics_dict(self):
        env = make_env("Pendulum-v1")
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = _make_agent(obs_dim, action_dim)

        traj, *_ = agent.collect(env, None, None, task_label=0, num_steps=T_RL)
        env.close()
        eeg_seg = agent.sample_eeg_segment(label=0)
        metrics = agent.update(traj, eeg_seg)

        assert "brain_loss" in metrics
        assert "alpha_mean" in metrics
        assert "qf_loss" in metrics
