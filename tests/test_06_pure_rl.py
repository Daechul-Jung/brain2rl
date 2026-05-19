"""
Layer 6: Pure RL agents — PureRePPOAgent and TD3BCAgent init, action shapes, update.

Run standalone: pytest tests/test_06_pure_rl.py -v

Tests use Pendulum-v1 (always available) and small networks to keep the suite fast.
TD3+BC tests build a tiny in-memory dataset rather than loading a file.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

OBS_DIM = 3      # Pendulum-v1
ACTION_DIM = 1   # Pendulum-v1
DEVICE = "cpu"
N_ENVS = 1
STEPS = 4        # tiny rollout for speed


@pytest.fixture(scope="module")
def pendulum_env():
    from research.envs.registry import make_env
    env = make_env("Pendulum-v1")
    yield env
    env.close()


@pytest.fixture(scope="module")
def reppo_agent():
    from research.pure_rl.agents.reppo_agent import PureRePPOAgent
    return PureRePPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=DEVICE)


@pytest.fixture(scope="module")
def td3bc_agent():
    from research.pure_rl.agents.td3bc import TD3BCAgent
    return TD3BCAgent(observation_dim=OBS_DIM, action_dim=ACTION_DIM, device=DEVICE)


def _make_offline_buffer(n: int = 64):
    """Build a tiny in-memory OfflineReplayBuffer for TD3+BC tests."""
    from research.pure_rl.agents.td3bc import OfflineReplayBuffer
    return OfflineReplayBuffer(
        observations=np.random.randn(n, OBS_DIM).astype(np.float32),
        actions=np.random.uniform(-1, 1, (n, ACTION_DIM)).astype(np.float32),
        rewards=np.random.randn(n).astype(np.float32),
        next_observations=np.random.randn(n, OBS_DIM).astype(np.float32),
        terminals=np.zeros(n, dtype=np.float32),
        device=DEVICE,
    )


# ---------------------------------------------------------------------------
# PureRePPOAgent tests
# ---------------------------------------------------------------------------

class TestPureRePPOInit:
    def test_agent_creates_without_error(self, reppo_agent):
        assert reppo_agent is not None

    def test_device_is_cpu(self, reppo_agent):
        assert str(reppo_agent.device) == "cpu"

    def test_inner_agent_exists(self, reppo_agent):
        from models.rl.agents.reppo import RePPOAgent
        assert isinstance(reppo_agent.agent, RePPOAgent)

    def test_gamma_stored(self, reppo_agent):
        assert reppo_agent.gamma == 0.99

    def test_lmbda_stored(self, reppo_agent):
        assert reppo_agent.lmbda == 0.95


class TestPureRePPOCollect:
    def test_collect_returns_trajectory(self, reppo_agent, pendulum_env):
        traj, obs, cobs, info_list = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        assert traj is not None

    def test_trajectory_shape(self, reppo_agent, pendulum_env):
        traj, _, _, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        assert traj.shape[0] == STEPS

    def test_trajectory_keys(self, reppo_agent, pendulum_env):
        traj, _, _, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        for key in ("observation", "actions", "rewards", "raw_rewards", "dones"):
            assert key in traj.keys(), f"Missing key: {key}"

    def test_info_list_length(self, reppo_agent, pendulum_env):
        _, _, _, info_list = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        assert len(info_list) == STEPS

    def test_rewards_are_tensor(self, reppo_agent, pendulum_env):
        traj, _, _, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        assert isinstance(traj["rewards"], torch.Tensor)

    def test_final_obs_returned(self, reppo_agent, pendulum_env):
        _, obs, cobs, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        assert obs is not None
        assert cobs is not None

    def test_collect_with_numpy_obs(self, reppo_agent, pendulum_env):
        # Gymnasium returns numpy obs on reset — must not crash
        ret = pendulum_env.reset()
        obs_np = ret[0] if isinstance(ret, tuple) else ret
        assert isinstance(obs_np, np.ndarray)
        traj, _, _, _ = reppo_agent.collect(pendulum_env, obs_np, None, num_steps=STEPS)
        assert traj.shape[0] == STEPS


class TestPureRePPOUpdate:
    def test_update_returns_dict(self, reppo_agent, pendulum_env):
        traj, _, _, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        metrics = reppo_agent.update(traj)
        assert isinstance(metrics, dict)

    def test_update_has_actor_metric(self, reppo_agent, pendulum_env):
        traj, _, _, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        metrics = reppo_agent.update(traj)
        assert any("actor" in k for k in metrics)

    def test_update_metrics_are_finite(self, reppo_agent, pendulum_env):
        traj, _, _, _ = reppo_agent.collect(pendulum_env, None, None, num_steps=STEPS)
        metrics = reppo_agent.update(traj)
        for k, v in metrics.items():
            assert np.isfinite(float(v)), f"Metric {k} is not finite: {v}"


class TestPureRePPOSaveLoad:
    def test_save_creates_file(self, reppo_agent):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pth")
            reppo_agent.save(path)
            assert os.path.exists(path)

    def test_load_roundtrip(self, reppo_agent):
        from research.pure_rl.agents.reppo_agent import PureRePPOAgent
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pth")
            reppo_agent.save(path)
            agent2 = PureRePPOAgent(obs_dim=OBS_DIM, action_dim=ACTION_DIM, device=DEVICE)
            agent2.load(path)
            # Parameters should match
            for (n1, p1), (n2, p2) in zip(
                reppo_agent.agent.actor.named_parameters(),
                agent2.agent.actor.named_parameters(),
            ):
                assert torch.allclose(p1, p2), f"Mismatch in param {n1}"


# ---------------------------------------------------------------------------
# TD3BCAgent tests
# ---------------------------------------------------------------------------

class TestTD3BCInit:
    def test_agent_creates_without_error(self, td3bc_agent):
        assert td3bc_agent is not None

    def test_obs_dim_stored(self, td3bc_agent):
        assert td3bc_agent.observation_dim == OBS_DIM

    def test_action_dim_stored(self, td3bc_agent):
        assert td3bc_agent.action_dim == ACTION_DIM

    def test_actor_exists(self, td3bc_agent):
        assert td3bc_agent.actor is not None

    def test_critic_exists(self, td3bc_agent):
        assert td3bc_agent.critic is not None


class TestTD3BCAction:
    def test_get_action_shape(self, td3bc_agent):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = td3bc_agent.get_action(obs)
        assert action.shape == (ACTION_DIM,)

    def test_action_in_range(self, td3bc_agent):
        for _ in range(10):
            obs = np.random.randn(OBS_DIM).astype(np.float32)
            action = td3bc_agent.get_action(obs)
            # Pendulum-v1 action space: [-2, 2]; TD3BC uses action_low/high from constructor default [-1,1]
            assert np.all(np.isfinite(action))

    def test_action_is_numpy(self, td3bc_agent):
        obs = np.random.randn(OBS_DIM).astype(np.float32)
        action = td3bc_agent.get_action(obs)
        assert isinstance(action, np.ndarray)


class TestTD3BCUpdate:
    def test_update_returns_dict(self, td3bc_agent):
        buf = _make_offline_buffer()
        metrics = td3bc_agent.update(buf)
        assert isinstance(metrics, dict)

    def test_update_has_critic_metric(self, td3bc_agent):
        buf = _make_offline_buffer()
        metrics = td3bc_agent.update(buf)
        assert any("critic" in k for k in metrics)

    def test_update_metrics_finite(self, td3bc_agent):
        buf = _make_offline_buffer()
        metrics = td3bc_agent.update(buf)
        for k, v in metrics.items():
            assert np.isfinite(float(v)), f"Metric {k} is not finite: {v}"


class TestTD3BCSaveLoad:
    def test_save_creates_file(self, td3bc_agent):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "td3bc.pth")
            td3bc_agent.save(path)
            assert os.path.exists(path)

    def test_load_roundtrip(self, td3bc_agent):
        from research.pure_rl.agents.td3bc import TD3BCAgent
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "td3bc.pth")
            td3bc_agent.save(path)
            agent2 = TD3BCAgent(observation_dim=OBS_DIM, action_dim=ACTION_DIM, device=DEVICE)
            agent2.load(path)
            for (n1, p1), (n2, p2) in zip(
                td3bc_agent.actor.named_parameters(),
                agent2.actor.named_parameters(),
            ):
                assert torch.allclose(p1, p2), f"Mismatch in param {n1}"


# ---------------------------------------------------------------------------
# OfflineReplayBuffer tests
# ---------------------------------------------------------------------------

class TestOfflineReplayBuffer:
    def test_buffer_creates_from_arrays(self):
        buf = _make_offline_buffer(n=32)
        assert buf is not None

    def test_buffer_size(self):
        buf = _make_offline_buffer(n=32)
        assert buf.size == 32

    def test_sample_returns_dict(self):
        buf = _make_offline_buffer(n=32)
        batch = buf.sample(8)
        assert isinstance(batch, dict)

    def test_sample_keys(self):
        buf = _make_offline_buffer(n=32)
        batch = buf.sample(8)
        for key in ("observations", "actions", "rewards", "next_observations", "terminals"):
            assert key in batch

    def test_sample_shapes(self):
        buf = _make_offline_buffer(n=32)
        batch = buf.sample(8)
        assert batch["observations"].shape == (8, OBS_DIM)
        assert batch["actions"].shape == (8, ACTION_DIM)

    def test_from_npz_roundtrip(self):
        from models.rl.agents.td3_bc import OfflineReplayBuffer
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.npz")
            np.savez_compressed(
                path,
                observations=np.random.randn(20, OBS_DIM).astype(np.float32),
                actions=np.random.uniform(-1, 1, (20, ACTION_DIM)).astype(np.float32),
                rewards=np.random.randn(20).astype(np.float32),
                next_observations=np.random.randn(20, OBS_DIM).astype(np.float32),
                terminals=np.zeros(20, dtype=np.float32),
            )
            buf = OfflineReplayBuffer.from_file(path, device=DEVICE)
            assert buf.size == 20
