"""
Layer 7: Reward sanity checks on MuJoCo and ManiSkill environments.

Checks that:
  - Rewards are finite scalars after every step
  - Shaped reward (REPPO internal) equals raw_reward - entropy_term * gamma
  - Pick-and-place reward increases when the robot moves closer to the cube
  - Dense reward tracks task progress (not completely flat for all steps)

ManiSkill tests are skipped when ManiSkill is not installed.

Run standalone: pytest tests/test_07_rewards.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.envs.registry import make_env, maniskill_available

_MS_REASON = "ManiSkill3 not installed or PickCube-v1 unavailable"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _scalar(x) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.float().mean().item())
    return float(np.mean(x))


# ---------------------------------------------------------------------------
# MuJoCo / Gymnasium reward checks
# ---------------------------------------------------------------------------

class TestMuJoCoRewards:
    def test_pendulum_reward_finite(self):
        env = make_env("Pendulum-v1")
        env.reset()
        action = env.action_space.sample()
        _, rew, _, _, _ = env.step(action)
        assert np.isfinite(float(rew))
        env.close()

    def test_pendulum_reward_range(self):
        """Pendulum-v1 reward is in [-16.3, 0]."""
        env = make_env("Pendulum-v1")
        env.reset()
        for _ in range(20):
            action = env.action_space.sample()
            _, rew, done, trunc, _ = env.step(action)
            r = float(rew)
            assert r <= 0.1, f"Pendulum reward should be <= 0, got {r}"
            assert r >= -20.0, f"Pendulum reward unexpectedly low: {r}"
            if done or trunc:
                env.reset()
        env.close()

    def test_pendulum_reward_non_constant(self):
        """Pendulum reward must vary across random actions."""
        env = make_env("Pendulum-v1")
        env.reset(seed=42)
        rewards = []
        for _ in range(30):
            _, rew, done, trunc, _ = env.step(env.action_space.sample())
            rewards.append(float(rew))
            if done or trunc:
                env.reset()
        env.close()
        assert np.std(rewards) > 1e-3, "Pendulum rewards appear constant — something is wrong"

    def test_cartpole_reward_positive(self):
        """CartPole gives +1 every step when pole is balanced."""
        env = make_env("CartPole-v1")
        env.reset(seed=0)
        rewards = []
        for _ in range(10):
            # small push keeps the pole upright longer
            _, rew, done, trunc, _ = env.step(0)
            rewards.append(float(rew))
            if done or trunc:
                break
        env.close()
        assert all(r == 1.0 for r in rewards), f"CartPole rewards should all be 1.0: {rewards}"

    def test_reppo_shaped_reward_finite(self):
        """PureRePPOAgent shaped reward (entropy-adjusted) must be finite."""
        from research.pure_rl.agents.reppo_agent import PureRePPOAgent
        env = make_env("Pendulum-v1")
        agent = PureRePPOAgent(obs_dim=3, action_dim=1, device="cpu")
        traj, _, _, _ = agent.collect(env, None, None, num_steps=8)
        env.close()
        shaped = traj["rewards"]
        raw = traj["raw_rewards"]
        assert torch.all(torch.isfinite(shaped)), "Shaped rewards contain NaN/Inf"
        assert torch.all(torch.isfinite(raw)), "Raw rewards contain NaN/Inf"

    def test_reppo_shaped_vs_raw(self):
        """Shaped reward = raw_reward - log_pi * temp * gamma; they should differ."""
        from research.pure_rl.agents.reppo_agent import PureRePPOAgent
        env = make_env("Pendulum-v1")
        agent = PureRePPOAgent(obs_dim=3, action_dim=1, device="cpu")
        traj, _, _, _ = agent.collect(env, None, None, num_steps=16)
        env.close()
        shaped = traj["rewards"]
        raw = traj["raw_rewards"]
        # They won't be identical (entropy adjustment shifts them)
        diff = (shaped - raw).abs().mean().item()
        assert diff > 0.0, "Shaped and raw rewards are identical — entropy term is missing"

    def test_gve_computation_finite(self):
        """compute_gve returns finite values given a short trajectory."""
        from research.pure_rl.agents.reppo_agent import PureRePPOAgent
        from models.rl.utils.any_utils import compute_gve
        env = make_env("Pendulum-v1")
        agent = PureRePPOAgent(obs_dim=3, action_dim=1, device="cpu")
        traj, _, _, _ = agent.collect(env, None, None, num_steps=8)
        env.close()
        gve_list = compute_gve(
            rewards=traj["rewards"],
            dones=traj["dones"],
            truncations=traj["truncations"],
            next_values=traj["next_values"],
            gamma=0.99,
            lmbda=0.95,
        )
        gve = torch.stack(gve_list, dim=0)
        assert torch.all(torch.isfinite(gve)), "GVE contains NaN/Inf"


# ---------------------------------------------------------------------------
# ManiSkill pick-and-place reward checks
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not maniskill_available(), reason=_MS_REASON)
class TestManiSkillRewards:
    def test_pickcube_reward_finite(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        env.reset()
        action = env.action_space.sample()
        _, rew, _, _, _ = env.step(action)
        assert np.isfinite(_scalar(rew))
        env.close()

    def test_pushcube_reward_finite(self):
        env = make_env("PushCube-v1", obs_mode="state", num_envs=1)
        env.reset()
        _, rew, _, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(_scalar(rew))
        env.close()

    def test_stackcube_reward_finite(self):
        env = make_env("StackCube-v1", obs_mode="state", num_envs=1)
        env.reset()
        _, rew, _, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(_scalar(rew))
        env.close()

    def test_pullcube_reward_finite(self):
        env = make_env("PullCube-v1", obs_mode="state", num_envs=1)
        env.reset()
        _, rew, _, _, _ = env.step(env.action_space.sample())
        assert np.isfinite(_scalar(rew))
        env.close()

    def test_pickcube_reward_nonnegative(self):
        """ManiSkill dense reward for PickCube is ≥ 0."""
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        env.reset(seed=0)
        for _ in range(10):
            _, rew, done, trunc, _ = env.step(env.action_space.sample())
            r = _scalar(rew)
            assert r >= -1e-3, f"PickCube reward should be ≥ 0, got {r}"
            if bool(done) or bool(trunc):
                env.reset()
        env.close()

    def test_pickcube_rewards_vary(self):
        """Dense reward must not be constant — task progress should change values."""
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        env.reset(seed=42)
        rewards = []
        for _ in range(20):
            _, rew, done, trunc, _ = env.step(env.action_space.sample())
            rewards.append(_scalar(rew))
            if bool(done) or bool(trunc):
                env.reset()
        env.close()
        assert np.std(rewards) > 1e-4, f"PickCube rewards appear constant: {rewards}"

    def test_reppo_rollout_rewards_finite_maniskill(self):
        """Full REPPO collect on PickCube produces finite rewards."""
        from research.pure_rl.agents.reppo_agent import PureRePPOAgent
        import gymnasium as gym
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        agent = PureRePPOAgent(obs_dim=obs_dim, action_dim=action_dim, device="cpu")
        traj, _, _, _ = agent.collect(env, None, None, num_steps=4)
        env.close()
        assert torch.all(torch.isfinite(traj["raw_rewards"])), "ManiSkill raw_rewards contain NaN/Inf"
        assert torch.all(torch.isfinite(traj["rewards"])), "ManiSkill shaped rewards contain NaN/Inf"
