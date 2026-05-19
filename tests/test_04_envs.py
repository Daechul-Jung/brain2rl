"""
Layer 4: Environment registry — make_env() factory and basic step tests.

Run standalone: pytest tests/test_04_envs.py -v
ManiSkill and OpenArm tests are skipped automatically when libraries are not installed.

ManiSkill 3.0 notes:
  - Observations and rewards are returned as torch.Tensors with shape (N, ...).
  - All environments use num_envs=1 here for simplicity.
"""

import numpy as np
import pytest
import torch
import gymnasium as gym

from research.envs.registry import make_env, maniskill_available


# ---------------------------------------------------------------------------
# Standard Gymnasium (always available)
# ---------------------------------------------------------------------------

class TestGymnasiumEnvs:
    @pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])
    def test_make_env_no_error(self, env_id):
        env = make_env(env_id)
        assert env is not None
        env.close()

    def test_reset_returns_obs(self):
        env = make_env("Pendulum-v1")
        obs, info = env.reset()
        assert obs is not None
        assert obs.shape == env.observation_space.shape
        env.close()

    def test_step_returns_5_tuple(self):
        env = make_env("Pendulum-v1")
        obs, _ = env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        env.close()

    def test_action_space_is_box(self):
        env = make_env("Pendulum-v1")
        assert isinstance(env.action_space, gym.spaces.Box)
        env.close()

    def test_full_episode_runs(self):
        env = make_env("CartPole-v1")
        obs, _ = env.reset(seed=0)
        total_reward = 0.0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, done, trunc, _ = env.step(action)
            total_reward += reward
            if done or trunc:
                break
        env.close()
        assert total_reward > 0

    def test_unknown_env_raises(self):
        with pytest.raises((ValueError, gym.error.Error)):
            make_env("ThisDoesNotExist-v99")


# ---------------------------------------------------------------------------
# ManiSkill 3.0 (skipped if not installed)
# ---------------------------------------------------------------------------

_MS_REASON = "ManiSkill3 not installed or PickCube-v1 unavailable"


@pytest.mark.skipif(not maniskill_available(), reason=_MS_REASON)
class TestManiSkillEnvs:
    """ManiSkill 3.0 returns obs/rewards as torch.Tensors with shape (N, ...)."""

    def test_make_pick_cube(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        assert env is not None
        env.close()

    def test_make_push_cube(self):
        env = make_env("PushCube-v1", obs_mode="state", num_envs=1)
        assert env is not None
        env.close()

    def test_make_stack_cube(self):
        env = make_env("StackCube-v1", obs_mode="state", num_envs=1)
        assert env is not None
        env.close()

    def test_reset_returns_tensor_obs(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        obs, info = env.reset()
        # ManiSkill 3 returns torch.Tensor (N, obs_dim)
        assert isinstance(obs, torch.Tensor)
        assert obs.ndim == 2 and obs.shape[0] == 1
        env.close()

    def test_step_returns_5_tuple(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        env.close()

    def test_reward_is_finite(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        env.reset()
        action = env.action_space.sample()
        _, rew, _, _, _ = env.step(action)
        rew_val = rew.item() if isinstance(rew, torch.Tensor) else float(rew)
        assert np.isfinite(rew_val)
        env.close()

    def test_obs_dim_consistent_across_steps(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        obs0, _ = env.reset()
        action = env.action_space.sample()
        obs1, _, _, _, _ = env.step(action)
        assert obs0.shape == obs1.shape
        env.close()

    def test_action_space_box_bounds(self):
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert np.allclose(env.action_space.low, -1.0)
        assert np.allclose(env.action_space.high, 1.0)
        env.close()

    def test_parallel_envs_cli(self):
        """num_envs>1 triggers GPU PhysX which can only init once per process.
        Test that the factory accepts the kwarg without error (env stays num_envs=1 here
        to avoid PhysX double-init across the test session).
        """
        env = make_env("PickCube-v1", obs_mode="state", num_envs=1)
        obs, _ = env.reset()
        # Single env still returns batched obs (1, obs_dim)
        assert obs.shape[0] == 1
        env.close()


# ---------------------------------------------------------------------------
# OpenArm (skipped if XML not present)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not pytest.importorskip("mujoco", reason="mujoco not installed"),
    reason="mujoco not installed",
)
class TestOpenArmEnv:
    def test_make_openarm_v0(self):
        try:
            env = make_env("OpenArm-v0")
            assert env is not None
            env.close()
        except (ImportError, FileNotFoundError) as e:
            pytest.skip(f"OpenArm env unavailable: {e}")
