"""
Pure RL REPPO agent for the research environment.

Thin wrapper around models.rl.agents.reppo.RePPOAgent that:
  - Handles ManiSkill 3.0 tensor observations (N, obs_dim) automatically
  - Provides a unified collect/update interface matching EEGRePPOAgent

Use this for pure RL baselines; switch to research.agents.eeg_reppo.EEGRePPOAgent
when you want to add EEG conditioning.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from models.rl.agents.reppo import RePPOAgent
from models.rl.utils.reppo_network import EmpiricalNormalizer
from models.rl.utils.any_utils import (
    _env_shape, _split_step_return, _to_tensor, wrap_batch_dim, compute_gve,
)

EPS = 1e-6


def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
    return t.unsqueeze(0) if t.ndim == 1 else t


def _to_float(x) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.as_tensor(x, dtype=torch.float32)


class PureRePPOAgent:
    """
    Drop-in REPPO agent compatible with both Gymnasium (numpy obs)
    and ManiSkill 3.0 (torch tensor obs).

    Args:
        obs_dim    : observation dimension
        action_dim : action dimension
        device     : 'cuda' or 'cpu'
        lr         : learning rate
        gamma      : discount factor
        lmbda      : GAE lambda
        num_atoms  : distributional Q number of atoms
        vmin/vmax  : distributional Q value range
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "cuda",
        lr: float = 3e-4,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        num_atoms: int = 151,
        vmin: float = -2500.0,
        vmax: float = 2500.0,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        norm = EmpiricalNormalizer(obs_dim, device=str(self.device))
        self.agent = RePPOAgent(
            observation_dim=obs_dim,
            action_dim=action_dim,
            num_atoms=num_atoms,
            vmin=vmin,
            vmax=vmax,
            device=str(self.device),
            lr=lr,
            gamma=gamma,
            lmbda=lmbda,
            obs_normalizer=norm,
            critic_obs_normalizer=norm,
        )
        self.gamma = gamma
        self.lmbda = lmbda

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        env,
        observation: Optional[torch.Tensor],
        critic_observation: Optional[torch.Tensor],
        num_steps: int = 128,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, list]:
        """
        On-policy rollout. Handles both numpy and torch obs (ManiSkill 3.0).

        Returns:
            trajectory, final_norm_obs, final_norm_cobs, info_list
        """
        N, _, _ = _env_shape(env)

        if observation is None:
            ret = env.reset()
            observation = ret[0] if isinstance(ret, tuple) else ret
        if critic_observation is None:
            critic_observation = observation

        observation = _ensure_2d(_to_float(_to_tensor(observation, self.device)))
        critic_observation = _ensure_2d(_to_float(_to_tensor(critic_observation, self.device)))

        trajectory, info_list = [], []

        for _ in range(num_steps):
            norm_obs = self.agent.observation_normalizer(observation)
            norm_cobs = self.agent.critic_observation_normalizer(critic_observation)

            pi, _, _, _ = self.agent._actor_forward(norm_obs)
            action = pi.sample().clamp(-1 + EPS, 1 - EPS)
            log_prob = pi.log_prob(action).sum(-1)

            env_action = action.cpu().numpy().astype(np.float32)
            if N == 1 and env_action.ndim > 1:
                env_action = env_action.squeeze(0)

            step_return = env.step(env_action)
            next_obs, rewards, dones, truncated, infos = _split_step_return(step_return)

            _next = _ensure_2d(_to_float(_to_tensor(next_obs, self.device)))
            next_norm = self.agent.observation_normalizer(_next)

            next_pi, _, next_temp, _ = self.agent._actor_forward(next_norm)
            next_action = next_pi.sample().clamp(-1 + EPS, 1 - EPS)
            next_log_prob = next_pi.log_prob(next_action).sum(-1)
            next_value, _, _, next_features = self.agent._critic_forward(next_norm, next_action)

            rewards_t = _to_float(_to_tensor(rewards, self.device)).view(-1)
            shaped_r = rewards_t - next_log_prob * next_temp * self.gamma

            obs_b, cobs_b, act_b, logp_b, rew_b, raw_b, nfeat_b, nval_b, done_b, trunc_b = (
                wrap_batch_dim(
                    norm_obs, norm_cobs, action, log_prob,
                    shaped_r, rewards_t, next_features, next_value,
                    dones, truncated, self.device,
                )
            )

            td = TensorDict(
                {
                    "observation": obs_b, "critic_observation": cobs_b,
                    "actions": act_b, "log_prob": logp_b,
                    "rewards": rew_b, "raw_rewards": raw_b,
                    "next_embedding": nfeat_b, "next_values": nval_b,
                    "dones": done_b, "truncations": trunc_b,
                },
                batch_size=(N,),
            )
            trajectory.append(td)
            info_list.append(infos)
            observation = _ensure_2d(_to_float(_to_tensor(next_obs, self.device)))
            critic_observation = observation

        return torch.stack(trajectory, dim=0), norm_obs, norm_cobs, info_list

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trajectory: TensorDict) -> dict:
        """Compute GVE and run REPPO actor + critic updates."""
        gve_list = compute_gve(
            rewards=trajectory["rewards"],
            dones=trajectory["dones"],
            truncations=trajectory["truncations"],
            next_values=trajectory["next_values"],
            gamma=self.gamma,
            lmbda=self.lmbda,
        )
        gve = torch.stack(gve_list, dim=0)
        flat = trajectory.reshape(-1)
        flat["gve"] = gve.reshape(-1, 1)

        self.agent.old_actor.load_state_dict(self.agent.actor.state_dict())
        critic_m = self.agent.update_critic(flat)
        actor_m = self.agent.update_actor(flat)
        return {**critic_m, **actor_m}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str, step: int | None = None) -> None:
        self.agent.save(path, step=step)

    def load(self, path: str) -> int | None:
        return self.agent.load(path)
