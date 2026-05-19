import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TD3BCConfig
from .networks import Actor, Critic
from .replay_buffer import OfflineReplayBuffer


class TD3BCAgent:
    """
    TD3+BC for offline continuous-control learning.

    Actor loss:
      -lambda * Q(s, pi(s)) + MSE(pi(s), dataset_action)

    The Q scale lambda follows the TD3+BC paper convention:
      lambda = alpha / mean(abs(Q))
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        device: str = "cuda",
        config: Optional[TD3BCConfig] = None,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.config = config or TD3BCConfig()

        if action_low is None:
            action_low = -np.ones(action_dim, dtype=np.float32)
        if action_high is None:
            action_high = np.ones(action_dim, dtype=np.float32)
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)

        self.actor = Actor(
            observation_dim, action_dim, self.action_low, self.action_high, self.config.hidden_dim,
        ).to(self.device)
        self.actor_target = Actor(
            observation_dim, action_dim, self.action_low, self.action_high, self.config.hidden_dim,
        ).to(self.device)
        self.critic = Critic(observation_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic_target = Critic(observation_dim, action_dim, self.config.hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        self.total_updates = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(self, observation: np.ndarray, normalize_fn=None) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        if normalize_fn is not None:
            obs = normalize_fn(obs)
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        action = self.actor(obs_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, replay_buffer: OfflineReplayBuffer) -> Dict[str, float]:
        cfg = self.config
        batch = replay_buffer.sample(cfg.batch_size)
        obs = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_observations"]
        terminals = batch["terminals"]

        action_scale = torch.as_tensor((self.action_high - self.action_low) / 2.0, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(actions) * (cfg.policy_noise * action_scale)
            noise = torch.clamp(noise, -cfg.noise_clip * action_scale, cfg.noise_clip * action_scale)
            next_actions = self.actor_target(next_obs) + noise
            low = torch.as_tensor(self.action_low, device=self.device)
            high = torch.as_tensor(self.action_high, device=self.device)
            next_actions = torch.max(torch.min(next_actions, high), low)

            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - terminals) * cfg.gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        logs = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "q_mean": float(current_q1.detach().mean().cpu().item()),
        }

        if self.total_updates % cfg.policy_freq == 0:
            pi = self.actor(obs)
            q = self.critic.q1_value(obs, pi)
            q_scale = cfg.alpha / q.abs().mean().detach().clamp(min=1e-6)
            actor_loss = -q_scale * q.mean() + F.mse_loss(pi, actions)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

            logs.update({
                "actor_loss": float(actor_loss.detach().cpu().item()),
                "bc_loss": float(F.mse_loss(pi.detach(), actions).cpu().item()),
                "lambda": float(q_scale.cpu().item()),
            })

        self.total_updates += 1
        return logs

    def _soft_update(self, target: nn.Module, source: nn.Module):
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1.0 - self.config.tau)
                tp.data.add_(self.config.tau * sp.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str, replay_buffer: Optional[OfflineReplayBuffer] = None):
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        checkpoint = {
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": self.config.__dict__,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "action_low": self.action_low,
            "action_high": self.action_high,
            "total_updates": self.total_updates,
        }
        if replay_buffer is not None:
            checkpoint["obs_mean"] = replay_buffer.obs_mean
            checkpoint["obs_std"] = replay_buffer.obs_std
        torch.save(checkpoint, filepath)

    def load(self, filepath: str, load_optimizers: bool = True) -> Dict:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_updates = int(checkpoint.get("total_updates", 0))
        return checkpoint
