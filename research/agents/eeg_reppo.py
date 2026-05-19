"""
EEGRePPOAgent — REPPO with a pluggable BrainConditioner.

Architecture (ARCHITECTURE.md Data Flow):
  1. At the start of each episode, tokenize the EEG segment → (T_rl, token_dim).
  2. At step t:
       token_seq = eeg_tokens[:t+1]       # tokens seen so far
       delta, alpha = brain(token_seq, t)  # conditioner output
       base_pi = reppo.actor(obs)          # base action distribution
       final_mean = base_mean + alpha * delta
       action ~ TanhNormal(final_mean, base_log_std)
  3. RL loss backprops through: actor + critic + brain conditioner + EEG tokenizer.

End-to-end backprop (ADR-008):
  During update(), raw EEG segments stored in the trajectory are re-tokenized
  inside the computation graph, so gradients flow back to EEGTokenizer parameters.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.rl.agents.reppo import RePPOAgent
from models.rl.utils.any_utils import (
    _env_shape, _split_step_return, _to_tensor, wrap_batch_dim, compute_gve,
)
from research.eeg.tokenizer import EEGTokenizer
from research.brain.base import BrainConditioner


class EEGRePPOAgent:
    """
    Wraps RePPOAgent to inject EEG brain conditioning at every step.

    The brain conditioner is fully swappable: pass any BrainConditioner subclass.
    Gradients from the RL loss flow through brain → tokenizer (end-to-end).

    Args:
        reppo        : configured RePPOAgent instance
        tokenizer    : EEGTokenizer (Conv1D)
        brain        : BrainConditioner implementation (e.g., TransformerDelta)
        eeg_segments : (N_seg, C, T_eeg) — pre-loaded EEG segments for training
        eeg_labels   : (N_seg,) int — task labels per segment
        T_rl         : RL rollout horizon (= number of EEG tokens produced per segment)
        brain_lr     : learning rate for brain + tokenizer parameters
    """

    EPS = 1e-6

    def __init__(
        self,
        reppo: RePPOAgent,
        tokenizer: EEGTokenizer,
        brain: BrainConditioner,
        eeg_segments: Optional[torch.Tensor] = None,
        eeg_labels: Optional[np.ndarray] = None,
        T_rl: int = 128,
        brain_lr: float = 3e-4,
    ):
        self.reppo = reppo
        self.tokenizer = tokenizer.to(reppo.device)
        self.brain = brain.to(reppo.device)
        self.device = reppo.device
        self.T_rl = T_rl

        # EEG dataset (indexed by label for quick sampling)
        self._seg_by_label: dict[int, torch.Tensor] = {}
        if eeg_segments is not None and eeg_labels is not None:
            for lbl in np.unique(eeg_labels):
                mask = eeg_labels == int(lbl)
                self._seg_by_label[int(lbl)] = torch.from_numpy(
                    eeg_segments[mask].astype(np.float32)
                )

        # Separate optimizer for brain + tokenizer so RL agent LR doesn't interfere
        self.brain_optimizer = torch.optim.AdamW(
            list(self.tokenizer.parameters()) + list(self.brain.parameters()),
            lr=brain_lr,
        )

    # ------------------------------------------------------------------
    # EEG helpers
    # ------------------------------------------------------------------

    def sample_eeg_segment(self, label: int = 0) -> torch.Tensor:
        """Return a random EEG segment (1, C, T_eeg) for the given task label."""
        pool = self._seg_by_label.get(label)
        if pool is None:
            raise ValueError(
                f"No EEG segments for label {label}. "
                f"Available labels: {list(self._seg_by_label.keys())}"
            )
        idx = torch.randint(len(pool), (1,)).item()
        return pool[idx].unsqueeze(0).to(self.device)  # (1, C, T_eeg)

    def tokenize(self, eeg_seg: torch.Tensor) -> torch.Tensor:
        """(B, C, T_eeg) → (B, T_rl, token_dim). Runs inside grad graph when needed."""
        return self.tokenizer(eeg_seg, self.T_rl)

    # ------------------------------------------------------------------
    # Action computation
    # ------------------------------------------------------------------

    def _conditioned_action(
        self,
        norm_obs: torch.Tensor,
        token_seq: torch.Tensor,
        t: int,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        """
        Compute final action by combining REPPO base distribution with brain delta.

        Returns:
            action      : (B, action_dim) sampled and clamped
            log_prob    : (B,) log probability under the base distribution
            pi          : the base TanhNormal distribution (for further queries)
        """
        pi, base_mean, _, _ = self.reppo._actor_forward(norm_obs)
        base_action = pi.sample()

        brain_out = self.brain(token_seq[:, : t + 1, :], t)
        delta = brain_out["delta_action"]            # (B, action_dim)
        alpha = brain_out.get("alpha", torch.ones(norm_obs.shape[0], 1, device=self.device))

        final_action = torch.tanh(base_action + alpha * delta)
        final_action = final_action.clamp(-1 + self.EPS, 1 - self.EPS)
        log_prob = pi.log_prob(final_action).sum(-1)  # (B,)
        return final_action, log_prob, pi

    # ------------------------------------------------------------------
    # Rollout (collect)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        env,
        observation: Optional[torch.Tensor],
        critic_observation: Optional[torch.Tensor],
        task_label: int = 0,
        num_steps: int = 128,
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, list, torch.Tensor]:
        """
        On-policy rollout with EEG conditioning.

        Returns:
            trajectory      : TensorDict (T, N) — also stores 'eeg_segment' for re-tokenization
            final_norm_obs
            final_norm_cobs
            info_list
            eeg_tokens      : (1, T_rl, token_dim) — pre-computed for this episode
        """
        N, _, _ = _env_shape(env)

        # Sample and tokenize EEG for this episode (no grad during collect)
        eeg_seg = self.sample_eeg_segment(task_label)          # (1, C, T_eeg)
        eeg_tokens = self.tokenize(eeg_seg)                    # (1, T_rl, token_dim)
        # Expand to batch size
        eeg_tokens_batch = eeg_tokens.expand(N, -1, -1)       # (N, T_rl, token_dim)

        if observation is None:
            ret = env.reset()
            observation = ret[0] if isinstance(ret, tuple) else ret
        if critic_observation is None:
            critic_observation = observation

        def _ensure_2d(t: torch.Tensor) -> torch.Tensor:
            return t.unsqueeze(0) if t.ndim == 1 else t

        observation = _ensure_2d(_to_tensor(observation, self.device))
        critic_observation = _ensure_2d(_to_tensor(critic_observation, self.device))

        trajectory = []
        info_list = []

        for t in range(num_steps):
            norm_obs = self.reppo.observation_normalizer(observation)
            norm_cobs = self.reppo.critic_observation_normalizer(critic_observation)

            action, log_prob, pi = self._conditioned_action(norm_obs, eeg_tokens_batch, t)
            # Send to env: squeeze batch dim for single-env, keep for multi-env
            action_np = action.cpu().numpy().astype(np.float32)
            env_action = action_np.squeeze(0) if N == 1 and action_np.ndim > 1 else action_np

            step_return = env.step(env_action)
            next_obs, rewards, dones, truncated, infos = _split_step_return(step_return)
            next_cobs = next_obs

            _next_obs = _ensure_2d(_to_tensor(next_obs, self.device))
            _next_cobs = _ensure_2d(_to_tensor(next_cobs, self.device))
            next_norm_obs = self.reppo.observation_normalizer(_next_obs)
            next_norm_cobs = self.reppo.critic_observation_normalizer(_next_cobs)

            next_pi, _, next_temp, _ = self.reppo._actor_forward(next_norm_obs)
            next_action = next_pi.sample().clamp(-1 + self.EPS, 1 - self.EPS)
            next_log_prob = next_pi.log_prob(next_action).sum(-1)
            next_value, _, _, next_features = self.reppo._critic_forward(next_norm_cobs, next_action)

            rewards_t = _to_tensor(rewards, self.device).view(-1)
            shaped_r = rewards_t - next_log_prob * next_temp * self.reppo.gamma

            obs_b, cobs_b, act_b, logp_b, rew_b, raw_b, nfeat_b, nval_b, done_b, trunc_b = (
                wrap_batch_dim(
                    norm_obs, norm_cobs, action, log_prob,
                    shaped_r, rewards_t, next_features, next_value,
                    dones, truncated, self.device,
                )
            )

            td = TensorDict(
                {
                    "observation":        obs_b,
                    "critic_observation": cobs_b,
                    "actions":            act_b,
                    "log_prob":           logp_b,
                    "rewards":            rew_b,
                    "raw_rewards":        raw_b,
                    "next_embedding":     nfeat_b,
                    "next_values":        nval_b,
                    "dones":              done_b,
                    "truncations":        trunc_b,
                    # Store EEG segment index for re-tokenization during update
                    "eeg_tokens":         eeg_tokens_batch.detach(),
                    "eeg_timestep":       torch.tensor([t], device=self.device).expand(N),
                },
                batch_size=(N,),
            )
            trajectory.append(td)
            info_list.append(infos)

            observation = _ensure_2d(_to_tensor(next_obs, self.device))
            critic_observation = _ensure_2d(_to_tensor(next_cobs, self.device))

        return torch.stack(trajectory, dim=0), norm_obs, norm_cobs, info_list, eeg_tokens

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trajectory: TensorDict, eeg_seg: torch.Tensor) -> dict:
        """
        Update REPPO + brain conditioner + tokenizer from the collected trajectory.

        eeg_seg : (1, C, T_eeg) — the EEG segment used in this rollout.
                  Re-tokenized here inside the grad graph for end-to-end backprop.

        Returns merged metrics dict.
        """
        # Compute GVE targets for the whole trajectory
        gve_list = compute_gve(
            rewards=trajectory["rewards"],
            dones=trajectory["dones"],
            truncations=trajectory["truncations"],
            next_values=trajectory["next_values"],
            gamma=self.reppo.gamma,
            lmbda=self.reppo.lmbda,
        )
        gve = torch.stack(gve_list, dim=0)  # (T, N, 1)
        flat = trajectory.reshape(-1)
        flat["gve"] = gve.reshape(-1, 1)

        # --- REPPO actor / critic updates (standard) ---
        critic_metrics = self.reppo.update_critic(flat)
        self.reppo.old_actor.load_state_dict(self.reppo.actor.state_dict())
        actor_metrics = self.reppo.update_actor(flat)

        # --- Brain conditioner + tokenizer update ---
        # Re-tokenize inside the computation graph so gradients reach EEGTokenizer
        eeg_tokens = self.tokenize(eeg_seg)                    # (1, T_rl, token_dim) with grad
        T_rl = eeg_tokens.shape[1]
        B = flat["observation"].shape[0]
        eeg_tokens_batch = eeg_tokens.expand(B, -1, -1)       # (B, T_rl, token_dim)

        # Use stored timestep info to sample proper token windows
        ts = flat["eeg_timestep"].long().clamp(0, T_rl - 1)   # (B,)
        # For efficiency, use the last timestep context for the entire batch
        t_max = int(ts.max().item())
        token_seq = eeg_tokens_batch[:, : t_max + 1, :]        # (B, t_max+1, token_dim)

        brain_out = self.brain(token_seq, t_max)
        delta = brain_out["delta_action"]
        alpha = brain_out.get("alpha", torch.ones(B, 1, device=self.device))

        obs = flat["observation"].detach()
        pi, base_mean, _, _ = self.reppo._actor_forward(obs)
        base_action = pi.rsample()
        final_action = torch.tanh(base_action + alpha * delta).clamp(-1 + self.EPS, 1 - self.EPS)

        # RL loss: maximize Q(s, final_action)
        q_val, _, _, _ = self.reppo._critic_forward(flat["critic_observation"].detach(), final_action)
        brain_loss = -q_val.mean()

        self.brain_optimizer.zero_grad()
        brain_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.tokenizer.parameters()) + list(self.brain.parameters()), max_norm=0.5
        )
        self.brain_optimizer.step()

        return {
            **critic_metrics,
            **actor_metrics,
            "brain_loss": brain_loss.detach().item(),
            "delta_norm": delta.detach().norm(dim=-1).mean().item(),
            "alpha_mean": alpha.detach().mean().item(),
        }

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str, step: int | None = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "actor": self.reppo.actor.state_dict(),
                "critic": self.reppo.critic.state_dict(),
                "tokenizer": self.tokenizer.state_dict(),
                "brain": self.brain.state_dict(),
                "actor_optimizer": self.reppo.actor_optimizer.state_dict(),
                "critic_optimizer": self.reppo.critic_optimizer.state_dict(),
                "brain_optimizer": self.brain_optimizer.state_dict(),
                "step": step,
            },
            path,
        )

    def load(self, path: str) -> int | None:
        ckpt = torch.load(path, map_location=self.device)
        self.reppo.actor.load_state_dict(ckpt["actor"])
        self.reppo.critic.load_state_dict(ckpt["critic"])
        self.tokenizer.load_state_dict(ckpt["tokenizer"])
        self.brain.load_state_dict(ckpt["brain"])
        if "actor_optimizer" in ckpt:
            self.reppo.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        if "critic_optimizer" in ckpt:
            self.reppo.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        if "brain_optimizer" in ckpt:
            self.brain_optimizer.load_state_dict(ckpt["brain_optimizer"])
        with torch.no_grad():
            for p, q in zip(self.reppo.actor.parameters(), self.reppo.old_actor.parameters()):
                q.data.copy_(p.data)
        return ckpt.get("step")
