from __future__ import annotations

import os
import sys
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tensordict import TensorDict
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.tokenization.eeg_tokenizer import *
from models.rl.agents.reppo import RePPOAgent, EmpiricalNormalizer
from models.rl.utils.any_utils import (compute_gve, _env_shape,
                                       _split_step_return, _to_tensor,
                                       wrap_batch_dim)
class EEGConditionedREPPO:
    """
    Wraps RePPOAgent to inject an EEG-conditioned action delta at every step.

    Args
    ----
    reppo       : RePPOAgent instance
    tokenizer   : EEGRLTokenizer  (CNN)
    action_head : EEGActionHead
    token_pool  : EEGTokenPool
    task_label  : current task int label (updated externally between episodes)
    """

    def __init__(self, reppo: RePPOAgent, tokenizer: EEGRLTokenizer,
                 action_head: EEGActionHead, token_pool: EEGTokenPool,
                 task_label: int = 0):
        self.reppo       = reppo
        self.tokenizer   = tokenizer
        self.action_head = action_head
        self.pool        = token_pool
        self.task_label  = task_label
        self.device      = reppo.device

    # convenience
    @property
    def actor(self): return self.reppo.actor
    @property
    def critic(self): return self.reppo.critic
    @property
    def old_actor(self): return self.reppo.old_actor

    def _eeg_delta(self, norm_obs: torch.Tensor) -> torch.Tensor:
        """Sample EEG tokens and compute action delta for batch of obs."""
        B = norm_obs.shape[0]
        eeg_tokens = self.pool.sample_batch([self.task_label] * B).to(self.device)
        return self.action_head(norm_obs, eeg_tokens)              # (B, act_dim)

    def collect(self, env, observation, critic_observation, num_steps: int = 128):
        """
        Same signature as RePPOAgent.collect, but blends in EEG action delta.
        """
        N, _, asymmetric = _env_shape(env)
        trajectory = []
        info_list  = []
        EPS = 1e-6

        if observation is None:
            ret = env.reset()
            observation = ret[0] if isinstance(ret, tuple) else ret
        if critic_observation is None:
            critic_observation = observation

        observation = _to_tensor(observation, self.device)
        critic_observation = _to_tensor(critic_observation, self.device)
        observation = ensure_batch(observation)
        critic_observation = ensure_batch(critic_observation)

        for step_idx in range(num_steps):
            if step_idx % 10 == 0:
                print(f"[collect] step {step_idx}/{num_steps}", flush=True)
            norm_obs  = self.reppo.observation_normalizer(observation)
            norm_cobs = self.reppo.critic_observation_normalizer(critic_observation)

            with torch.no_grad():
                pi, _, _, _ = self.reppo._actor_forward(norm_obs)
                base_action = pi.sample().clamp(-1 + EPS, 1 - EPS)
                delta = self._eeg_delta(norm_obs)
                blended_action = torch.tanh(base_action + delta).clamp(-1 + EPS, 1 - EPS)
                log_prob = pi.log_prob(blended_action).sum(-1)

                action_np = blended_action.detach().cpu().numpy().astype(np.float32)

            step_return = env.step(action_np)
            if getattr(env, "render_mode", None) == "human":
                try:
                    env.render()
                except Exception as e:
                    print(f"[render warning] {e}", flush=True)
            next_obs, rewards, dones, truncated, infos = _split_step_return(step_return)
            next_cobs = next_obs

            _next_obs  = _to_tensor(next_obs, self.device)
            _next_cobs = _to_tensor(next_cobs, self.device)
            _next_obs = ensure_batch(_next_obs)
            _next_cobs = ensure_batch(_next_cobs)
            next_norm_obs  = self.reppo.observation_normalizer(_next_obs)
            next_norm_cobs = self.reppo.critic_observation_normalizer(_next_cobs)

            with torch.no_grad():
                next_pi, _, next_temp, _ = self.reppo._actor_forward(next_norm_obs)
                next_action = next_pi.sample().clamp(-1 + EPS, 1 - EPS)
                next_log_prob = next_pi.log_prob(next_action).sum(-1)
                next_value, _, _, next_features = self.reppo._critic_forward(
                    next_norm_cobs, next_action)
                rewards_t = _to_tensor(rewards, self.device).view(-1)
                shaped_r  = rewards_t - next_log_prob * next_temp * self.reppo.gamma
                blended_t = _to_tensor(blended_action.detach().cpu().numpy(), self.device)

            (obs_b, cobs_b, act_b, logp_b, rew_b,
             raw_rew_b, nfeat_b, nval_b, done_b, trunc_b) = wrap_batch_dim(
                norm_obs, norm_cobs, blended_t, log_prob,
                shaped_r, rewards_t, next_features, next_value,
                dones, truncated, self.device
            )

            td = TensorDict({
                "observation":        obs_b,
                "critic_observation": cobs_b,
                "actions":            act_b,
                "log_prob":           logp_b,
                "rewards":            rew_b,
                "raw_rewards":        raw_rew_b,
                "next_embedding":     nfeat_b,
                "next_values":        nval_b,
                "dones":              done_b,
                "truncations":        trunc_b,
            }, batch_size=(N,))

            trajectory.append(td)
            info_list.append(infos)

            observation = _to_tensor(next_obs, self.device)
            critic_observation = _to_tensor(next_cobs, self.device)

        return torch.stack(trajectory, dim=0), norm_obs, norm_cobs, info_list