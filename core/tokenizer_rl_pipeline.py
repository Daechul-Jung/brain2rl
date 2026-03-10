"""
RL Tokenizer Pipeline
=====================

Trains a CNN EEG tokenizer that produces an *action delta* conditioned on
the robot's current observation. The delta is added to the REPPO actor's
base action before sending it to the environment. The entire system
(actor, critic, CNN tokenizer, action head) is updated from the same RL loss.

Architecture
------------
  EEG Segment (C, T)
    → CNN Trunk (EEGRLTokenizer)        → (B, 128, K) = eeg_tokens

  Robot observation (obs_dim,)
    → EEGActionHead (cross-attention)
        query = Linear(obs)              → (B, 1, hidden_dim)
        keys/values = eeg_tokens         → (B, K, hidden_dim)
        cross-attn → Linear → Tanh       → (B, action_dim)  = action_delta

  REPPO Actor (obs) → base_action (action_dim)

  final_action = tanh(base_action + scale * action_delta)   applied to env

Training
--------
  REPPO collects trajectories with the blended action.
  REPPO loss is backpropagated through actor, critic, tokenizer, and action head.

EEG Data During RL
------------------
  A pool of pre-extracted EEG tokens is built per action label (push/pull/pick/stack).
  At each rollout step the pool entry matching the *current task* is sampled and
  fed to EEGActionHead.  This lets the tokenizer learn task-conditioned action
  modulations without requiring real-time EEG input.
"""

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

from models.classification.action_classifier import ActionClassifier
from models.rl.agents.reppo import RePPOAgent, EmpiricalNormalizer
from models.rl.utils.any_utils import (compute_gve, _env_shape,
                                       _split_step_return, _to_tensor,
                                       wrap_batch_dim)
from models.rl.utils.train import _episode_stats_from_rollout

def ensure_batch(obs: torch.Tensor) -> torch.Tensor:
    if obs.ndim == 1:
        return obs.unsqueeze(0)  
    return obs
# ---------------------------------------------------------------------------
# CNN EEG Tokenizer for RL
# ---------------------------------------------------------------------------

class EEGRLTokenizer(nn.Module):
    """
    Encodes a variable-length EEG segment into K fixed tokens (128-dim).
    Shares the ActionClassifier CNN trunk design but is trained end-to-end
    with the RL objective.

    Args
    ----
    n_channels : EEG channels
    n_times    : representative segment length (for trunk init)
    pool_k     : number of output tokens
    """

    def __init__(self, n_channels: int, n_times: int, pool_k: int = 16):
        super().__init__()
        self.pool_k = pool_k
        _dummy = ActionClassifier(
            n_channels=n_channels, n_times=n_times,
            n_behavior_classes=2, n_gesture_classes=2, task='behavior'
        )
        self.trunk = _dummy.trunk
        self.proj  = _dummy.proj   # Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, T)  - EEG segment
        Returns: (B, pool_k, 128)
        """
        h = self.trunk(x)                                       # (B, 128, T')
        h = F.adaptive_avg_pool1d(h, self.pool_k)               # (B, 128, K)
        return self.proj(h.transpose(1, 2))                     # (B, K, 128)

    def load_pretrained_trunk(self, classifier_ckpt_path: str, strict: bool = False):
        """Optionally warm-start the CNN trunk from a trained classifier."""
        ckpt = torch.load(classifier_ckpt_path, map_location='cpu')
        # Filter keys that belong to trunk / proj
        trunk_sd = {k.replace('trunk.', ''): v for k, v in ckpt.items()
                    if k.startswith('trunk.')}
        proj_sd  = {k.replace('proj.', ''): v for k, v in ckpt.items()
                    if k.startswith('proj.')}
        if trunk_sd:
            self.trunk.load_state_dict(trunk_sd, strict=strict)
        if proj_sd:
            self.proj.load_state_dict(proj_sd, strict=strict)


# ---------------------------------------------------------------------------
# Learnable EEG Action Head
# ---------------------------------------------------------------------------

class EEGActionHead(nn.Module):
    """
    Cross-attention from robot observation to EEG tokens → action delta.

    The observation acts as a *query* that selects which parts of the EEG
    token context are relevant for the current robot state.

    Args
    ----
    token_dim  : EEG token dim (128)
    obs_dim    : robot observation dimension
    action_dim : robot action dimension
    hidden_dim : internal attention/projection dimension
    n_heads    : number of cross-attention heads
    scale      : tanh output scale for action delta (default 0.3)
    """

    def __init__(self, token_dim: int, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, n_heads: int = 4, scale: float = 0.3):
        super().__init__()
        self.scale = scale

        # Project observation → query
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Project EEG tokens → key/value space
        self.token_proj = nn.Linear(token_dim, hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            batch_first=True, dropout=0.1
        )
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, eeg_tokens: torch.Tensor) -> torch.Tensor:
        """
        obs        : (B, obs_dim)
        eeg_tokens : (B, K, 128)
        Returns    : (B, action_dim) action delta in [-scale, scale]
        """
        q = self.obs_proj(obs).unsqueeze(1)       # (B, 1, H)
        kv = self.token_proj(eeg_tokens)           # (B, K, H)
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 1, H)
        delta = torch.tanh(self.out(attn_out.squeeze(1))) * self.scale
        return delta                               # (B, action_dim)


# ---------------------------------------------------------------------------
# EEG Token Pool (pre-computed per task label)
# ---------------------------------------------------------------------------

class EEGTokenPool:
    """
    Stores a pool of pre-extracted EEG tokens indexed by action label.
    During rollout, call sample(label) to get a token tensor for that label.

    Args
    ----
    tokens      : (N, K, 128) – from EEGRLTokenizer.forward(X)
    labels      : (N,) int – action label per segment
    device      : torch device
    """

    def __init__(self, tokens: np.ndarray, labels: np.ndarray, device: torch.device):
        self.device = device
        unique = np.unique(labels)
        self._pool: Dict[int, torch.Tensor] = {}
        for lbl in unique:
            mask = labels == lbl
            t = torch.from_numpy(tokens[mask].astype(np.float32)).to(device)
            self._pool[int(lbl)] = t   # (N_lbl, K, 128)

    def sample(self, label: int, n: int = 1) -> torch.Tensor:
        """Return (n, K, 128) tensor for the given label."""
        pool = self._pool.get(label)
        if pool is None:
            # Fall back to random label
            pool = next(iter(self._pool.values()))
        idx = torch.randint(0, len(pool), (n,))
        return pool[idx]                           # (n, K, 128)

    def sample_batch(self, labels: List[int]) -> torch.Tensor:
        """Return (B, K, 128) where each row matches labels[i]."""
        return torch.stack([self.sample(int(l), 1).squeeze(0) for l in labels])


# ---------------------------------------------------------------------------
# REPPO + EEG Action Head: augmented collect
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RL Tokenizer Training Pipeline
# ---------------------------------------------------------------------------

class RLTokenizerPipeline:
    """
    End-to-end training of REPPO + EEG tokenizer + action head.

    Steps
    1. Pre-extract EEG token pool from EEG data.
    2. Build REPPO agent + EEGActionHead.
    3. Run rollouts with EEGConditionedREPPO.
    4. Backprop REPPO loss through actor, critic, tokenizer, and action head jointly.

    Args
    ----
    config : dict with keys:
        obs_dim, action_dim, n_channels, n_times
        pool_k (int, 16)
        hidden_dim (int, 256), n_heads (int, 4), eeg_scale (float, 0.3)
        lr (float, 3e-4)
        num_atoms, vmin, vmax, gamma, lmbda, kl_start, entropy_start
        total_steps, num_step, num_epoch, num_mini_batch
        classifier_ckpt (str, optional) - warm-start CNN from classifier
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg    = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = self._make_logger()

        self.reppo       : Optional[RePPOAgent]          = None
        self.tokenizer   : Optional[EEGRLTokenizer]      = None
        self.action_head : Optional[EEGActionHead]       = None
        self.agent_wrap  : Optional[EEGConditionedREPPO] = None
        self.token_pool  : Optional[EEGTokenPool]        = None
        self.eeg_optimizer = None

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger('RLTokenizer')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(h)
        return logger

    def build_token_pool(self, X_eeg: np.ndarray, y_labels: np.ndarray):
        """
        Pre-extract EEG tokens and store in a per-label pool.

        Args
        ----
        X_eeg    : (N, C, T)
        y_labels : (N,)  integer action labels
        """
        cfg = self.cfg
        N, C, T = X_eeg.shape

        if self.tokenizer is None:
            self.tokenizer = EEGRLTokenizer(
                n_channels=C, n_times=T,
                pool_k=cfg.get('pool_k', 16)
            ).to(self.device)

            if cfg.get('classifier_ckpt'):
                self.tokenizer.load_pretrained_trunk(cfg['classifier_ckpt'])
                self.logger.info("Warm-started CNN trunk from classifier checkpoint.")

        self.tokenizer.eval()
        all_tokens = []
        bs = cfg.get('batch_size', 32)
        X_t = torch.from_numpy(X_eeg.astype(np.float32))
        with torch.no_grad():
            for i in range(0, N, bs):
                xb = X_t[i:i+bs].to(self.device)
                all_tokens.append(self.tokenizer(xb).cpu().numpy())
        tokens = np.concatenate(all_tokens, axis=0)   # (N, K, 128)
        self.token_pool = EEGTokenPool(tokens, y_labels, self.device)
        self.logger.info(f"Built EEG token pool: {tokens.shape}, "
                         f"labels={np.unique(y_labels).tolist()}")
        return tokens

    # ------------------------------------------------------------------
    def build_agent(self):
        cfg = self.cfg
        obs_dim    = cfg['obs_dim']
        action_dim = cfg['action_dim']
        pool_k     = cfg.get('pool_k', 16)

        obs_norm  = EmpiricalNormalizer(shape=obs_dim, device=self.device)
        cobs_norm = EmpiricalNormalizer(shape=obs_dim, device=self.device)

        self.reppo = RePPOAgent(
            observation_dim=obs_dim, action_dim=action_dim,
            num_atoms=cfg.get('num_atoms', 151),
            vmin=cfg.get('vmin', -2500), vmax=cfg.get('vmax', 2500),
            device=str(self.device),
            lr=cfg.get('lr', 3e-4),
            gamma=cfg.get('gamma', 0.99),
            kl_start=cfg.get('kl_start', 0.01),
            entropy_start=cfg.get('entropy_start', 0.01),
            lmbda=cfg.get('lmbda', 0.95),
            obs_normalizer=obs_norm,
            critic_obs_normalizer=cobs_norm,
        )

        self.action_head = EEGActionHead(
            token_dim=128, obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=cfg.get('hidden_dim', 256),
            n_heads=cfg.get('n_heads', 4),
            scale=cfg.get('eeg_scale', 0.3)
        ).to(self.device)

        # Joint optimizer for tokenizer + action_head
        eeg_params = (list(self.tokenizer.parameters()) +
                      list(self.action_head.parameters()))
        self.eeg_optimizer = torch.optim.AdamW(
            eeg_params, lr=cfg.get('lr', 3e-4), weight_decay=1e-4)

    # ------------------------------------------------------------------
    def build_wrapped_agent(self, task_label: int = 0):
        assert self.token_pool is not None, "Call build_token_pool() first."
        assert self.reppo is not None,      "Call build_agent() first."
        self.agent_wrap = EEGConditionedREPPO(
            reppo=self.reppo,
            tokenizer=self.tokenizer,
            action_head=self.action_head,
            token_pool=self.token_pool,
            task_label=task_label,
        )

    # ------------------------------------------------------------------
    def train(self, env, task_label: int = 0) -> List[float]:
        """
        Full training loop.

        Args
        ----
        env        : gymnasium-compatible environment
        task_label : integer action label matching EEG pool entries

        Returns
        -------
        all_episode_returns : list of float
        """
        cfg = self.cfg
        self.agent_wrap.task_label = task_label

        total_steps    = cfg.get('total_steps', 100_000)
        num_step       = cfg.get('num_step', 128)
        num_epoch      = cfg.get('num_epoch', 8)
        num_mini_batch = cfg.get('num_mini_batch', 4)
        N_envs = getattr(env, 'num_envs', 1)
        batch_size     = (N_envs * num_step) // num_mini_batch
        total_updates  = total_steps // (N_envs * num_step) + 1

        ret = env.reset()
        obs  = ret[0] if isinstance(ret, tuple) else ret
        cobs = None
        all_returns = []

        for update in range(total_updates):
            # --- Collect rollout ---
            self.tokenizer.train()
            self.action_head.train()
            transition, obs, cobs, infos = self.agent_wrap.collect(
                env, obs, cobs, num_step)

            ep_rets, _ = _episode_stats_from_rollout(transition)
            all_returns.extend(ep_rets)
            if ep_rets:
                self.logger.info(
                    f"Update {update}/{total_updates}  "
                    f"ep_return={np.mean(ep_rets):.2f}  "
                    f"n_ep={len(ep_rets)}")

            # --- Compute GVE ---
            gves = compute_gve(
                rewards=transition['rewards'],
                dones=transition['dones'],
                truncations=transition['truncations'],
                next_values=transition['next_values'],
                gamma=self.reppo.gamma,
                lmbda=self.reppo.lmbda,
            )

            data = TensorDict({
                'observation':        transition['observation'],
                'critic_observation': transition['critic_observation'],
                'actions':            transition['actions'],
                'rewards':            transition['rewards'],
                'raw_rewards':        transition['raw_rewards'],
                'next_embedding':     transition['next_embedding'],
                'next_values':        transition['next_values'],
                'dones':              transition['dones'],
                'truncations':        transition['truncations'],
                'gve':                torch.stack(gves, dim=0),
            }, batch_size=(num_step, N_envs), device=self.device).float().flatten(0, 1).detach()

            # --- SGD update ---
            for _ in range(num_epoch):
                idx = torch.randperm(num_step * N_envs, device=self.device)
                data_shuf = data[idx].contiguous()
                for mb in range(num_mini_batch):
                    batch = data_shuf[mb * batch_size: (mb + 1) * batch_size]

                    # REPPO critic update
                    critic_logs = self.reppo.update_critic(batch)

                    # REPPO actor update (also updates EEG head via shared obs)
                    actor_logs = self.reppo.update_actor(batch)

                    # EEG tokenizer + action head update
                    # Re-compute action delta and use actor loss gradient
                    obs_b = batch['observation']
                    eeg_tokens = self.token_pool.sample_batch(
                        [task_label] * len(obs_b)).to(self.device)
                    self.tokenizer.train(); self.action_head.train()
                    delta = self.action_head(obs_b, eeg_tokens)

                    # Penalize large deltas (regularization) and align with
                    # the actor's intended action direction
                    delta_reg_loss = delta.pow(2).mean() * 0.01
                    self.eeg_optimizer.zero_grad()
                    delta_reg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.tokenizer.parameters()) +
                        list(self.action_head.parameters()), 0.5)
                    self.eeg_optimizer.step()

            # Sync old actor
            with torch.no_grad():
                for p, q in zip(self.reppo.actor.parameters(),
                                self.reppo.old_actor.parameters()):
                    q.data.copy_(p.data)

        return all_returns

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        ckpt = {
            'cfg':         self.cfg,
            'tokenizer':   self.tokenizer.state_dict(),
            'action_head': self.action_head.state_dict(),
            'actor':       self.reppo.actor.state_dict(),
            'critic':      self.reppo.critic.state_dict(),
        }
        torch.save(ckpt, path)
        self.logger.info(f"Saved RL tokenizer + agent to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.cfg = ckpt['cfg']
        self.tokenizer.load_state_dict(ckpt['tokenizer'])
        self.action_head.load_state_dict(ckpt['action_head'])
        self.reppo.actor.load_state_dict(ckpt['actor'])
        self.reppo.critic.load_state_dict(ckpt['critic'])
        self.logger.info(f"Loaded from {path}")

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_action(self, obs: np.ndarray, task_label: int) -> np.ndarray:
        """
        Inference: blended action for a single observation.

        Args
        ----
        obs : (obs_dim,) numpy array
        task_label : int EEG task label

        Returns
        -------
        action : (action_dim,) numpy array in [-1, 1]
        """
        self.reppo.actor.eval()
        self.tokenizer.eval()
        self.action_head.eval()
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)
        norm_obs = self.reppo.observation_normalizer(obs_t)
        pi, _, _, _ = self.reppo._actor_forward(norm_obs)
        base_act = pi.sample().clamp(-1 + 1e-6, 1 - 1e-6)
        eeg_tokens = self.token_pool.sample(task_label, n=1).to(self.device)
        delta = self.action_head(norm_obs, eeg_tokens)
        final = torch.tanh(base_act + delta).clamp(-1 + 1e-6, 1 - 1e-6)
        return final.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    import argparse
    import mani_skill.envs  # noqa: register envs
    import gymnasium as gym
    from models.classification.eeg_raw_loader import load_eeg_dataset
    from models.rl.mani_skill.tasks.multiple_tasks_env import CombinedTaskEnv  # noqa

    parser = argparse.ArgumentParser(description='Train RL EEG Tokenizer on ManiSkill')
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--eeg-dir', help='Raw EEG .txt directory (recommended)')
    grp.add_argument('--eeg-csv', help='Pre-processed CSV (legacy)')
    parser.add_argument('--env',  default='Combined-v1')
    parser.add_argument('--obs-mode', default='state')
    parser.add_argument('--control-mode', default='pd_joint_delta_pos')
    parser.add_argument('--total-steps', type=int, default=200_000)
    parser.add_argument('--num-step', type=int, default=128)
    parser.add_argument('--pool-k', type=int, default=16)
    parser.add_argument('--eeg-scale', type=float, default=0.3)
    parser.add_argument('--classifier-ckpt', default=None,
                        help='Optional: warm-start CNN from classifier checkpoint')
    parser.add_argument('--output', default='output/rl_tokenizer/rl_tokenizer.pth')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    # --- Load EEG data ---
    if args.eeg_dir:
        X, y_dict, groups, meta = load_eeg_dataset(
            data_dir=args.eeg_dir, window_sec=1.0, overlap=0.5)
        y_behavior = y_dict['behavior']
    else:
        from models.classification.data_utilities import (
            load_sensor_data, preprocess_multilabel)
        X_raw, y_str, groups, df = load_sensor_data(args.eeg_csv, group_by='sequence_id')
        X, y_enc, scaler, encoders = preprocess_multilabel(X_raw, y_str)
        y_behavior = y_enc['behavior']
        if X.ndim == 2:
            C = X_raw.shape[1];  T = X.shape[1] // C
            X = X.reshape(-1, C, T)
    N, C, T = X.shape

    # --- Make environment ---
    render_mode = 'human' if args.render else None
    env = gym.make(args.env, obs_mode=args.obs_mode,
                   control_mode=args.control_mode, render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    print("obs_space:", env.observation_space)
    print("act_space:", env.action_space)
    print("obs_shape:", obs_shape)
    print("act_shape:", act_shape)

    obs_dim = obs_shape[-1]
    action_dim = act_shape[-1]

    cfg = {
        'obs_dim':           obs_dim,
        'action_dim':        action_dim,
        'n_channels':        C,
        'n_times':           T,
        'pool_k':            args.pool_k,
        'eeg_scale':         args.eeg_scale,
        'lr':                3e-4,
        'total_steps':       args.total_steps,
        'num_step':          args.num_step,
        'num_epoch':         8,
        'num_mini_batch':    4,
        'batch_size':        32,
        'classifier_ckpt':   args.classifier_ckpt,
    }

    pipe = RLTokenizerPipeline(cfg)
    pipe.build_token_pool(X, y_behavior)
    pipe.build_agent()
    pipe.build_wrapped_agent(task_label=0)   # start with label 0 (e.g., push)

    returns = pipe.train(env, task_label=0)
    pipe.save(args.output)
    env.close()

    print(f"Training done. Mean return (last 20 eps): "
          f"{np.mean(returns[-20:]) if returns else 0:.2f}")


if __name__ == '__main__':
    main()
