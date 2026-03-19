"""
RL Tokenizer Pipeline

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

from models.tokenization.eeg_tokenizer import *
from models.rl.agents.eeg_reppo import *
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

        self.reppo       : Optional[RePPOAgent] = None
        self.tokenizer   : Optional[EEGRLTokenizer] = None
        self.action_head : Optional[EEGActionHead] = None
        self.agent_wrap  : Optional[EEGConditionedREPPO] = None
        self.token_pool  : Optional[EEGTokenPool] = None
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

            for _ in range(num_epoch):
                idx = torch.randperm(num_step * N_envs, device=self.device)
                data_shuf = data[idx].contiguous()
                for mb in range(num_mini_batch):
                    batch = data_shuf[mb * batch_size: (mb + 1) * batch_size]

                    critic_logs = self.reppo.update_critic(batch)

                    actor_logs = self.reppo.update_actor(batch)

                    obs_b = batch['observation']
                    eeg_tokens = self.token_pool.sample_batch(
                        [task_label] * len(obs_b)).to(self.device)
                    self.tokenizer.train(); self.action_head.train()
                    delta = self.action_head(obs_b, eeg_tokens)

                    delta_reg_loss = delta.pow(2).mean() * 0.01
                    self.eeg_optimizer.zero_grad()
                    delta_reg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.tokenizer.parameters()) +
                        list(self.action_head.parameters()), 0.5)
                    self.eeg_optimizer.step()

            with torch.no_grad():
                for p, q in zip(self.reppo.actor.parameters(),
                                self.reppo.old_actor.parameters()):
                    q.data.copy_(p.data)

        return all_returns

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
    parser.add_argument('--num-step', type=int, default=12800)
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
    if args.render:
        os.environ.pop("PYOPENGL_PLATFORM", None)

    render_mode = 'human' if args.render else None
    print(f"[env] render_mode={render_mode}, PYOPENGL_PLATFORM={os.environ.get('PYOPENGL_PLATFORM')}")

    env = gym.make(
        args.env,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=render_mode
    )
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
