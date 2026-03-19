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

        # Project observation -> query
        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        # Project EEG tokens -> key/value space
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


class EEGTokenPool:
    """
    Stores a pool of pre-extracted EEG tokens indexed by action label.
    During rollout, call sample(label) to get a token tensor for that label.

    Args
    ----
    tokens      : (N, K, 128): from EEGRLTokenizer.forward(X)
    labels      : (N,) int: action label per segment
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
