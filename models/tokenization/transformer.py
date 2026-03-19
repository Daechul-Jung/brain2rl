import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import math

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.tokenization.brain_tokenizer_transformer import BrainTokenizer


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Q/K/V matrix generation"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        B, T, E = query.size()  ### batch size, sequence length, E

        Q = self.W_q(query).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  ### (B, H, T, head dim)
        V = self.W_v(value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  ### (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim = -1)) ### (B, H, T, T)

        ctx = torch.matmul(attn, V) ## (B, H, T, D)
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, E)
        out = self.W_o(ctx)

        return out, {
            'query': Q, 'key': K, 'value': V,
            'attention_weights': attn, 'scores': scores
        }
    

class IntentTransformerDelta(nn.Module):
    """
    Takes EEG tokens (B, num_tokens, token_dim) and outputs a delta action for RL.

    Outputs:
      - delta_action: (B, action_dim)          # additive perturbation
      - Optional alpha: (B,1) in [0,1]         # gate/confidence for blending if you want
      - Optional log_std: (B, action_dim)      # if using SAC/PPO stochastic delta

    Design:
      - a light TransformerEncoder on tokens
      - a learned [ACT] query that cross-attends to tokens (gives robust control summary)
      - MLP head(s) to produce outputs
      """
    
    def __init__(self, 
                token_dim: int,
                action_dim: int,
                d_model: int,
                n_heads: int,
                num_layers: int,
                dropout: int, 
                use_alpha_gate: bool = True,
                stochastic_delta: bool = False):
        
        super().__init__()
        self.stochastic = stochastic_delta
        self.use_alpha_gate = use_alpha_gate

        self.in_proj = nn.Linear(token_dim, d_model) if token_dim != d_model else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = d_model, nhead=n_heads, dim_feedforward=4*d_model,
            dropout=dropout, batch_first=True 
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.act_query = nn.Parameter(torch.randn(1, 1, d_model))

        self.cross_attn = MultiHeadAttention(d_model, n_heads=n_heads, dropout=dropout)

        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),nn.ReLU(), 
            nn.Linear(d_model, action_dim)
        )

        self.logstd_head = None 
        if stochastic_delta:
            self.logstd_head = nn.Linear(d_model, action_dim)

        self.alpha_head = None
        if use_alpha_gate:
            self.alpha_head = nn.Sequential(
                nn.Linear(d_model, d_model//2), nn.ReLU(),
                nn.Linear(d_model//2, 1),
                nn.Sigmoid()
            )

    def forward(self, tokens):
        """
        tokens: (B, num_tokens, token_dim)
        return dict with key: 'delta_action', 
        """

        B, T, D = tokens.shape
        z = self.in_proj(tokens)
        z = self.encoder(z)

        q = self.act_query.expand(B, -1, -1)  ## (B, 1, d_model)
        act_ctx, _ = self.cross_attn(q, z, z)
        act_feat = act_ctx[:, 0, :]

        out = {}
        out['delta_action'] = self.delta_head(act_feat)

        if self.stochastic and self.logstd_head is not None:
            out['log_std'] = self.logstd_head(act_feat)

        if self.use_alpha_gate and self.alpha_head is not None:
            out['alpha'] = self.alpha_head(act_feat)     # (B,1) in [0,1]

        return out

class TokensEpisodeDataset(Dataset):
    """
    token_wins: (n_windows, Tprime, 128) float
    actions: (n_windows, ) int,
    Build episodes by concatenating K consecutive windows along time

    NOTE: kept as you wrote it. It yields (x, y) where x is (128, time_total).
    If you prefer (time_total, 128), just remove the transpose in __getitem__.
    """
    def __init__(self, tokens_win: np.ndarray, 
                 actions: np.ndarray,
                 windows_per_episode: int = 8,
                 stride_window: int = 8):
        assert tokens_win.ndim == 3 and tokens_win.shape[2] == 128, "Expect (N, T', 128)"
        assert len(tokens_win) == len(actions)

        self.X = tokens_win
        self.y = actions.astype(np.int64)
        self.wp = int(windows_per_episode)
        self.sw = int(stride_window)
        self.Tp = tokens_win.shape[1]
        self.n = len(tokens_win)
        self.n_episodes = max(0, (self.n - self.wp) // self.sw + 1)

    def __len__(self) -> int:
        return self.n_episodes
    
    def __getitem__(self, i):
        start = i * self.sw
        end   = start + self.wp
        seq = self.X[start:end]                        # (wp, T', 128)
        seq = seq.reshape(self.wp * self.Tp, 128)      # (time_total, 128)

        a = self.y[end - 1]
        x = torch.from_numpy(seq).float().transpose(0, 1)  # (128, time_total)
        y = torch.tensor(a, dtype=torch.long)
        return x, y