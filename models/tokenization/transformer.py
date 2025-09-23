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

# Add project root to path
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
        K = self.W_v(value).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

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

class TokensEpisodeDataset(Dataset):
    """
    token_wins: (n_windows, Tprime, 128) float
    actions: (n_windows, ) int, 
    Build episodes by concatenating K consecutive windows along time
    """

    def __init__(self, tokens_win: np.ndarray, 
                 actions: np.ndarray,
                 windows_per_episode: int= 8,
                 stride_window: int= 8):
        assert tokens_win.ndim == 3 and tokens_win.shape[2] == 128, "Expect (N, T', 128)"
        assert len(tokens_win) == len(actions)

        self.X = tokens_win
        self.y = actions.astype(np.int64)
        self.wp = int(windows_per_episode)
        self.sw = int(stride_window)
        self.Tp = tokens_win.shape[1]
        self.n = len(tokens_win)
        self.n_episodes = max(0, (self.n -self.wp) // self.sw +1)

    def __len__(self) -> int:
        return self.n_episodes
    

    def __getitem__(self, i):
        start = i * self.sw
        end  = start + self.wp
        seq = self.X[start:end]
        seq = seq.reshape(self.wp * self.Tp, 128) ### (time total, 128)

        a = self.y[end - 1]

        x = torch.from_numpy(seq).float().transpose(0, 1) ### (128, time total)
        y = torch.tensor(a, dtype=torch.long)

        return x, y 
