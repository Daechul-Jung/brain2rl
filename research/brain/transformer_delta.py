"""
TransformerDelta — BrainConditioner implemented as a causal transformer.

Idea spec: research/ideas/001_transformer_delta.md

Design (ADR-005):
  - A causal TransformerEncoder attends over EEG tokens[0..t] at step t.
  - A learned [ACT] query cross-attends to the encoded tokens, producing a
    single action-context feature.
  - Two output heads:
      delta_action : (B, action_dim) — additive perturbation to REPPO base action
      alpha        : (B, 1) in [0,1] — confidence gate; suppresses delta when EEG is noisy
  - Optionally stochastic: outputs log_std for the delta distribution (for future SAC-style use).

Integration with REPPO:
  final_mean = base_mean + alpha * delta_action
  action ~ TanhNormal(final_mean, exp(base_log_std))
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from research.brain.base import BrainConditioner


class TransformerDelta(BrainConditioner):
    """
    Causal transformer over EEG token history → action delta + gating scalar.

    Args:
        token_dim      : dimensionality of input EEG tokens (must match EEGTokenizer)
        action_dim     : robot action dimensionality
        d_model        : transformer internal width
        n_heads        : attention heads (d_model must be divisible by n_heads)
        n_layers       : number of TransformerEncoder layers
        dropout        : dropout rate
        max_seq_len    : maximum episode length (for positional encoding buffer)
        stochastic     : if True, also output log_std for the delta distribution
    """

    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        stochastic: bool = False,
    ):
        super().__init__()
        self._token_dim = token_dim
        self._action_dim = action_dim
        self.stochastic = stochastic

        self.in_proj = nn.Linear(token_dim, d_model) if token_dim != d_model else nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.act_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.delta_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )
        self.logstd_head = nn.Linear(d_model, action_dim) if stochastic else None

        # Sinusoidal positional encoding (not a parameter — won't be optimized)
        pe = self._build_pe(max_seq_len, d_model)
        self.register_buffer("pe", pe)  # (1, max_seq_len, d_model)

    @staticmethod
    def _build_pe(max_len: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask so position t cannot attend to t+1, t+2, …"""
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(self, token_seq: torch.Tensor, t: int) -> dict[str, torch.Tensor]:
        """
        Args:
            token_seq : (B, t+1, token_dim)
            t         : current step index

        Returns:
            dict with 'delta_action' (B, action_dim) and 'alpha' (B, 1).
        """
        B, T, _ = token_seq.shape

        z = self.in_proj(token_seq)                                  # (B, T, d_model)
        z = z + self.pe[:, :T, :]                                   # add positional encoding
        mask = self._causal_mask(T, token_seq.device)
        z = self.encoder(z, mask=mask)                               # (B, T, d_model)

        q = self.act_query.expand(B, -1, -1)                        # (B, 1, d_model)
        act_feat, _ = self.cross_attn(q, z, z)                      # (B, 1, d_model)
        act_feat = act_feat.squeeze(1)                               # (B, d_model)

        out: dict[str, torch.Tensor] = {
            "delta_action": self.delta_head(act_feat),               # (B, action_dim)
            "alpha": self.alpha_head(act_feat),                      # (B, 1)
        }
        if self.stochastic and self.logstd_head is not None:
            out["log_std"] = self.logstd_head(act_feat)             # (B, action_dim)

        return out
