"""
Action Classifier CNN Module
============================

This module provides the ActionClassifier CNN model for classifying time series sensor data
into different action categories.
"""

import torch
import torch.nn as nn
from typing import Tuple


# action_classifier.py
class ActionClassifier(nn.Module):
    def __init__(self, n_channels: int, n_times: int,
                 n_behavior_classes: int, n_gesture_classes: int,
                 dropout_rate: float = 0.3, task: str = 'behavior', device = 'cuda'):
        super().__init__()
        self.task = task
        self.device = device
        self.trunk = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3,2,1), nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3,2,1), nn.Dropout(dropout_rate),
            nn.Conv1d(64,128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(3,2,1), nn.Dropout(dropout_rate),
        ).to(self.device)

        # compute pooled length roughly; we’ll adaptively pool anyway
        self.time_after_pool = max(1, n_times // 8)

        self.proj = nn.Linear(128, 128).to(self.device)  # token projection per time step
        # heads are fed a time-aggregated token (mean over time)
        self.behavior_head = nn.Linear(128, n_behavior_classes).to(self.device)
        self.gesture_head  = nn.Linear(128, n_gesture_classes).to(self.device)

    def forward(self, x, return_tokens: bool = False, pool_tokens_to: int | None = None):
        # x: (B, C, T)
        h = self.trunk(x)                     # (B, 128, T')
        if pool_tokens_to is not None:
            h = torch.nn.functional.adaptive_avg_pool1d(h, pool_tokens_to)
        if h.size(-1) != self.time_after_pool:
            h = torch.nn.functional.adaptive_avg_pool1d(h, self.time_after_pool)

        # tokens = projected per-timestep features
        tokens = self.proj(h.transpose(1,2))  # (B, T', 128)

        # global representation for classification
        glob = tokens.mean(dim=1)             # (B, 128)

        out = {}
        if self.task in ('behavior','both'):
            out['behavior_logits'] = self.behavior_head(glob)
        if self.task in ('gesture','both'):
            out['gesture_logits']  = self.gesture_head(glob)

        if return_tokens:
            out['tokens'] = tokens

        return out

    @torch.no_grad()
    def tokens(self, x):
        """Return per-window 'tokens' (B, T', 128) for downstream usage."""
        h = self.trunk(x)
        h = torch.nn.functional.adaptive_avg_pool1d(h, self.time_after_pool)
        return self.proj(h.transpose(1,2))

    @torch.no_grad()
    def tokens_from_raw(self, x, pool_tokens_to: int | None = None ):
        """
        return per-sequence tokens with optional pooling to fixed K
        """
        h = self.trunk(x) ## (B, 128, T')
        if pool_tokens_to is not None:
            h = torch.nn.functional.adaptive_avg_pool1d(h, pool_tokens_to)
        return self.proj(h.transpose(1, 2))  # (B, K or T', 128)


