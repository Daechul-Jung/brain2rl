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
                 dropout_rate: float = 0.3, task: str = 'behavior'):
        super().__init__()
        self.task = task

        self.trunk = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(3,2,1), nn.Dropout(dropout_rate),
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(3,2,1), nn.Dropout(dropout_rate),
            nn.Conv1d(64,128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(3,2,1), nn.Dropout(dropout_rate),
        )

        # compute pooled length roughly; we’ll adaptively pool anyway
        self.time_after_pool = max(1, n_times // 8)

        self.proj = nn.Linear(128, 128)  # token projection per time step
        # heads are fed a time-aggregated token (mean over time)
        self.behavior_head = nn.Linear(128, n_behavior_classes)
        self.gesture_head  = nn.Linear(128, n_gesture_classes)

    def forward(self, x):
        # x: (B, C, T)
        h = self.trunk(x)                     # (B, 128, T')
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

        return out

    @torch.no_grad()
    def tokens(self, x):
        """Return per-window 'tokens' (B, T', 128) for downstream usage."""
        h = self.trunk(x)
        h = torch.nn.functional.adaptive_avg_pool1d(h, self.time_after_pool)
        return self.proj(h.transpose(1,2))



class ActionClassifierFirst(nn.Module):
    """
    CNN model for action classification from time series sensor data
    """
    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout_rate: float = 0.3):
        super(ActionClassifierFirst, self).__init__()
        
        # Temporal convolution layers
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate)
        )
        
        # Feature extraction layers
        self.feature_conv = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate)
        )
        
        self.feature_conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(dropout_rate)
        )
        
        # Calculate output size after convolutions and pooling
        # After 3 pooling layers with stride 2: n_times // 8
        # But we need to handle cases where n_times is not divisible by 8
        self.time_after_pool = max(1, n_times // 8)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * self.time_after_pool, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, channels, time)
        x = self.temporal_conv(x)
        x = self.feature_conv(x)
        x = self.feature_conv2(x)
        
        # Ensure the output size matches our expected dimensions
        if x.size(-1) != self.time_after_pool:
            # Adaptive pooling to ensure correct size
            x = torch.nn.functional.adaptive_avg_pool1d(x, self.time_after_pool)
        
        x = self.fc(x)
        return x

    
    def get_feature_maps(self, x):
        """Get intermediate feature maps for analysis"""
        features = {}
        
        # Temporal convolution
        x = self.temporal_conv(x)
        features['temporal'] = x
        
        # Feature extraction
        x = self.feature_conv(x)
        features['feature1'] = x
        
        x = self.feature_conv2(x)
        features['feature2'] = x
        
        return features
