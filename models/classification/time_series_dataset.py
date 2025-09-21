"""
Time Series Dataset Module
==========================

This module provides the TimeSeriesDataset class for handling time series sensor data
with sliding window support for action classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict


class TimeSeriesDataset(Dataset):
    """Dataset for time series sensor data with windowing support"""
    
    def __init__(self, data: np.ndarray, labels: Dict[str, np.ndarray], 
                 window_size: int = 30, overlap: float= 0.5, task: str= 'both'):
        """
        Args:
            data: Sensor data with shape (n_samples, n_channels)
            labels: Action labels
            window_size: Size of sliding window
            overlap: Overlap between consecutive windows (0-1)
            transform: Optional transform to apply
        """
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.step_size = max(1, int(window_size * (1 - overlap)))
        self.overlap = overlap
        self.task = task
        
        self.window_indices = []
        for start in range(0, len(data) - window_size + 1, self.step_size):
            self.window_indices.append((start, start + window_size))

        
    def __len__(self):
        return len(self.window_indices)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.window_indices[idx]
        
        x_window = self.data[start_idx: end_idx]
        x_windowT = x_window.T

        out = {}
        for key in ['behavior', 'gesture']:
            y_window = self.labels[key][start_idx: end_idx]
            values, counts = np.unique(y_window, return_counts=True)
            out[key] = int(values[np.argmax(counts)])
        
        if self.task == 'behavior':
            y = out['behavior']
        elif self.task == 'gesture':
            y = out['gesture']
        else:
            y = (out['behavior'], out['gesture'])

        return torch.from_numpy(x_windowT).float(), y
    
    def get_data_info(self):
        """Get information about the dataset"""
        return {
            'n_windows': self.n_windows,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.step_size,
            'n_channels': self.data.shape[1],
            'total_samples': len(self.data)
        }

