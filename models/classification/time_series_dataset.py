"""
Time Series Dataset Module
==========================

This module provides the TimeSeriesDataset class for handling time series sensor data
with sliding window support for action classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional


class TimeSeriesDataset(Dataset):
    """Dataset for time series sensor data with windowing support"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int = 100, 
                 overlap: float = 0.5, transform=None):
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
        self.overlap = overlap
        self.transform = transform
        
        # Calculate step size and number of windows
        self.step_size = int(window_size * (1 - overlap))
        self.n_windows = max(1, (len(data) - window_size) // self.step_size + 1)
        
        # Create window indices
        self.window_indices = []
        for i in range(self.n_windows):
            start_idx = i * self.step_size
            end_idx = start_idx + window_size
            if end_idx <= len(data):
                self.window_indices.append((start_idx, end_idx))
        
        self.n_windows = len(self.window_indices)
        
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.window_indices[idx]
        
        window_data = self.data[start_idx:end_idx]
        window_labels = self.labels[start_idx:end_idx]
        
        # Use majority voting for window label
        unique_labels, counts = np.unique(window_labels, return_counts=True)
        window_label = unique_labels[np.argmax(counts)]
        
        # Transpose to (channels, time) format for CNN
        window_data = window_data.T
        
        if self.transform:
            window_data = self.transform(window_data)
        
        return torch.FloatTensor(window_data), torch.LongTensor([window_label])
    
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

