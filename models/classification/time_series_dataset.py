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
    
    def __init__(self, data: np.ndarray, labels: Dict[str, np.ndarray], groups: np.ndarray,
                 window_size: int = 50, overlap: float= 0.5, task: str= 'both'):
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
        self.groups = groups.astype(str)
        self.step = max(1, int(window_size * (1 - overlap)))
        self.overlap = overlap
        self.task = task
        
        # self.window_indices = []
        # for start in range(0, len(data) - window_size + 1, self.step_size):
        #     self.window_indices.append((start, start + window_size))

        self.idxs = []

        unique_groups = pd_factorize_stable(self.groups)
        for gval in unique_groups:
            idx = np.where(self.groups == gval)[0]
            if len(idx)< self.window_size:
                continue
            start_position = range(idx[0], idx[-1] - self.window_size + 2, self.step)
            local_len = len(idx)
            for s_local in range(0, local_len - self.window_size + 1, self.step):
                start = idx[s_local]
                end = start + self.window_size
                self.idxs.append((start, end))
        
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, idx):
        start_idx, end_idx = self.idxs[idx]
        
        xw = self.data[start_idx:end_idx].astype(np.float32).T  # (C, T)

        # majority vote label(s) inside the window
        if self.task == 'behavior':
            yw = self.majority(self.labels['behavior'][start_idx:end_idx])
            y_out = torch.LongTensor([yw]).squeeze(0)
        elif self.task == 'gesture':
            yw = self.majority(self.labels['gesture'][start_idx:end_idx])
            y_out = torch.LongTensor([yw]).squeeze(0)
        else:  # both
            yb = self.majority(self.labels['behavior'][start_idx:end_idx])
            yg = self.majority(self.labels['gesture'][start_idx:end_idx])
            y_out = (torch.LongTensor([yb]).squeeze(0),
                     torch.LongTensor([yg]).squeeze(0))

        return torch.from_numpy(xw), y_out
    
    def get_data_info(self):
        """Get information about the dataset"""
        return {
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.step,
            'n_channels': self.data.shape[1],
            'total_samples': len(self.data)
        }

    @staticmethod
    def majority(arr_1d: np.ndarray):
        vals, counts = np.unique(arr_1d, return_counts=True)
        return int(vals[np.argmax(counts)])
    
def pd_factorize_stable(a: np.ndarray):
    seen, order = set(), []

    for v in a:
        if v not in seen:
            seen.add(v); order.append(v)
    return order