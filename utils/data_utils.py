import os
import numpy as np
import pandas as pd
import mne
import nibabel as nib
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class BrainSignalDataset(Dataset):
    """Custom dataset for brain signal data"""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample, label

def load_eeg_data(data_path, preprocess=True):
    """
    Load EEG data from the Grasp and Lift dataset
    Args:
        data_path: Path to the EEG data
        preprocess: Whether to preprocess the data
    Returns:
        X: EEG data (n_samples, n_channels, n_times)
        y: Labels (n_samples,)
    """
    print(f"Loading EEG data from {data_path}")
    
    # This is a placeholder - actual implementation will depend on dataset format
    # For Grasp and Lift dataset, you would typically load a .mat file
    
    # Simulated data for now
    n_samples = 1000
    n_channels = 32
    n_times = 512  # ~2 seconds at 256 Hz
    X = np.random.randn(n_samples, n_channels, n_times)
    y = np.random.randint(0, 6, size=n_samples)  # 6 action classes
    
    if preprocess:
        X = preprocess_eeg(X)
    
    return X, y

def preprocess_eeg(data):
    """
    Preprocess EEG data:
    - Apply bandpass filtering
    - Remove artifacts
    - Normalize
    """
    # Apply a band-pass filter (typically 1-40 Hz for EEG)
    # This would use MNE in a real implementation
    
    # Normalize per channel
    for i in range(data.shape[0]):
        scaler = StandardScaler()
        data[i] = scaler.fit_transform(data[i].T).T
    
    return data

def load_fmri_data(data_path, preprocess=True):
    """
    Load fMRI data from HCP dataset
    Args:
        data_path: Path to the fMRI data
        preprocess: Whether to preprocess the data
    Returns:
        X: fMRI data
        y: Labels
    """
    print(f"Loading fMRI data from {data_path}")
    
    # This is a placeholder - actual implementation will depend on dataset format
    # For HCP dataset, you would typically load NIfTI files
    
    # Simulated data for now
    n_samples = 500
    n_voxels = 10000  # Simplified representation
    n_times = 100     # Time points
    X = np.random.randn(n_samples, n_voxels, n_times)
    y = np.random.randint(0, 4, size=n_samples)  # 4 action classes
    
    if preprocess:
        X = preprocess_fmri(X)
    
    return X, y

def preprocess_fmri(data):
    """
    Preprocess fMRI data:
    - Motion correction
    - Spatial normalization
    - Temporal filtering
    - Intensity normalization
    """
    # These preprocessing steps would use NiBabel in a real implementation
    
    # Basic normalization (simplified)
    for i in range(data.shape[0]):
        data[i] = (data[i] - np.mean(data[i])) / np.std(data[i])
    
    return data

def create_dataloader(X, y, batch_size=32, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True):
    """
    Create PyTorch DataLoaders for train, validation, and test sets
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Convert to torch tensors
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    
    # Split indices
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create datasets
    train_dataset = BrainSignalDataset(X[train_indices], y[train_indices])
    val_dataset = BrainSignalDataset(X[val_indices], y[val_indices])
    test_dataset = BrainSignalDataset(X[test_indices], y[test_indices])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader 