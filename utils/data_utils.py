"""
Data Utilities for Brain2RL Pipeline
====================================

This module provides utility functions for loading, preprocessing, and managing
sensor data (EEG, IMU, etc.) used in the Brain2RL pipeline.

Author: Brain2RL Team
Version: 1.0.0
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import pickle
import h5py
from scipy import signal
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class SensorDataProcessor:
    """
    Processor for sensor data with various preprocessing capabilities
    """
    
    def __init__(self, sampling_rate: float = 250.0):
        """
        Initialize the sensor data processor
        
        Args:
            sampling_rate: Sampling rate of the sensor data in Hz
        """
        self.sampling_rate = sampling_rate
        self.scaler = StandardScaler()
        self.logger = logging.getLogger('Brain2RL.DataProcessor')
    
    def bandpass_filter(self, data: np.ndarray, low_freq: float = 0.5, 
                       high_freq: float = 100.0, order: int = 4) -> np.ndarray:
        """
        Apply bandpass filter to sensor data
        
        Args:
            data: Input data (channels x time)
            low_freq: Low cutoff frequency
            high_freq: High cutoff frequency
            order: Filter order
            
        Returns:
            Filtered data
        """
        nyquist = self.sampling_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = butter(order, [low, high], btype='band')
        filtered_data = filtfilt(b, a, data, axis=1)
        
        return filtered_data
    
    def notch_filter(self, data: np.ndarray, notch_freq: float = 50.0, 
                    quality: float = 30.0) -> np.ndarray:
        """
        Apply notch filter to remove power line noise
        
        Args:
            data: Input data (channels x time)
            notch_freq: Notch frequency (power line frequency)
            quality: Quality factor
            
        Returns:
            Filtered data
        """
        b, a = signal.iirnotch(notch_freq, quality, self.sampling_rate)
        filtered_data = filtfilt(b, a, data, axis=1)
        
        return filtered_data
    
    def normalize_data(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize sensor data
        
        Args:
            data: Input data
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            Normalized data
        """
        if method == 'zscore':
            return (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        elif method == 'minmax':
            min_val = np.min(data, axis=1, keepdims=True)
            max_val = np.max(data, axis=1, keepdims=True)
            return (data - min_val) / (max_val - min_val)
        elif method == 'robust':
            median = np.median(data, axis=1, keepdims=True)
            mad = np.median(np.abs(data - median), axis=1, keepdims=True)
            return (data - median) / mad
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def remove_artifacts(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Remove artifacts based on statistical thresholding
        
        Args:
            data: Input data (channels x time)
            threshold: Z-score threshold for artifact detection
            
        Returns:
            Data with artifacts removed
        """
        z_scores = np.abs((data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True))
        artifact_mask = z_scores > threshold
        
        # Replace artifacts with interpolated values
        cleaned_data = data.copy()
        for ch in range(data.shape[0]):
            artifact_indices = np.where(artifact_mask[ch])[0]
            if len(artifact_indices) > 0:
                good_indices = np.where(~artifact_mask[ch])[0]
                if len(good_indices) > 0:
                    cleaned_data[ch, artifact_indices] = np.interp(
                        artifact_indices, good_indices, data[ch, good_indices]
                    )
        
        return cleaned_data


def load_sensor_data(data_path: str, file_type: str = 'auto') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load sensor data from various file formats
    
    Args:
        data_path: Path to data file
        file_type: File type ('auto', 'npy', 'npz', 'csv', 'mat', 'h5')
        
    Returns:
        Tuple of (data, labels) where labels can be None
    """
    if file_type == 'auto':
        file_type = Path(data_path).suffix.lower()
    
    if file_type in ['.npy', 'npy']:
        data = np.load(data_path)
        return data, None
    
    elif file_type in ['.npz', 'npz']:
        data_file = np.load(data_path)
        data = data_file['data'] if 'data' in data_file else data_file['arr_0']
        labels = data_file['labels'] if 'labels' in data_file else None
        return data, labels
    
    elif file_type in ['.csv', 'csv']:
        df = pd.read_csv(data_path)
        data = df.values
        return data, None
    
    elif file_type in ['.mat', 'mat']:
        from scipy.io import loadmat
        mat_data = loadmat(data_path)
        # Try common variable names
        for key in ['data', 'EEG', 'signals', 'X']:
            if key in mat_data:
                data = mat_data[key]
                labels = mat_data.get('labels', None)
                return data, labels
        
        # If no common key found, use the first non-metadata key
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if keys:
            data = mat_data[keys[0]]
            return data, None
        else:
            raise ValueError("No data found in .mat file")
    
    elif file_type in ['.h5', 'h5', '.hdf5', 'hdf5']:
        with h5py.File(data_path, 'r') as f:
            # Try common dataset names
            for key in ['data', 'EEG', 'signals', 'X']:
                if key in f:
                    data = f[key][:]
                    labels = f['labels'][:] if 'labels' in f else None
                    return data, labels
            
            # If no common key found, use the first dataset
            keys = list(f.keys())
            if keys:
                data = f[keys[0]][:]
                return data, None
            else:
                raise ValueError("No data found in .h5 file")
    
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def preprocess_sensor_data(data: np.ndarray, sampling_rate: float = 250.0,
                          apply_filters: bool = True, normalize: bool = True,
                          remove_artifacts: bool = True) -> np.ndarray:
    """
    Preprocess sensor data with standard pipeline
    
    Args:
        data: Input data (time x channels or channels x time)
        sampling_rate: Sampling rate in Hz
        apply_filters: Whether to apply bandpass and notch filters
        normalize: Whether to normalize the data
        remove_artifacts: Whether to remove artifacts
        
    Returns:
        Preprocessed data
    """
    # Ensure data is in channels x time format
    if data.shape[0] > data.shape[1]:
        data = data.T
    
    processor = SensorDataProcessor(sampling_rate)
    
    # Apply filters
    if apply_filters:
        data = processor.bandpass_filter(data, low_freq=0.5, high_freq=100.0)
        data = processor.notch_filter(data, notch_freq=50.0)
    
    # Remove artifacts
    if remove_artifacts:
        data = processor.remove_artifacts(data, threshold=3.0)
    
    # Normalize
    if normalize:
        data = processor.normalize_data(data, method='zscore')
    
    return data


def load_eeg_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load EEG data from directory
    
    Args:
        data_dir: Directory containing EEG data files
        
    Returns:
        Tuple of (data, labels)
    """
    data_files = []
    label_files = []
    
    # Find data files
    for file_path in Path(data_dir).rglob('*.npz'):
        if 'data' in file_path.name.lower():
            data_files.append(file_path)
        elif 'label' in file_path.name.lower():
            label_files.append(file_path)
    
    if not data_files:
        # Try other formats
        for ext in ['*.npy', '*.csv', '*.mat', '*.h5']:
            data_files.extend(Path(data_dir).rglob(ext))
    
    if not data_files:
        raise ValueError(f"No data files found in {data_dir}")
    
    # Load and combine data
    all_data = []
    all_labels = []
    
    for data_file in data_files:
        try:
            data, labels = load_sensor_data(str(data_file))
            data = preprocess_sensor_data(data)
            
            all_data.append(data)
            
            if labels is not None:
                all_labels.append(labels)
            else:
                # Create dummy labels
                all_labels.append(np.zeros(data.shape[0]))
        
        except Exception as e:
            logging.warning(f"Could not load {data_file}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data files could be loaded")
    
    # Combine all data
    combined_data = np.concatenate(all_data, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    return combined_data, combined_labels


def load_fmri_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load fMRI data from directory
    
    Args:
        data_dir: Directory containing fMRI data files
        
    Returns:
        Tuple of (data, labels)
    """
    # Similar to EEG data loading but with fMRI-specific preprocessing
    return load_eeg_data(data_dir)  # Simplified for now


def create_dataloader(data: np.ndarray, labels: np.ndarray, 
                     batch_size: int = 32, train_split: float = 0.7,
                     val_split: float = 0.15, test_split: float = 0.15,
                     shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test data loaders
    
    Args:
        data: Input data
        labels: Labels
        batch_size: Batch size
        train_split: Training split ratio
        val_split: Validation split ratio  
        test_split: Test split ratio
        shuffle: Whether to shuffle data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create dataset
    dataset = TensorDataset(torch.FloatTensor(data), torch.LongTensor(labels))
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, test_loader


class TensorDataset(Dataset):
    """Simple tensor dataset"""
    
    def __init__(self, data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def generate_synthetic_sensor_data(n_samples: int = 1000, n_channels: int = 32, 
                                  n_timesteps: int = 512, n_classes: int = 6,
                                  sampling_rate: float = 250.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic sensor data for testing
    
    Args:
        n_samples: Number of samples
        n_channels: Number of channels
        n_timesteps: Number of time steps
        n_classes: Number of classes
        sampling_rate: Sampling rate
        
    Returns:
        Tuple of (data, labels)
    """
    # Create time vector
    time = np.linspace(0, n_timesteps / sampling_rate, n_timesteps)
    
    # Generate synthetic data
    data = np.zeros((n_samples, n_channels, n_timesteps))
    labels = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        # Random class
        class_id = np.random.randint(0, n_classes)
        labels[i] = class_id
        
        # Generate signals based on class
        for ch in range(n_channels):
            # Base signal with class-specific characteristics
            base_freq = 8 + class_id * 2  # Alpha to gamma range
            signal_wave = np.sin(2 * np.pi * base_freq * time)
            
            # Add harmonics
            for harmonic in range(2, 5):
                signal_wave += 0.3 * np.sin(2 * np.pi * base_freq * harmonic * time)
            
            # Add noise
            noise = np.random.normal(0, 0.5, n_timesteps)
            
            # Channel-specific modulation
            channel_mod = np.sin(2 * np.pi * (ch + 1) * 0.1 * time)
            
            # Combine signals
            data[i, ch, :] = signal_wave + noise + 0.2 * channel_mod
    
    return data, labels


def save_processed_data(data: np.ndarray, labels: np.ndarray, 
                       save_path: str, metadata: Optional[Dict] = None):
    """
    Save processed data to file
    
    Args:
        data: Processed data
        labels: Labels
        save_path: Path to save file
        metadata: Optional metadata dictionary
    """
    save_dict = {
        'data': data,
        'labels': labels
    }
    
    if metadata:
        save_dict['metadata'] = metadata
    
    np.savez_compressed(save_path, **save_dict)


def load_processed_data(load_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """
    Load processed data from file
    
    Args:
        load_path: Path to load file
        
    Returns:
        Tuple of (data, labels, metadata)
    """
    data_file = np.load(load_path)
    data = data_file['data']
    labels = data_file['labels']
    metadata = data_file.get('metadata', None)
    
    return data, labels, metadata


def visualize_sensor_data(data: np.ndarray, labels: np.ndarray, 
                         n_samples: int = 5, sampling_rate: float = 250.0,
                         save_path: Optional[str] = None):
    """
    Visualize sensor data
    
    Args:
        data: Sensor data (samples x channels x time)
        labels: Labels
        n_samples: Number of samples to visualize
        sampling_rate: Sampling rate
        save_path: Optional path to save plot
    """
    n_samples = min(n_samples, len(data))
    time = np.linspace(0, data.shape[2] / sampling_rate, data.shape[2])
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Plot first few channels
        n_channels_to_plot = min(5, data.shape[1])
        for ch in range(n_channels_to_plot):
            ax.plot(time, data[i, ch, :], label=f'Channel {ch+1}', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Sample {i+1} - Class {labels[i]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_data_statistics(data: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Compute statistics for sensor data
    
    Args:
        data: Sensor data
        labels: Labels
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'n_samples': len(data),
        'n_channels': data.shape[1],
        'n_timesteps': data.shape[2],
        'n_classes': len(np.unique(labels)),
        'class_distribution': {str(cls): int(np.sum(labels == cls)) for cls in np.unique(labels)},
        'data_shape': data.shape,
        'data_dtype': str(data.dtype),
        'mean_amplitude': float(np.mean(data)),
        'std_amplitude': float(np.std(data)),
        'min_amplitude': float(np.min(data)),
        'max_amplitude': float(np.max(data))
    }
    
    return stats


def main():
    """Example usage of data utilities"""
    # Generate synthetic data
    print("Generating synthetic sensor data...")
    data, labels = generate_synthetic_sensor_data(
        n_samples=1000, n_channels=32, n_timesteps=512, n_classes=6
    )
    
    # Compute statistics
    stats = compute_data_statistics(data, labels)
    print(f"Data statistics: {stats}")
    
    # Visualize data
    print("Visualizing data...")
    visualize_sensor_data(data, labels, n_samples=3)
    
    # Save data
    print("Saving data...")
    save_processed_data(data, labels, 'synthetic_sensor_data.npz', {'stats': stats})
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloader(
        data, labels, batch_size=32
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    print("Data utilities example completed!")


if __name__ == '__main__':
    main() 