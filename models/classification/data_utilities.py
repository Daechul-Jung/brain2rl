"""
Data Utilities Module
This module provides utilities for loading, preprocessing, and managing sensor data
for the classification pipeline.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any

from time_series_dataset import TimeSeriesDataset


def load_sensor_data(data_dir: str, subject_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load sensor data from CSV files
    
    Args:
        data_dir: Directory containing data files
        subject_ids: List of subject IDs to load (if None, load all)
        
    Returns:
        Tuple of (data, labels)
    """
    print(f"Loading sensor data from {data_dir}")
    
    all_data = []
    all_labels = []
    
    # If no subject IDs specified, try to find all CSV files
    if subject_ids is None:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        subject_ids = [f.replace('.csv', '') for f in csv_files]
    
    for subject_id in subject_ids:
        data_file = os.path.join(data_dir, f'{subject_id}.csv')
        
        if not os.path.exists(data_file):
            print(f"Warning: File for subject {subject_id} not found. Skipping...")
            continue
        
        print(f"Loading subject {subject_id}...")
        
        try:
            df = pd.read_csv(data_file)
            
            # Extract sensor features (adjust column names as needed)
            sensor_columns = [col for col in df.columns if any(x in col.lower() for x in ['acc', 'gyro', 'sensor', 'signal'])]
            
            if not sensor_columns:
                # Fallback to common column names
                sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            
            # Check which columns exist
            available_columns = [col for col in sensor_columns if col in df.columns]
            
            if not available_columns:
                print(f"Warning: No sensor columns found for {subject_id}. Available columns: {list(df.columns)}")
                continue
            
            X = df[available_columns].values
            y = df['label'].values if 'label' in df.columns else np.zeros(len(X)) #### Should be replaced into action indicator 
            
            all_data.append(X)
            all_labels.append(y)
            
        except Exception as e:
            print(f"Error loading {subject_id}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from the specified directory")
    
    # Combine data from all subjects
    X = np.concatenate(all_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    print(f"Loaded data shape: {X.shape}, labels shape: {y.shape}")
    return X, y


def preprocess_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler, LabelEncoder]:
    """
    Preprocess the sensor data
    
    Args:
        X: Raw sensor data
        y: Raw labels
        
    Returns:
        Tuple of (preprocessed_data, encoded_labels, scaler, label_encoder)
    """
    print("Preprocessing data...")
    
    # Scale sensor data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"Data preprocessed. Unique labels: {np.unique(y_encoded)}")
    return X_scaled, y_encoded, scaler, label_encoder


def create_dataloaders(X: np.ndarray, y: np.ndarray, window_size: int = 100, 
                      batch_size: int = 32, overlap: float = 0.5) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test dataloaders
    
    Args:
        X: Preprocessed data
        y: Encoded labels
        window_size: Size of sliding window
        batch_size: Batch size for training
        overlap: Overlap between consecutive windows
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, window_size=window_size, overlap=overlap)
    val_dataset = TimeSeriesDataset(X_val, y_val, window_size=window_size, overlap=overlap)
    test_dataset = TimeSeriesDataset(X_test, y_test, window_size=window_size, overlap=overlap)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def validate_data_format(data_dir: str) -> bool:
    """
    Validate that the data directory contains properly formatted sensor data
    
    Args:
        data_dir: Directory containing sensor data
        
    Returns:
        True if data format is valid, False otherwise
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' does not exist")
        return False
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"Error: No CSV files found in '{data_dir}'")
        print("Expected format: CSV files with sensor data columns and 'label' column")
        return False
    
    print(f"Found {len(csv_files)} CSV files in data directory")
    for f in csv_files[:5]:  # Show first 5 files
        print(f"  - {f}")
    if len(csv_files) > 5:
        print(f"  ... and {len(csv_files) - 5} more files")
    
    return True


def get_data_info(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Get information about the dataset
    
    Args:
        X: Sensor data
        y: Labels
        
    Returns:
        Dictionary with dataset information
    """
    return {
        'original_shape': X.shape,
        'n_samples': X.shape[0],
        'n_channels': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': np.bincount(y),
        'unique_labels': np.unique(y),
        'data_type': str(X.dtype),
        'label_type': str(y.dtype)
    }


def save_preprocessing_info(scaler: StandardScaler, label_encoder: LabelEncoder, save_path: str):
    """
    Save preprocessing information for later use
    
    Args:
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
        save_path: Path to save the preprocessing info
    """
    import pickle
    
    preprocessing_info = {
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': getattr(scaler, 'feature_names_in_', None),
        'n_features': scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(preprocessing_info, f)
    
    print(f"Preprocessing information saved to: {save_path}")


def load_preprocessing_info(load_path: str) -> Dict[str, Any]:
    """
    Load preprocessing information
    
    Args:
        load_path: Path to the saved preprocessing info
        
    Returns:
        Dictionary with preprocessing information
    """
    import pickle
    
    with open(load_path, 'rb') as f:
        preprocessing_info = pickle.load(f)
    
    print(f"Preprocessing information loaded from: {load_path}")
    return preprocessing_info

