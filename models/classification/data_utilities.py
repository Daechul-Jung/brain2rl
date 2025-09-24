"""
Data Utilities Module
This module provides utilities for loading, preprocessing, and managing sensor data
for the classification pipeline.
"""

import os, sys
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classification.time_series_dataset import TimeSeriesDataset


def load_sensor_data(data_dir: str, subject_ids: Optional[List[str]] = None, group_by:str = 'sequence_id') -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Load sensor data from CSV files
    
    Args:
        data_dir: Directory containing data files
        subject_ids: List of subject IDs to load (if None, load all)
        group_by: Group by sequence_id to make it as time series 
        
    Returns:
        Returns:
        X: np.ndarray of shape (N_rows, C_sensors)
        y: dict with 'behavior' and 'gesture' arrays of length N_rows
        groups: np.ndarray of length N_rows (e.g., sequence_id)
        df: the cleaned pandas DataFrame (for debugging/inspection)
    """
    print(f"Loading sensor data from {data_dir}")
    df = pd.read_csv(data_dir)
    subject_ids = np.unique(df['subject'].dropna().astype(str).unique())
    print(f'total subjects in the training datset: {len(subject_ids)}')
    if subject_ids is not None:
        subject_ids = [str(s) for s in subject_ids]
        df = df[df['subject'].astype(str).isin(subject_ids)].copy()

    if 'sequence_id' not in df.columns:
        if 'row_id' in df.columns:
            df['sequence_id'] = df['row_id'].astype(str).str.split('_').str[:2].str.join('_')
        ## Change sequence id 
        elif 'sequence_id' in df.columns:
            df['sequence_id'] = df['sequence'].astype(str)
            

    sort_cols = ['subject', 'sequence_id']
    if 'sequence_counter' in df.columns:
        sort_cols.append('sequence_counter')

    df = df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)
    prefixed = ("acc_", "rot_", "thm_", "tof_")
    sensor_cols = [c for c in df.columns if c.startswith(prefixed)]

    if not sensor_cols:
        fallback = ["acc_x","acc_y","acc_z","rot_w","rot_x","rot_y","rot_z"]
        sensor_cols = [c for c in fallback if c in df.columns]
    if not sensor_cols:
        raise ValueError("No sensor columns found. Check your column names and prefixes.")

    if 'behavior' not in df.columns or 'gesture' not in df.columns:
        raise ValueError("Expected 'behavior' and 'gesture' columns in the CSV.")

    df[sensor_cols] = df[sensor_cols].replace(-1, np.nan) ### replace -1 value to nan

    df[sensor_cols] = (
        df.groupby('sequence_id', sort=False)[sensor_cols] ## group by sequence id and do not sort
        .apply(lambda g: g.ffill().bfill()) ## 
        .reset_index(level=0, drop=True)
    )

    for c in sensor_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    X = df[sensor_cols].astype(np.float32).values
    y = {
        'behavior': df['behavior'].astype(str).values,
        'gesture': df['gesture'].astype(str).values
    }

    groups = df[group_by].astype(str).values if group_by in df.columns else np.zeros(len(df), dtype=str)

    print(f"Loaded X: {X.shape} (Number of subjects, number of sensors). "
          f"Behaviors: {len(np.unique(y['behavior']))}, Gestures: {len(np.unique(y['gesture']))}. "
          f"Group key: '{group_by}'.")

    return X, y, groups, df

def preprocess_multilabel(X: np.ndarray, y_str: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray], StandardScaler, Dict[str, LabelEncoder]]:
    print('Preprocessing (scaling + encoding two targets)')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    encoders = {}
    y_encoders = {}
    for key in ['behavior', 'gesture']:
        le = LabelEncoder()
        y_encoders[key] = le.fit_transform(y_str[key])
        encoders[key] = le
        
        print(f'{key}: {len(le.classes_)} classes')
    return X_scaled, y_encoders, scaler, encoders

def create_train_val_loaders(
    X_train, y_train_enc, groups_train,
    window_size=50, batch_size=64, overlap=0.5, task='both'):
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    tr_idx, val_idx = next(gss.split(X_train, y_train_enc['behavior'], groups=groups_train))

    tr = (X_train[tr_idx], {k: v[tr_idx] for k,v in y_train_enc.items()}, groups_train[tr_idx])
    va = (X_train[val_idx], {k: v[val_idx] for k,v in y_train_enc.items()}, groups_train[val_idx])

    ds_tr = TimeSeriesDataset(*tr, window_size=window_size, overlap=overlap, task=task)
    ds_va = TimeSeriesDataset(*va, window_size=window_size, overlap=overlap, task=task)

    return (
        DataLoader(ds_tr, batch_size=batch_size, shuffle=True),
        DataLoader(ds_va, batch_size=batch_size, shuffle=False),
    )

def create_test_loader(X_test, y_test_enc, groups_test,window_size=256, batch_size=64, overlap=0.5, task='both'):
    ds_te = TimeSeriesDataset(X_test, y_test_enc, groups_test, window_size=window_size, overlap=overlap, task=task)
    return DataLoader(ds_te, batch_size=batch_size, shuffle=False)



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

