import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm

class SensorDataProcessor:
    """Data processing pipeline for sensor data"""
    def __init__(self, window_size=100, overlap=0.5, feature_dir='features'):
        self.window_size = window_size
        self.overlap = overlap
        self.feature_dir = feature_dir
        self.scaler = StandardScaler()
        self.feature_extractor = None
        
        # Create feature directory if it doesn't exist
        os.makedirs(feature_dir, exist_ok=True)
    
    def load_and_preprocess(self, data_path, is_training=True):
        """
        Load and preprocess sensor data
        
        Args:
            data_path: Path to the data file
            is_training: Whether this is training data (for scaling)
        
        Returns:
            Processed data and labels (if training)
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Extract sensor data
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        X = df[sensor_cols].values
        
        # Scale the data
        if is_training:
            X = self.scaler.fit_transform(X)
            # Save the scaler for later use
            joblib.dump(self.scaler, os.path.join(self.feature_dir, 'scaler.joblib'))
        else:
            # Load and use the saved scaler
            scaler = joblib.load(os.path.join(self.feature_dir, 'scaler.joblib'))
            X = scaler.transform(X)
        
        if is_training:
            y = df['label'].values
            return X, y
        return X
    
    def create_windows(self, data, labels=None):
        """
        Create overlapping windows from the data
        
        Args:
            data: Sensor data array
            labels: Optional labels array
        
        Returns:
            Windowed data and labels (if provided)
        """
        n_samples = len(data)
        stride = int(self.window_size * (1 - self.overlap))
        
        windows = []
        window_labels = []
        
        for start in range(0, n_samples - self.window_size, stride):
            end = start + self.window_size
            window = data[start:end]
            windows.append(window)
            
            if labels is not None:
                window_label = labels[start:end]
                # Use majority voting for window label
                label = np.bincount(window_label).argmax()
                window_labels.append(label)
        
        windows = np.array(windows)
        
        if labels is not None:
            return windows, np.array(window_labels)
        return windows
    
    def extract_cnn_features(self, windows, model, device):
        """
        Extract features from CNN layers
        
        Args:
            windows: Windowed sensor data
            model: Trained CNN model
            device: Device to run inference on
        
        Returns:
            Extracted features from different CNN layers
        """
        model.eval()
        features = {
            'temporal': [],
            'feature1': [],
            'feature2': []
        }
        
        # Create a DataLoader for efficient processing
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(windows).transpose(1, 2)  # [batch, channels, time]
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting features"):
                x = batch[0].to(device)
                
                # Get features from each layer
                temporal = model.temporal_conv(x)
                feature1 = model.feature_conv(temporal)
                feature2 = model.feature_conv2(feature1)
                
                # Store features
                features['temporal'].append(temporal.cpu().numpy())
                features['feature1'].append(feature1.cpu().numpy())
                features['feature2'].append(feature2.cpu().numpy())
        
        # Concatenate features from all batches
        for key in features:
            features[key] = np.concatenate(features[key], axis=0)
        
        return features
    
    def save_features(self, features, subject_id, split='train'):
        """
        Save extracted features
        
        Args:
            features: Dictionary of features
            subject_id: Subject ID
            split: Data split (train/test)
        """
        for layer_name, feature_data in features.items():
            save_path = os.path.join(
                self.feature_dir,
                f'{split}_{subject_id}_{layer_name}_features.npy'
            )
            np.save(save_path, feature_data)
    
    def load_features(self, subject_id, split='train'):
        """
        Load saved features
        
        Args:
            subject_id: Subject ID
            split: Data split (train/test)
        
        Returns:
            Dictionary of loaded features
        """
        features = {}
        for layer_name in ['temporal', 'feature1', 'feature2']:
            load_path = os.path.join(
                self.feature_dir,
                f'{split}_{subject_id}_{layer_name}_features.npy'
            )
            if os.path.exists(load_path):
                features[layer_name] = np.load(load_path)
        return features

class RLFeatureDataset(Dataset):
    """Dataset for RL using CNN features"""
    def __init__(self, features, labels=None):
        """
        Args:
            features: Dictionary of features from different CNN layers
            labels: Optional labels for supervised learning
        """
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(next(iter(self.features.values())))
    
    def __getitem__(self, idx):
        # Combine features from different layers
        combined_features = []
        for layer_name in ['temporal', 'feature1', 'feature2']:
            feature = self.features[layer_name][idx]
            # Flatten the feature
            combined_features.append(feature.flatten())
        
        # Concatenate all features
        x = np.concatenate(combined_features)
        x = torch.FloatTensor(x)
        
        if self.labels is not None:
            y = torch.LongTensor([self.labels[idx]])
            return x, y
        return x

def create_rl_state_representation(features, method='concat'):
    """
    Create state representation for RL from CNN features
    
    Args:
        features: Dictionary of features from different CNN layers
        method: Method to combine features ('concat', 'mean', 'max')
    
    Returns:
        State representation
    """
    if method == 'concat':
        # Concatenate features from all layers
        return np.concatenate([f.flatten() for f in features.values()])
    
    elif method == 'mean':
        # Take mean of features across time
        return np.mean([f for f in features.values()], axis=0)
    
    elif method == 'max':
        # Take max of features across time
        return np.max([f for f in features.values()], axis=0)
    
    else:
        raise ValueError(f"Unknown method: {method}")

def main():
    # Example usage
    processor = SensorDataProcessor()
    
    # Process training data
    train_data, train_labels = processor.load_and_preprocess('data/train.csv', is_training=True)
    train_windows, train_window_labels = processor.create_windows(train_data, train_labels)
    
    # Load trained model
    model = torch.load('best_model.pth')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Extract features
    train_features = processor.extract_cnn_features(train_windows, model, device)
    
    # Save features
    processor.save_features(train_features, 'train_subject_1', split='train')
    
    # Create RL dataset
    rl_dataset = RLFeatureDataset(train_features, train_window_labels)
    
    # Create state representation for RL
    state_representation = create_rl_state_representation(train_features, method='concat')
    
    print("Feature shapes:")
    for layer_name, feature in train_features.items():
        print(f"{layer_name}: {feature.shape}")
    
    print(f"\nRL state representation shape: {state_representation.shape}")

if __name__ == '__main__':
    main() 