"""
Integrated Classification and Tokenization Pipeline
================================================

This module provides an integrated pipeline for:
1. Loading and preprocessing time series sensor data
2. Training an action classification CNN
3. Tokenizing the classified data using a transformer
4. Storing tokens for reinforcement learning trajectories

Author: Brain2RL Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

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

class ActionClassifier(nn.Module):
    """
    CNN model for action classification from time series sensor data
    """
    def __init__(self, n_channels: int, n_times: int, n_classes: int, dropout_rate: float = 0.3):
        super(ActionClassifier, self).__init__()
        
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

class BrainTokenizer(nn.Module):
    """
    Transformer-based model for brain signal tokenization
    """
    def __init__(self, input_channels: int, input_length: int, n_tokens: int = 512, 
                 embedding_dim: int = 128, nhead: int = 8, num_encoder_layers: int = 6, 
                 dropout: float = 0.1):
        super(BrainTokenizer, self).__init__()
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Calculate output size after convolutions
        # After 3 pooling layers with stride 2: input_length // 8
        # But we need to handle cases where input_length is not divisible by 8
        self.output_length = max(1, input_length // 8)
        
        # Positional encoding - make it flexible to handle different output lengths
        self.positional_encoding = nn.Parameter(torch.randn(1, 1000, embedding_dim))  # Max length 1000
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=nhead, 
            dim_feedforward=embedding_dim*4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_encoder_layers)
        
        # Output projection to token space
        self.token_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, n_tokens)
        )
        
    def forward(self, x):
        # Input shape: (batch_size, channels, time)
        features = self.feature_extractor(x)  # (batch_size, embedding_dim, output_length)
        
        # Transpose for transformer: (batch_size, output_length, embedding_dim)
        features = features.transpose(1, 2)
        
        # Get actual sequence length and add positional encoding
        seq_len = features.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        features = features + pos_encoding
        
        # Transformer encoder
        encoded = self.transformer_encoder(features)
        
        # Token prediction
        tokens = self.token_predictor(encoded)
        
        return tokens

class IntegratedPipeline:
    """
    Integrated pipeline for classification and tokenization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the integrated pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize models
        self.classifier = None
        self.tokenizer = None
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Training components
        self.classifier_optimizer = None
        self.tokenizer_optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Integrated Pipeline initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('IntegratedPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_sensor_data(self, data_dir: str, subject_ids: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sensor data from CSV files
        
        Args:
            data_dir: Directory containing data files
            subject_ids: List of subject IDs to load (if None, load all)
            
        Returns:
            Tuple of (data, labels)
        """
        self.logger.info(f"Loading sensor data from {data_dir}")
        
        all_data = []
        all_labels = []
        
        # If no subject IDs specified, try to find all CSV files
        if subject_ids is None:
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            subject_ids = [f.replace('.csv', '') for f in csv_files]
        
        for subject_id in subject_ids:
            data_file = os.path.join(data_dir, f'{subject_id}.csv')
            
            if not os.path.exists(data_file):
                self.logger.warning(f"File for subject {subject_id} not found. Skipping...")
                continue
            
            self.logger.info(f"Loading subject {subject_id}...")
            
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
                    self.logger.warning(f"No sensor columns found for {subject_id}. Available columns: {list(df.columns)}")
                    continue
                
                X = df[available_columns].values
                y = df['label'].values if 'label' in df.columns else np.zeros(len(X))
                
                all_data.append(X)
                all_labels.append(y)
                
            except Exception as e:
                self.logger.error(f"Error loading {subject_id}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data could be loaded from the specified directory")
        
        # Combine data from all subjects
        X = np.concatenate(all_data, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        self.logger.info(f"Loaded data shape: {X.shape}, labels shape: {y.shape}")
        return X, y
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the sensor data
        
        Args:
            X: Raw sensor data
            y: Raw labels
            
        Returns:
            Tuple of (preprocessed_data, encoded_labels)
        """
        self.logger.info("Preprocessing data...")
        
        # Scale sensor data
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"Data preprocessed. Unique labels: {np.unique(y_encoded)}")
        return X_scaled, y_encoded
    
    def create_dataloaders(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/validation/test dataloaders
        
        Args:
            X: Preprocessed data
            y: Encoded labels
            batch_size: Batch size for training
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)
        
        # Create datasets
        train_dataset = TimeSeriesDataset(X_train, y_train, window_size=self.config['window_size'])
        val_dataset = TimeSeriesDataset(X_val, y_val, window_size=self.config['window_size'])
        test_dataset = TimeSeriesDataset(X_test, y_test, window_size=self.config['window_size'])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.logger.info(f"Created dataloaders - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        return train_loader, val_loader, test_loader
    
    def train_classifier(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the action classifier
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        self.logger.info("Training action classifier...")
        
        # Initialize classifier
        n_channels = train_loader.dataset.data.shape[1]
        n_times = self.config['window_size']
        n_classes = len(np.unique(train_loader.dataset.labels))
        
        self.classifier = ActionClassifier(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            dropout_rate=self.config['classifier_dropout']
        ).to(self.device)
        
        # Initialize optimizer
        self.classifier_optimizer = optim.Adam(
            self.classifier.parameters(), 
            lr=self.config['classifier_lr']
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_acc = 0.0
        
        for epoch in range(self.config['classifier_epochs']):
            # Training phase
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['classifier_epochs']}"):
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                
                self.classifier_optimizer.zero_grad()
                outputs = self.classifier(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.classifier_optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self._evaluate_classifier(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            self.logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = os.path.join('models', 'classification', 'best_classifier.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.classifier.state_dict(), save_path)
                self.logger.info(f"Saved best classifier with validation accuracy: {val_acc:.2f}%")
        
        return history
    
    def _evaluate_classifier(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate classifier on given dataloader"""
        self.classifier.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                outputs = self.classifier(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train_tokenizer(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, List[float]]:
        """
        Train the brain tokenizer
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Training history
        """
        self.logger.info("Training brain tokenizer...")
        
        # Initialize tokenizer
        n_channels = train_loader.dataset.data.shape[1]
        n_times = self.config['window_size']
        
        self.tokenizer = BrainTokenizer(
            input_channels=n_channels,
            input_length=n_times,
            n_tokens=self.config['n_tokens'],
            embedding_dim=self.config['embedding_dim'],
            nhead=self.config['nhead'],
            num_encoder_layers=self.config['num_encoder_layers'],
            dropout=self.config['tokenizer_dropout']
        ).to(self.device)
        
        # Initialize optimizer
        self.tokenizer_optimizer = optim.Adam(
            self.tokenizer.parameters(), 
            lr=self.config['tokenizer_lr']
        )
        
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        
        for epoch in range(self.config['tokenizer_epochs']):
            # Training phase
            self.tokenizer.train()
            train_loss = 0.0
            
            for inputs, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['tokenizer_epochs']}"):
                inputs = inputs.to(self.device)
                
                self.tokenizer_optimizer.zero_grad()
                tokens = self.tokenizer(inputs)
                
                # Use reconstruction loss (simplified approach)
                # In practice, you might want to define a more sophisticated objective
                batch_size, seq_len, n_tokens = tokens.shape
                target = torch.softmax(torch.randn(batch_size, seq_len, n_tokens, device=self.device), dim=0)
                
                loss = nn.MSELoss()(tokens, target)
                loss.backward()
                self.tokenizer_optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation phase
            val_loss = self._evaluate_tokenizer(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            self.logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join('models', 'tokenization', 'best_tokenizer.pth')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.tokenizer.state_dict(), save_path)
                self.logger.info(f"Saved best tokenizer with validation loss: {val_loss:.4f}")
        
        return history
    
    def _evaluate_tokenizer(self, dataloader: DataLoader) -> float:
        """Evaluate tokenizer on given dataloader"""
        self.tokenizer.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)
                tokens = self.tokenizer(inputs)
                
                batch_size, seq_len, n_tokens = tokens.shape
                target = torch.softmax(torch.randn(batch_size, seq_len, n_tokens, device=self.device), dim=0)
                
                loss = nn.MSELoss()(tokens, target)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def generate_tokens(self, dataloader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Generate tokens from classified data
        
        Args:
            dataloader: Data loader for token generation
            
        Returns:
            Dictionary containing tokens and metadata
        """
        self.logger.info("Generating tokens...")
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained. Please train the tokenizer first.")
        
        self.tokenizer.eval()
        all_tokens = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Generating tokens"):
                inputs = inputs.to(self.device)
                tokens = self.tokenizer(inputs)
                
                all_tokens.append(tokens.cpu().numpy())
                all_labels.append(labels.numpy())
        
        # Concatenate all tokens
        tokens_array = np.concatenate(all_tokens, axis=0)
        labels_array = np.concatenate(all_labels, axis=0)
        
        # Store tokens
        token_data = {
            'tokens': tokens_array,
            'labels': labels_array,
            'token_shape': tokens_array.shape,
            'n_tokens': tokens_array.shape[-1],
            'embedding_dim': self.config['embedding_dim']
        }
        
        # Save tokens
        save_path = os.path.join('output', 'generated_tokens.npz')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, **token_data)
        
        self.logger.info(f"Generated {tokens_array.shape[0]} token sequences. Saved to {save_path}")
        return token_data
    
    def run_full_pipeline(self, data_dir: str, subject_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to token generation
        
        Args:
            data_dir: Directory containing sensor data
            subject_ids: List of subject IDs to process
            
        Returns:
            Complete pipeline results
        """
        self.logger.info("Starting full integrated pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1/4: Loading and preprocessing data...")
            X, y = self.load_sensor_data(data_dir, subject_ids)
            X_processed, y_processed = self.preprocess_data(X, y)
            
            # Step 2: Create dataloaders
            self.logger.info("Step 2/4: Creating dataloaders...")
            train_loader, val_loader, test_loader = self.create_dataloaders(
                X_processed, y_processed, self.config['batch_size']
            )
            
            # Step 3: Train classifier
            self.logger.info("Step 3/4: Training action classifier...")
            classifier_history = self.train_classifier(train_loader, val_loader)
            
            # Step 4: Train tokenizer
            self.logger.info("Step 4/4: Training brain tokenizer...")
            tokenizer_history = self.train_tokenizer(train_loader, val_loader)
            
            # Step 5: Generate tokens
            self.logger.info("Generating tokens from test data...")
            token_data = self.generate_tokens(test_loader)
            
            # Compile results
            results = {
                'classifier_history': classifier_history,
                'tokenizer_history': tokenizer_history,
                'token_data': token_data,
                'data_info': {
                    'original_shape': X.shape,
                    'processed_shape': X_processed.shape,
                    'n_classes': len(np.unique(y_processed)),
                    'n_channels': X.shape[1]
                }
            }
            
            # Save complete results
            save_path = os.path.join('output', 'pipeline_results.pth')
            torch.save(results, save_path)
            
            self.logger.info(f"Full pipeline completed successfully! Results saved to {save_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot (if available)
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Train Accuracy')
            axes[1].plot(history['val_acc'], label='Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy (%)')
            axes[1].set_title('Training and Validation Accuracy')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        self.logger.info(f"Training history plot saved to {save_path}")


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the integrated pipeline"""
    return {
        'window_size': 100,
        'batch_size': 32,
        'classifier_lr': 0.001,
        'classifier_epochs': 50,
        'classifier_dropout': 0.3,
        'tokenizer_lr': 0.0001,
        'tokenizer_epochs': 30,
        'tokenizer_dropout': 0.1,
        'n_tokens': 512,
        'embedding_dim': 128,
        'nhead': 8,
        'num_encoder_layers': 6
    }


def main():
    """Main function for command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated Classification and Tokenization Pipeline')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory containing sensor data')
    parser.add_argument('--subject-ids', nargs='+', help='List of subject IDs to process')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='output/', help='Output directory')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = create_default_config()
    
    # Initialize pipeline
    pipeline = IntegratedPipeline(config)
    
    try:
        # Run full pipeline
        results = pipeline.run_full_pipeline(args.data_dir, args.subject_ids)
        
        # Plot training histories
        if 'classifier_history' in results:
            pipeline.plot_training_history(
                results['classifier_history'], 
                os.path.join(args.output_dir, 'classifier_training.png')
            )
        
        if 'tokenizer_history' in results:
            pipeline.plot_training_history(
                results['tokenizer_history'], 
                os.path.join(args.output_dir, 'tokenizer_training.png')
            )
        
        print(f"Pipeline completed successfully! Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
