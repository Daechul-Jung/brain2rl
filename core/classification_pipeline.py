"""
Classification Pipeline
======================

This module handles the classification of sensor data (EEG, IMU, etc.) to identify actions.
Supports both offline training and real-time inference.

Author: Brain2RL Team
Version: 1.0.0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from brain2rl.models.classification.action_classifier import EEGConvNet
from brain2rl.utils.data_utils import load_sensor_data, preprocess_sensor_data


class SensorDataset(Dataset):
    """Dataset for sensor data with windowing support"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, window_size: int = 512, 
                 overlap: float = 0.5, transform=None):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        
        # Calculate step size and number of windows
        self.step_size = int(window_size * (1 - overlap))
        self.n_windows = (len(data) - window_size) // self.step_size + 1
        
    def __len__(self):
        return self.n_windows
    
    def __getitem__(self, idx):
        start_idx = idx * self.step_size
        end_idx = start_idx + self.window_size
        
        window_data = self.data[start_idx:end_idx]
        window_labels = self.labels[start_idx:end_idx]
        
        # For classification, we might want to use majority voting or take the label at the center
        if len(window_labels.shape) > 1:
            # Multi-class labels - take majority vote
            center_label = window_labels[len(window_labels) // 2]
        else:
            center_label = window_labels[len(window_labels) // 2]
        
        # Transpose to (channels, time) format for CNN
        window_data = window_data.T
        
        if self.transform:
            window_data = self.transform(window_data)
        
        return torch.FloatTensor(window_data), torch.LongTensor(center_label)


class ClassificationPipeline:
    """
    Pipeline for classifying sensor data into actions
    """
    
    def __init__(self, data_dir: str, model_config: Dict[str, Any]):
        """
        Initialize the classification pipeline
        
        Args:
            data_dir: Directory containing sensor data
            model_config: Configuration for the classification model
        """
        self.data_dir = data_dir
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logger
        self.logger = logging.getLogger('Brain2RL.Classification')
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Data components
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        self.logger.info("Classification pipeline initialized")
    
    def load_and_preprocess_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess sensor data
        
        Args:
            data_path: Path to sensor data file
            
        Returns:
            Tuple of (preprocessed_data, labels)
        """
        self.logger.info(f"Loading sensor data from {data_path}")
        
        # Load raw sensor data
        if data_path.endswith('.npy'):
            data = np.load(data_path)
        elif data_path.endswith('.npz'):
            data_file = np.load(data_path)
            data = data_file['data']
            labels = data_file['labels'] if 'labels' in data_file else None
        else:
            # Try to load as CSV or other format
            try:
                data = np.loadtxt(data_path, delimiter=',')
                labels = None
            except:
                raise ValueError(f"Unsupported data format: {data_path}")
        
        # Preprocess the data
        preprocessed_data = preprocess_sensor_data(data)
        
        # If labels are not provided, create dummy labels
        if labels is None:
            labels = np.zeros(len(preprocessed_data), dtype=int)
        
        self.logger.info(f"Loaded data shape: {preprocessed_data.shape}, Labels shape: {labels.shape}")
        return preprocessed_data, labels
    
    def prepare_datasets(self, data: np.ndarray, labels: np.ndarray, 
                        train_split: float = 0.7, val_split: float = 0.15):
        """
        Prepare train/validation/test datasets
        
        Args:
            data: Preprocessed sensor data
            labels: Action labels
            train_split: Proportion of data for training
            val_split: Proportion of data for validation
        """
        n_samples = len(data)
        train_size = int(n_samples * train_split)
        val_size = int(n_samples * val_split)
        
        # Split data
        train_data = data[:train_size]
        train_labels = labels[:train_size]
        
        val_data = data[train_size:train_size + val_size]
        val_labels = labels[train_size:train_size + val_size]
        
        test_data = data[train_size + val_size:]
        test_labels = labels[train_size + val_size:]
        
        # Create datasets
        self.train_dataset = SensorDataset(
            train_data, train_labels, 
            window_size=self.model_config['n_times'],
            overlap=0.5
        )
        
        self.val_dataset = SensorDataset(
            val_data, val_labels,
            window_size=self.model_config['n_times'],
            overlap=0.5
        )
        
        self.test_dataset = SensorDataset(
            test_data, test_labels,
            window_size=self.model_config['n_times'],
            overlap=0.5
        )
        
        self.logger.info(f"Created datasets - Train: {len(self.train_dataset)}, "
                        f"Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
    
    def initialize_model(self, n_channels: int, n_classes: int):
        """
        Initialize the classification model
        
        Args:
            n_channels: Number of sensor channels
            n_classes: Number of action classes
        """
        self.model = EEGConvNet(
            n_channels=n_channels,
            n_times=self.model_config['n_times'],
            n_classes=n_classes,
            dropout_rate=self.model_config['dropout_rate']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_config['learning_rate']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Model initialized with {n_channels} channels, {n_classes} classes")
    
    def train_model(self, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the classification model
        
        Args:
            epochs: Number of training epochs (uses config if not provided)
            
        Returns:
            Dictionary with training history
        """
        if epochs is None:
            epochs = self.model_config['epochs']
        
        if self.train_dataset is None:
            raise ValueError("No training data loaded. Call load_and_preprocess_data first.")
        
        if self.model is None:
            # Auto-initialize model based on data
            sample_data, sample_label = self.train_dataset[0]
            n_channels = sample_data.shape[0]
            n_classes = len(np.unique([self.train_dataset[i][1] for i in range(min(100, len(self.train_dataset)))]))
            self.initialize_model(n_channels, n_classes)
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.model_config['batch_size'],
            shuffle=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False
        )
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            val_loss /= len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def classify_sensor_data(self, data_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify sensor data using the trained model
        
        Args:
            data_path: Path to sensor data
            
        Returns:
            Tuple of (predicted_actions, confidence_scores)
        """
        if not self.is_trained and self.model is None:
            raise ValueError("Model must be trained or loaded before classification")
        
        # Load and preprocess data
        data, _ = self.load_and_preprocess_data(data_path)
        
        # Create dataset for inference
        inference_dataset = SensorDataset(
            data, np.zeros(len(data)),  # Dummy labels
            window_size=self.model_config['n_times'],
            overlap=0.5
        )
        
        inference_loader = DataLoader(
            inference_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False
        )
        
        # Inference
        self.model.eval()
        all_predictions = []
        all_confidences = []
        
        with torch.no_grad():
            for batch_data, _ in tqdm(inference_loader, desc="Classifying"):
                batch_data = batch_data.to(self.device)
                
                outputs = self.model(batch_data)
                probabilities = F.softmax(outputs, dim=1)
                
                predicted_classes = outputs.argmax(dim=1)
                confidence_scores = probabilities.max(dim=1)[0]
                
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_confidences.extend(confidence_scores.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_confidences)
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'model_config': self.model_config,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        torch.save(save_data, save_path)
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load a pre-trained model"""
        save_data = torch.load(load_path, map_location=self.device)
        
        self.model_config = save_data['model_config']
        self.training_history = save_data['training_history']
        self.is_trained = save_data['is_trained']
        
        # Initialize model with saved config
        if self.model is None:
            # We need to determine the model architecture
            # This requires knowing the number of channels and classes
            # For now, we'll assume they're in the config
            self.initialize_model(
                self.model_config['n_channels'],
                self.model_config['n_classes']
            )
        
        self.model.load_state_dict(save_data['model_state_dict'])
        self.logger.info(f"Model loaded from {load_path}")
    
    def evaluate_model(self) -> Dict[str, float]:
        """Evaluate the model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if self.test_dataset is None:
            raise ValueError("No test data available")
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.model_config['batch_size'],
            shuffle=False
        )
        
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                loss = self.criterion(outputs, batch_labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += batch_labels.size(0)
                test_correct += predicted.eq(batch_labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc
        }
        
        self.logger.info(f"Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        return results


def main():
    """Main function for standalone classification pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification Pipeline')
    parser.add_argument('--data-path', type=str, required=True, help='Path to sensor data')
    parser.add_argument('--mode', choices=['train', 'evaluate', 'classify'], default='train')
    parser.add_argument('--model-path', type=str, help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create model config
    model_config = {
        'n_channels': 32,  # Will be auto-detected
        'n_times': 512,
        'n_classes': 6,  # Will be auto-detected
        'dropout_rate': 0.5,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }
    
    # Initialize pipeline
    pipeline = ClassificationPipeline(
        data_dir=os.path.dirname(args.data_path),
        model_config=model_config
    )
    
    if args.mode == 'train':
        # Load and prepare data
        data, labels = pipeline.load_and_preprocess_data(args.data_path)
        pipeline.prepare_datasets(data, labels)
        
        # Train model
        history = pipeline.train_model()
        
        # Save model
        if args.model_path:
            pipeline.save_model(args.model_path)
        
        # Evaluate
        results = pipeline.evaluate_model()
        print(f"Training completed. Final test accuracy: {results['test_accuracy']:.2f}%")
        
    elif args.mode == 'evaluate':
        # Load model
        if args.model_path:
            pipeline.load_model(args.model_path)
        
        # Load test data
        data, labels = pipeline.load_and_preprocess_data(args.data_path)
        pipeline.prepare_datasets(data, labels)
        
        # Evaluate
        results = pipeline.evaluate_model()
        print(f"Evaluation completed. Test accuracy: {results['test_accuracy']:.2f}%")
        
    elif args.mode == 'classify':
        # Load model
        if args.model_path:
            pipeline.load_model(args.model_path)
        
        # Classify data
        predictions, confidences = pipeline.classify_sensor_data(args.data_path)
        
        print(f"Classification completed. Processed {len(predictions)} samples")
        print(f"Average confidence: {np.mean(confidences):.3f}")
        
        # Save results
        np.savez('classification_results.npz', 
                predictions=predictions, 
                confidences=confidences)


if __name__ == '__main__':
    main() 