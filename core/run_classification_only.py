#!/usr/bin/env python3
"""
Classification-Only Pipeline Runner

This script runs only the classification part of the Brain2RL pipeline,
skipping tokenization and RL state creation.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import classification components
from models.classification.time_series_dataset import TimeSeriesDataset
from models.classification.action_classifier import ActionClassifier
from models.classification.data_utilities import (
    load_sensor_data, preprocess_data, create_dataloaders, 
    validate_data_format, get_data_info, save_preprocessing_info
)


class ClassificationOnlyPipeline:
    """
    Pipeline that runs only the classification part
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the classification-only pipeline
        
        Args:
            config: Configuration dictionary with classification settings
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize classifier
        self.classifier = None
        
        # Data preprocessing
        self.scaler = None
        self.label_encoder = None
        
        # Training components
        self.classifier_optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Classification-Only Pipeline initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ClassificationOnlyPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_and_preprocess_data(self, data_dir: str, subject_ids: Optional[List[str]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Load and preprocess sensor data
        
        Args:
            data_dir: Directory containing sensor data
            subject_ids: List of subject IDs to load
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        self.logger.info("Loading and preprocessing data...")
        
        # Validate data format
        if not validate_data_format(data_dir):
            raise ValueError(f"Invalid data format in {data_dir}")
        
        # Load sensor data
        X, y = load_sensor_data(data_dir, subject_ids)
        
        # Preprocess data
        X_processed, y_processed, self.scaler, self.label_encoder = preprocess_data(X, y)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            X_processed, y_processed, 
            window_size=self.config['window_size'],
            batch_size=self.config['batch_size'],
            overlap=0.5
        )
        
        # Save preprocessing info
        os.makedirs('models/classification', exist_ok=True)
        save_preprocessing_info(
            self.scaler, self.label_encoder, 
            'models/classification/preprocessing_info.pkl'
        )
        
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
            
            for inputs, labels in train_loader:
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
    
    def evaluate_on_test_set(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate the trained classifier on the test set
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary with test results
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained. Please train the classifier first.")
        
        self.logger.info("Evaluating classifier on test set...")
        
        test_loss, test_acc = self._evaluate_classifier(test_loader)
        
        # Get predictions for confusion matrix
        all_predictions = []
        all_labels = []
        
        self.classifier.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.squeeze().to(self.device)
                outputs = self.classifier(inputs)
                _, predicted = outputs.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'predictions': np.array(all_predictions),
            'true_labels': np.array(all_labels),
            'n_test_samples': len(all_predictions)
        }
        
        self.logger.info(f"Test evaluation completed - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
        return results
    
    def run_classification_pipeline(self, data_dir: str, subject_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete classification pipeline
        
        Args:
            data_dir: Directory containing sensor data
            subject_ids: List of subject IDs to process
            
        Returns:
            Complete classification results
        """
        self.logger.info("Starting classification-only pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1/3: Loading and preprocessing data...")
            train_loader, val_loader, test_loader = self.load_and_preprocess_data(data_dir, subject_ids)
            
            # Step 2: Train classifier
            self.logger.info("Step 2/3: Training action classifier...")
            classifier_history = self.train_classifier(train_loader, val_loader)
            
            # Step 3: Evaluate on test set
            self.logger.info("Step 3/3: Evaluating on test set...")
            test_results = self.evaluate_on_test_set(test_loader)
            
            # Compile results
            results = {
                'classifier_history': classifier_history,
                'test_results': test_results,
                'data_info': {
                    'n_classes': len(np.unique(train_loader.dataset.labels)),
                    'n_channels': train_loader.dataset.data.shape[1],
                    'window_size': self.config['window_size']
                },
                'model_info': self.classifier.get_model_info() if self.classifier else None
            }
            
            # Save complete results
            save_path = os.path.join('output', 'classification_results.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(results, save_path)
            
            self.logger.info(f"Classification pipeline completed successfully! Results saved to {save_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Classification pipeline failed: {str(e)}")
            raise
    
    def plot_training_history(self, history: Dict[str, List[float]], save_path: str):
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
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


def create_classification_config() -> Dict[str, Any]:
    """Create configuration for classification-only pipeline"""
    return {
        'window_size': 100,
        'batch_size': 32,
        'classifier_lr': 0.001,
        'classifier_epochs': 100,
        'classifier_dropout': 0.3
    }


def main():
    """Main function for command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Classification-Only Pipeline')
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
        config = create_classification_config()
    
    # Initialize pipeline
    pipeline = ClassificationOnlyPipeline(config)
    
    try:
        # Run classification pipeline
        results = pipeline.run_classification_pipeline(args.data_dir, args.subject_ids)
        
        # Plot training history
        if 'classifier_history' in results:
            pipeline.plot_training_history(
                results['classifier_history'], 
                os.path.join(args.output_dir, 'classification_training.png')
            )
        
        print(f"\n🎉 Classification pipeline completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Best classifier saved to: models/classification/best_classifier.pth")
        print(f"Test accuracy: {results['test_results']['test_accuracy']:.2f}%")
        
    except Exception as e:
        print(f"\n Classification pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
