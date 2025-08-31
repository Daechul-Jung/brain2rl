"""
Main Pipeline Orchestrator
==========================

This module orchestrates the complete Brain2RL pipeline using the separated components:
1. Classification: Sensor data -> Action classification
2. Tokenization: Classified data -> Tokens with Q/K/V matrices
3. RL Integration: Tokens -> RL-ready states

Author: Brain2RL Team
Version: 2.0.0
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import separated components
from models.classification.time_series_dataset import TimeSeriesDataset
from models.classification.action_classifier_cnn import ActionClassifier
from models.classification.data_utilities import (
    load_sensor_data, preprocess_data, create_dataloaders, 
    validate_data_format, get_data_info, save_preprocessing_info
)
from models.tokenization.brain_tokenizer_transformer import BrainTokenizer
from models.rl.withsignal.token_based_rl_state import TokenBasedRLState


class Brain2RLPipeline:
    """
    Main pipeline that orchestrates the complete Brain2RL workflow
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the main pipeline
        
        Args:
            config: Configuration dictionary with pipeline settings
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Initialize models
        self.classifier = None
        self.tokenizer = None
        
        # Data preprocessing
        self.scaler = None
        self.label_encoder = None
        
        # Training components
        self.classifier_optimizer = None
        self.tokenizer_optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Brain2RL Pipeline initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('Brain2RLPipeline')
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
            
            for inputs, _ in train_loader:
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
            for inputs, labels in dataloader:
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
    
    def create_rl_states(self, token_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Create RL-ready states from generated tokens
        
        Args:
            token_data: Token data from the pipeline
            
        Returns:
            Dictionary with RL states and metadata
        """
        self.logger.info("Creating RL states from tokens...")
        
        # Initialize RL state manager
        rl_state_manager = TokenBasedRLState(token_data)
        
        # Create different types of RL states
        ppo_states = rl_state_manager.create_ppo_compatible_states()
        sac_states = rl_state_manager.create_sac_compatible_states()
        hierarchical_states = rl_state_manager.create_hierarchical_states()
        
        # Save RL states
        save_path = os.path.join('output', 'rl_states.npz')
        rl_state_manager.save_rl_states(save_path)
        
        rl_data = {
            'ppo_states': ppo_states,
            'sac_states': sac_states,
            'hierarchical_states': hierarchical_states,
            'statistics': rl_state_manager.get_state_statistics()
        }
        
        self.logger.info(f"RL states created and saved to {save_path}")
        return rl_data
    
    def run_full_pipeline(self, data_dir: str, subject_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the complete pipeline from data loading to RL state creation
        
        Args:
            data_dir: Directory containing sensor data
            subject_ids: List of subject IDs to process
            
        Returns:
            Complete pipeline results
        """
        self.logger.info("Starting full Brain2RL pipeline...")
        
        try:
            # Step 1: Load and preprocess data
            self.logger.info("Step 1/5: Loading and preprocessing data...")
            train_loader, val_loader, test_loader = self.load_and_preprocess_data(data_dir, subject_ids)
            
            # Step 2: Train classifier
            self.logger.info("Step 2/5: Training action classifier...")
            classifier_history = self.train_classifier(train_loader, val_loader)
            
            # Step 3: Train tokenizer
            self.logger.info("Step 3/5: Training brain tokenizer...")
            tokenizer_history = self.train_tokenizer(train_loader, val_loader)
            
            # Step 4: Generate tokens
            self.logger.info("Step 4/5: Generating tokens from test data...")
            token_data = self.generate_tokens(test_loader)
            
            # Step 5: Create RL states
            self.logger.info("Step 5/5: Creating RL states...")
            rl_states = self.create_rl_states(token_data)
            
            # Compile results
            results = {
                'classifier_history': classifier_history,
                'tokenizer_history': tokenizer_history,
                'token_data': token_data,
                'rl_states': rl_states,
                'data_info': {
                    'n_classes': len(np.unique(train_loader.dataset.labels)),
                    'n_channels': train_loader.dataset.data.shape[1],
                    'window_size': self.config['window_size']
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
    """Create default configuration for the pipeline"""
    return {
        'window_size': 100,
        'batch_size': 32,
        'classifier_lr': 0.001,
        'classifier_epochs': 100,
        'classifier_dropout': 0.3,
        'tokenizer_lr': 0.0001,
        'tokenizer_epochs': 50,
        'tokenizer_dropout': 0.1,
        'n_tokens': 512,
        'embedding_dim': 128,
        'nhead': 8,
        'num_encoder_layers': 6
    }


def main():
    """Main function for command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain2RL Pipeline')
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
    pipeline = Brain2RLPipeline(config)
    
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
