"""
Tokenization Pipeline
=====================

This module handles the tokenization of classified sensor data into tokens
with Query/Key/Value matrices for reinforcement learning trajectory control.

Author: Daechul Jung
Version: 1.0.0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.tokenization.brain_tokenizer import BrainTokenizer


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for Q/K/V matrix generation"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, _ = query.size()
        
        # Linear transformations
        Q = self.W_q(query)  # (batch_size, seq_len, d_model)
        K = self.W_k(key)    # (batch_size, seq_len, d_model)
        V = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch_size, n_heads, seq_len, seq_len)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len, seq_len)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)  # (batch_size, n_heads, seq_len, head_dim)
        
        # Reshape and combine heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)
        
        # Final linear transformation
        output = self.W_o(context)  # (batch_size, seq_len, d_model)
        
        # Return output and attention components
        attention_info = {
            'query': Q,
            'key': K,
            'value': V,
            'attention_weights': attention_weights,
            'scores': scores
        }
        
        return output, attention_info


class TokenizationDataset(Dataset):
    """Dataset for tokenization pipeline"""
    
    def __init__(self, classified_data: np.ndarray, action_labels: np.ndarray, 
                 sequence_length: int = 100, stride: int = 50):
        self.classified_data = classified_data
        self.action_labels = action_labels
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate number of sequences
        self.n_sequences = (len(classified_data) - sequence_length) // stride + 1
        
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        start_idx = idx * self.stride
        end_idx = start_idx + self.sequence_length
        
        sequence_data = self.classified_data[start_idx:end_idx]
        sequence_labels = self.action_labels[start_idx:end_idx]
        
        return torch.FloatTensor(sequence_data), torch.LongTensor(sequence_labels)


class TokenizationPipeline:
    """
    Pipeline for tokenizing classified sensor data with Q/K/V matrices
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the tokenization pipeline
        
        Args:
            model_config: Configuration for the tokenization model
        """
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logger
        self.logger = logging.getLogger('Brain2RL.Tokenization')
        
        # Model components
        self.tokenizer_model = None
        self.attention_model = None
        self.optimizer = None
        self.criterion = None
        
        # Training state
        self.is_trained = False
        self.training_history = []
        
        self.logger.info("Tokenization pipeline initialized")
    
    def initialize_models(self, input_dim: int, n_classes: int):
        """
        Initialize the tokenization models
        
        Args:
            input_dim: Input dimension (number of features)
            n_classes: Number of action classes
        """
        # Initialize tokenizer
        self.tokenizer_model = BrainTokenizer(
            input_channels=input_dim,
            input_length=self.model_config['max_sequence_length'],
            n_tokens=self.model_config['n_tokens'],
            embedding_dim=self.model_config['embedding_dim'],
            nhead=self.model_config['nhead'],
            num_encoder_layers=self.model_config['num_encoder_layers'],
            dropout=self.model_config['dropout']
        ).to(self.device)
        
        # Initialize attention model for Q/K/V generation
        self.attention_model = MultiHeadAttention(
            d_model=self.model_config['embedding_dim'],
            n_heads=self.model_config['nhead'],
            dropout=self.model_config['dropout']
        ).to(self.device)
        
        # Optimizer and criterion
        self.optimizer = torch.optim.Adam(
            list(self.tokenizer_model.parameters()) + list(self.attention_model.parameters()),
            lr=self.model_config.get('learning_rate', 0.001)
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.logger.info(f"Models initialized with input dim {input_dim}, embedding dim {self.model_config['embedding_dim']}")
    
    def prepare_data(self, classified_data: np.ndarray, action_labels: np.ndarray) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for tokenization training
        
        Args:
            classified_data: Classified sensor data
            action_labels: Action labels
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Split data into train/validation
        split_idx = int(0.8 * len(classified_data))
        
        train_data = classified_data[:split_idx]
        train_labels = action_labels[:split_idx]
        
        val_data = classified_data[split_idx:]
        val_labels = action_labels[split_idx:]
        
        # Create datasets
        train_dataset = TokenizationDataset(
            train_data, train_labels,
            sequence_length=self.model_config['max_sequence_length'],
            stride=self.model_config['max_sequence_length'] // 2
        )
        
        val_dataset = TokenizationDataset(
            val_data, val_labels,
            sequence_length=self.model_config['max_sequence_length'],
            stride=self.model_config['max_sequence_length'] // 2
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_config.get('batch_size', 32),
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.model_config.get('batch_size', 32),
            shuffle=False
        )
        
        self.logger.info(f"Data prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        return train_loader, val_loader
    
    def train_tokenizer(self, classified_data: np.ndarray, action_labels: np.ndarray, 
                       epochs: int = 100) -> Dict[str, List[float]]:
        """
        Train the tokenization model
        
        Args:
            classified_data: Classified sensor data
            action_labels: Action labels
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        # Initialize models if not done
        if self.tokenizer_model is None:
            input_dim = classified_data.shape[1] if len(classified_data.shape) > 1 else 1
            n_classes = len(np.unique(action_labels))
            self.initialize_models(input_dim, n_classes)
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(classified_data, action_labels)
        
        # Training loop
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        
        for epoch in range(epochs):
            # Training phase
            self.tokenizer_model.train()
            self.attention_model.train()
            
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Reshape data for tokenizer (batch_size, channels, sequence_length)
                if len(batch_data.shape) == 3:
                    batch_data = batch_data.transpose(1, 2)
                else:
                    batch_data = batch_data.unsqueeze(1)
                
                self.optimizer.zero_grad()
                
                # Forward pass through tokenizer
                tokens = self.tokenizer_model(batch_data)  # (batch_size, seq_len, n_tokens)
                
                # Generate Q/K/V matrices using attention
                batch_size, seq_len, n_tokens = tokens.shape
                
                # Use tokens as input to attention mechanism
                attention_output, attention_info = self.attention_model(tokens, tokens, tokens)
                
                # Token prediction task (predict next token)
                # Shift tokens for next token prediction
                input_tokens = tokens[:, :-1, :]  # (batch_size, seq_len-1, n_tokens)
                target_tokens = tokens[:, 1:, :].argmax(dim=-1)  # (batch_size, seq_len-1)
                
                # Predict next tokens
                logits = attention_output[:, :-1, :]  # (batch_size, seq_len-1, embedding_dim)
                
                # Add prediction head
                if not hasattr(self, 'prediction_head'):
                    self.prediction_head = nn.Linear(
                        self.model_config['embedding_dim'], 
                        self.model_config['n_tokens']
                    ).to(self.device)
                
                predictions = self.prediction_head(logits)  # (batch_size, seq_len-1, n_tokens)
                
                # Compute loss
                loss = self.criterion(
                    predictions.reshape(-1, self.model_config['n_tokens']),
                    target_tokens.reshape(-1)
                )
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predicted_tokens = predictions.argmax(dim=-1)
                train_correct += (predicted_tokens == target_tokens).sum().item()
                train_total += target_tokens.numel()
            
            # Validation phase
            self.tokenizer_model.eval()
            self.attention_model.eval()
            
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    # Reshape data for tokenizer
                    if len(batch_data.shape) == 3:
                        batch_data = batch_data.transpose(1, 2)
                    else:
                        batch_data = batch_data.unsqueeze(1)
                    
                    # Forward pass
                    tokens = self.tokenizer_model(batch_data)
                    attention_output, attention_info = self.attention_model(tokens, tokens, tokens)
                    
                    # Token prediction
                    input_tokens = tokens[:, :-1, :]
                    target_tokens = tokens[:, 1:, :].argmax(dim=-1)
                    
                    logits = attention_output[:, :-1, :]
                    predictions = self.prediction_head(logits)
                    
                    loss = self.criterion(
                        predictions.reshape(-1, self.model_config['n_tokens']),
                        target_tokens.reshape(-1)
                    )
                    
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    predicted_tokens = predictions.argmax(dim=-1)
                    val_correct += (predicted_tokens == target_tokens).sum().item()
                    val_total += target_tokens.numel()
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Store history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        self.is_trained = True
        self.training_history = history
        
        return history
    
    def tokenize_time_series(self, classified_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Tokenize time series data and generate Q/K/V matrices
        
        Args:
            classified_data: Classified sensor data
            
        Returns:
            Dictionary containing tokens and Q/K/V matrices
        """
        if not self.is_trained and self.tokenizer_model is None:
            raise ValueError("Tokenizer must be trained or loaded before tokenization")
        
        # Prepare data
        dataset = TokenizationDataset(
            classified_data, 
            np.zeros(len(classified_data)),  # Dummy labels
            sequence_length=self.model_config['max_sequence_length'],
            stride=self.model_config['max_sequence_length'] // 2
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.model_config.get('batch_size', 32),
            shuffle=False
        )
        
        # Tokenization
        self.tokenizer_model.eval()
        self.attention_model.eval()
        
        all_tokens = []
        all_queries = []
        all_keys = []
        all_values = []
        all_attention_weights = []
        
        with torch.no_grad():
            for batch_data, _ in tqdm(dataloader, desc="Tokenizing"):
                batch_data = batch_data.to(self.device)
                
                # Reshape data for tokenizer
                if len(batch_data.shape) == 3:
                    batch_data = batch_data.transpose(1, 2)
                else:
                    batch_data = batch_data.unsqueeze(1)
                
                # Generate tokens
                tokens = self.tokenizer_model(batch_data)  # (batch_size, seq_len, n_tokens)
                
                # Generate Q/K/V matrices
                attention_output, attention_info = self.attention_model(tokens, tokens, tokens)
                
                # Store results
                all_tokens.append(tokens.cpu().numpy())
                all_queries.append(attention_info['query'].cpu().numpy())
                all_keys.append(attention_info['key'].cpu().numpy())
                all_values.append(attention_info['value'].cpu().numpy())
                all_attention_weights.append(attention_info['attention_weights'].cpu().numpy())
        
        # Concatenate results
        result = {
            'tokens': np.concatenate(all_tokens, axis=0),
            'queries': np.concatenate(all_queries, axis=0),
            'keys': np.concatenate(all_keys, axis=0),
            'values': np.concatenate(all_values, axis=0),
            'attention_weights': np.concatenate(all_attention_weights, axis=0)
        }
        
        self.logger.info(f"Tokenization complete. Generated {result['tokens'].shape[0]} token sequences")
        return result
    
    def save_models(self, save_path: str):
        """Save the tokenization models"""
        save_data = {
            'tokenizer_state_dict': self.tokenizer_model.state_dict(),
            'attention_state_dict': self.attention_model.state_dict(),
            'model_config': self.model_config,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        if hasattr(self, 'prediction_head'):
            save_data['prediction_head_state_dict'] = self.prediction_head.state_dict()
        
        torch.save(save_data, save_path)
        self.logger.info(f"Models saved to {save_path}")
    
    def load_models(self, load_path: str):
        """Load pre-trained tokenization models"""
        save_data = torch.load(load_path, map_location=self.device)
        
        self.model_config = save_data['model_config']
        self.training_history = save_data['training_history']
        self.is_trained = save_data['is_trained']
        
        # Initialize models
        if self.tokenizer_model is None:
            # We need to determine the input dimensions from the saved config
            input_dim = self.model_config.get('input_dim', 1)
            n_classes = self.model_config.get('n_classes', 6)
            self.initialize_models(input_dim, n_classes)
        
        # Load state dictionaries
        self.tokenizer_model.load_state_dict(save_data['tokenizer_state_dict'])
        self.attention_model.load_state_dict(save_data['attention_state_dict'])
        
        if 'prediction_head_state_dict' in save_data:
            self.prediction_head = nn.Linear(
                self.model_config['embedding_dim'],
                self.model_config['n_tokens']
            ).to(self.device)
            self.prediction_head.load_state_dict(save_data['prediction_head_state_dict'])
        
        self.logger.info(f"Models loaded from {load_path}")
    
    def visualize_attention(self, tokens: np.ndarray, attention_weights: np.ndarray, 
                          sample_idx: int = 0) -> None:
        """
        Visualize attention patterns
        
        Args:
            tokens: Token sequences
            attention_weights: Attention weight matrices
            sample_idx: Index of sample to visualize
        """
        import matplotlib.pyplot as plt
        
        # Select sample
        sample_attention = attention_weights[sample_idx]  # (n_heads, seq_len, seq_len)
        
        # Average across heads
        avg_attention = np.mean(sample_attention, axis=0)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.imshow(avg_attention, cmap='Blues', aspect='auto')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title(f'Attention Pattern - Sample {sample_idx}')
        plt.tight_layout()
        plt.show()
        
        self.logger.info(f"Attention visualization displayed for sample {sample_idx}")


def main():
    """Main function for standalone tokenization pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tokenization Pipeline')
    parser.add_argument('--classified-data', type=str, required=True, help='Path to classified data')
    parser.add_argument('--mode', choices=['train', 'tokenize'], default='train')
    parser.add_argument('--model-path', type=str, help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--n-tokens', type=int, default=512, help='Number of tokens')
    
    args = parser.parse_args()
    
    # Create model config
    model_config = {
        'embedding_dim': args.embedding_dim,
        'n_tokens': args.n_tokens,
        'nhead': 8,
        'num_encoder_layers': 6,
        'dropout': 0.1,
        'max_sequence_length': 1000,
        'batch_size': args.batch_size,
        'learning_rate': 0.001
    }
    
    # Initialize pipeline
    pipeline = TokenizationPipeline(model_config)
    
    # Load data
    data = np.load(args.classified_data)
    if isinstance(data, np.lib.npyio.NpzFile):
        classified_data = data['predictions']
        action_labels = data['confidences']  # Use confidences as labels for now
    else:
        classified_data = data
        action_labels = np.zeros(len(classified_data))
    
    if args.mode == 'train':
        # Train tokenizer
        history = pipeline.train_tokenizer(classified_data, action_labels, epochs=args.epochs)
        
        if args.model_path:
            pipeline.save_models(args.model_path)
        
        print(f"Training completed. Final validation accuracy: {history['val_acc'][-1]:.2f}%")
        
    elif args.mode == 'tokenize':
        # Load model
        if args.model_path:
            pipeline.load_models(args.model_path)
        token_data = pipeline.tokenize_time_series(classified_data)
        
        np.savez('tokenization_results.npz', **token_data)
        print(f"Tokenization completed. Generated {token_data['tokens'].shape[0]} token sequences")


if __name__ == '__main__':
    main() 