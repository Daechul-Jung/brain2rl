"""
Brain Tokenizer Transformer Module
==================================

This module provides the BrainTokenizer transformer-based model for brain signal tokenization
and generating tokens suitable for reinforcement learning trajectories.
"""

import torch
import torch.nn as nn
from typing import Dict, Any


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
    
    def get_model_info(self):
        """Get information about the model architecture"""
        return {
            'input_channels': self.feature_extractor[0].in_channels,
            'input_length': self.output_length * 8,  # Approximate original length
            'output_length': self.output_length,
            'embedding_dim': self.positional_encoding.size(-1),
            'n_tokens': self.token_predictor[-1].out_features,
            'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
            'num_encoder_layers': len(self.transformer_encoder.layers),
            'dropout': self.transformer_encoder.layers[0].dropout.p,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
    
    def get_attention_weights(self, x):
        """Get attention weights for analysis (requires modification of transformer)"""
        # This would require modifying the transformer to return attention weights
        # For now, just return the encoded features
        features = self.feature_extractor(x)
        features = features.transpose(1, 2)
        
        seq_len = features.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :]
        features = features + pos_encoding
        
        # Get encoded features
        encoded = self.transformer_encoder(features)
        
        return {
            'features': features,
            'encoded': encoded,
            'positional_encoding': pos_encoding
        }
    
    def generate_tokens_with_attention(self, x, return_attention: bool = False):
        """Generate tokens with optional attention information"""
        if return_attention:
            attention_info = self.get_attention_weights(x)
            tokens = self.token_predictor(attention_info['encoded'])
            return tokens, attention_info
        else:
            return self.forward(x)
