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
    def __init__(self, 
                 input_channels: int, 
                 input_length: int, 
                 n_tokens: int = 512, 
                 embedding_dim: int = 128, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 dropout: float = 0.1,
                 use_conv_frontend: bool = True):
        
        super(BrainTokenizer, self).__init__()

        self.input_channels = input_channels
        self.input_length = input_length
        self.n_tokens = n_tokens
        self.embedding_dim = embedding_dim
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.use_conv_frontend = use_conv_frontend

        # Feature extraction
        if use_conv_frontend:
            # Feature extractor for raw signals
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(input_channels = input_channels, out_channels=128, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(input_channels = 128, out_channels = 128, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(embedding_dim), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
            # after 3 stride-2 pools: len // 8 (approx); we still adaptively pool at runtime
            self._pool_div = 8
        else:
            # Linear projection for precomputed tokens: (B, C=128, T) -> (B, E, T)
            self.feature_extractor = nn.Conv1d(input_channels, embedding_dim, kernel_size=1, stride=1, padding=0)
            self._pool_div = 1

        self.max_len = 4096
        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_len, embedding_dim))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim, nhead=nhead,
            dim_feedforward=4*embedding_dim,
            dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=self.num_encoder_layers)

        ### Token predictor head
        self.token_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim *2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embedding_dim *2 , n_tokens)
        )


    def forward(self, x):
        """
        x : (B, C, T)
        """
        feat = self.feature_extractor(x)
        Tenc = min(feat.size(-1), self.max_len)
        feat = nn.functional.adaptive_avg_pool1d(feat, Tenc)
        feat = feat.transpose(1, 2)

        pos = self.positional_encoding[:, :Tenc, :]
        feat = feat + pos
        enc = self.transformer_encoder(feat)
        tokens = self.token_predictor(enc)
        return tokens 
    

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns contextualized embeddings (before predictor): (B, T_enc, E)
        """
        feat = self.feature_extractor(x)
        Tenc = min(feat.size(-1), self.max_len)
        feat = nn.functional.adaptive_avg_pool1d(feat, Tenc)
        feat = feat.transpose(1, 2)
        enc = self.transformer_encoder(feat + self.positional_encoding[:, :Tenc, :])
        return enc  # (B, Tenc, E)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            'input_channels': self.input_channels,
            'input_length': self.input_length,
            'embedding_dim': self.embedding_dim,
            'n_tokens': self.token_predictor[-1].out_features,
            'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
            'num_encoder_layers': len(self.transformer_encoder.layers),
            'use_conv_frontend': self.use_conv_frontend,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }

