"""
Brain Tokenizer Transformer Module
==================================

This module provides the BrainTokenizer transformer-based model for brain signal tokenization
and generating tokens suitable for reinforcement learning trajectories.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn.functional as F

class BrainTokenizer(nn.Module):
    """
    Transformer-based model for brain signal tokenization
    """
    def __init__(self, 
                 input_channels: int, 
                 token_dim: int = 512, 
                 num_tokens_out: int = 512, 
                 nhead: int = 8, 
                 num_encoder_layers: int = 6, 
                 dropout: float = 0.1,
                 use_conv_frontend: bool = True,
                 conv_hidden: int = 256):
        
        super(BrainTokenizer, self).__init__()

        self.input_channels = input_channels
        self.token_dim = token_dim
        self.num_tokens_out = num_tokens_out
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dropout = dropout
        self.use_conv_frontend = use_conv_frontend
        self.conv_hidden = conv_hidden

        # Feature extraction when I do not tokenize signals in classification
        if use_conv_frontend:
            # Feature extractor for raw signals
            self.feature_extractor = nn.Sequential(
                nn.Conv1d(input_channels = input_channels, out_channels=self.conv_hidden, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm1d(self.conv_hidden), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(input_channels = 128, out_channels = self.conv_hidden, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm1d(self.conv_hidden), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1),

                nn.Conv1d(self.conv_hidden, self.token_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(self.token_dim), nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            )
            # after 3 stride-2 pools: len // 8 (approx); we still adaptively pool at runtime
            self._pool_div = 8
        else:
            ## Normally, we use only already tokenized result
            # Linear projection for precomputed tokens: (B, C=128, T) -> (B, E, T) Embedding tokens
            self.feature_extractor = nn.Conv1d(input_channels, self.token_dim, kernel_size=1, stride=1, padding=0)
            self._expects_channel_first = False

        self.max_len = 4096
        self.positional_encoding = nn.Parameter(torch.randn(1, self.max_len, self.token_dim))

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=nhead,
            dim_feedforward=4*self.token_dim,
            dropout=dropout, batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=self.num_encoder_layers)

        ### Token predictor head
        # self.token_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim *2), nn.ReLU(), nn.Dropout(dropout),
        #     nn.Linear(embedding_dim *2 , n_tokens)
        # )


    def _to_features(self, x:torch.Tensor):
        """
         Returns features as (B, token_dim, T_feat_before_pool)

        - If use_conv_frontend:
            x: (B, C_in, T_raw)
            -> feature_extractor -> (B, token_dim, T_feat)
        - Else (pre-tokenized):
            x: (B, num_tokens_in, token_dim_in)
            -> transpose to (B, token_dim_in, num_tokens_in) -> 1x1 conv -> (B, token_dim, num_tokens_in)
        """
        if self._expects_channel_first:
            feature = self.feature_extractor(x)

        else:
            feature = self.feature_extractor(x.transpose(1,2)) ## (B, token_dim, num_token)

        return feature

    def forward(self, x:torch.Tensor):
        """
        Return contextualized tokens: (B, num_tokens_out, token_dim)
        """
        feature = self._to_features(x)

        ## Pool along time to a fixed num_token_out
        feature = F.adaptive_avg_pool1d(feature, self.num_tokens_out)

        ## Prepare for transformer: transpose to (B, num_tokens_out, token_dim)
        tok = feature.transpose(1, 2) ## 

        ## Add Positional encoding and encode
        pos = self.positional_encoding[:, :self.num_tokens_out, :] ## (1, num_tokens_out, token_dim)
        enc = self.transformer_encoder(tok + pos)

        return enc ## (B, num_token_out, token_dim)
    
    def encode(self, x:torch.Tensor):
        """
        Alias for forward: return (B, num_tokens_out, token_dim)
        """
        return self.forward(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'use_conv_frontend': self.use_conv_frontend,
            'input_channels': self.input_channels,
            'token_dim': self.token_dim,
            'num_tokens_out': self.num_tokens_out,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'conv_hidden': self.conv_hidden,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }