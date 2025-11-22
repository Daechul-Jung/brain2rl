"""
This is adapted from https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py
But in pytorch version
"""

import torch 
import torch.nn as nn
import numpy as np
from typing import Literal

class FilmConditioning(nn.Module):
    """
    FiLM: Feature-Wise Linear Modulation. Simple way to inject a conditioning signal such as text and etc into a network by scaling and shifting its intermediate features  
    Applies FiLM conditioning to a convolutional feature map. One way to condition one feature vector or tensor on another. 
    Using Linear layers for adding and multiplication part, put conv filter which is data that the model would condition based on 

    Args:
        conv_filters: A tensor of shape [batch_size, height, width, channels].
        conditioning: A tensor of shape [batch_size, conditioning_size].

    Returns:
        A tensor of shape [batch_size, height, width, channels].

    Example Usage:
    film = FilmConditioning(cond_dim = cond_size, )
    """
    def __init__(self, cond_dim, channels = None, data_format: Literal['NCHW', 'NHWC'] = 'NHWC'):
        super().__init__()
        self.data_format = data_format
        self.cond_dim = cond_dim
        self.channels = channels
        self.projected_cond_add = None
        self.projected_cond_mult = None

        if channels is not None:
            self._build_linear(channels)

    def _build_linear(self, channels):
        self.projected_cond_add = nn.Linear(self.cond_dim, channels, bias = True)
        self.projected_cond_mult = nn.Linear(self.cond_dim, channels, bias = True)

        nn.init.zeros_(self.projected_cond_add.weight)
        nn.init.zeros_(self.projected_cond_add.bias)
        nn.init.zeros_(self.projected_cond_mult.weight)
        nn.init.zeros_(self.projected_cond_mult.bias)


    def forward(self, conv_filter: torch.Tensor, conditioning: torch.Tensor):
        """
        conv_filters: [B, C, H, W] if NCHW, or [B, H, W, C] if NHWC -> Data that the model would condition 
        conditioning: [B, cond_dim] -> Conditioning vector 

        returns: same shape as conv_filters which is [batch_size, channels, heigth, weight] for pytorch and [batch_size, height, weight, channels] for jax 
        """

        ### build channels after getting input to know the dimensionality of it 
        if self.channels is None:
            C = conv_filter.shape[1] if self.data_format == 'NCHW' else conv_filter.shape[-1]
            self.channels = C
            self._build_linear(C)

        ## putting them into linear model 
        project_cond_add = self.projected_cond_add(conditioning) ## shape: [B, C]
        project_cond_mult = self.projected_cond_mult(conditioning) ## shape: [B, C]

        if self.data_format == 'NCHW':
            project_cond_add = project_cond_add.unsqueeze(-1).unsqueeze(-1) ## [B, C, 1, 1]
            project_cond_mult = project_cond_mult.unsqueeze(-1).unsqueeze(-1) 

        else:
            project_cond_add = project_cond_add.unsqueeze(1).unsqueeze(1) ## [B, 1, 1, C]
            project_cond_mult = project_cond_mult.unsqueeze(1).unsqueeze(1) 

        ### 
        return conv_filter * (1 + project_cond_add) + project_cond_mult