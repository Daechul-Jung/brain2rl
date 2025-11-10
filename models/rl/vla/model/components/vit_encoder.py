"""
Encoders more suitable for ViT architectures.

- PatchEncoder: Just patchifies the image
- SmallStem: 3 conv layers, then patchifies the image (from xiao et al. 2021)
- ViTResnet: ResNetv2, followed by patchification (from google-research/vision_transformer)
"""
import os, sys
import functools as ft
from typing import Callable, Sequence, TypeVar
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.film_conditioning_layer import FilmConditioning

T = TypeVar('T')

def normalize_image(img, img_norm_type = 'default'):
    if img_norm_type == 'default':
        return img.astype(np.float32) / 127.5 - 1.0
    
    elif img_norm_type == 'imagenet':
        img = img.astype(np.float32) / 255
        assert img.shape[-1] % 3 == 0, 'Image should have rgb channel'

        # define pixel-wise mean/std stats calculated from ImageNet
        mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3))
        std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3))
        
        # tile mean and std
        num_tile = (1, 1, 1, int(img.shape[-1]/3))
        mean_tile = torch.tile(mean, num_tile)
        std_tile = torch.tile(std, num_tile)

        return (img - mean_tile) / std_tile
    raise ValueError

def weight_standardize(w, axis, eps):
    """
    Subtract mean and divides by standard deviation
    """
    w = w - np.mean(w, axis=axis)
    w = w / (np.std(w, axis=axis) + eps)
    return w

class StdConv2d(nn.Conv2d):
    """
    Conv2d with weight standardization (per out-channel)
    """
    def __init__(self, *args, eps: float = 1e-5, axis = (1,2,3), **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.axis = axis

    def forward(self, x: torch.Tensor):
        w_norm = weight_standardize(self.weight)
        return F.conv2d(
            x, w_norm, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

class PatchEncoder(nn.Module):
    """
    Takes an image and breaks it up into patches of size (patch_size x patch_size),
    applying a fully connected network to each patch individually.

    The default 'encoder' used by most ViTs in practice
    """

    def __init__(self, use_film: bool = False, patch_size: int = 32, num_features: int = 512, img_norm_type: str = 'default', cond_dim = None, use_weight_standardized_conv: bool = False, in_channels: int = 3):
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.num_features = 512
        self.img_nomr_type = img_norm_type
        Conv = StdConv2d if use_weight_standardized_conv else nn.Conv2d

        self.embedding = Conv(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        if use_film:
            assert cond_dim is not None, "cond_dim must be provided when use_film=True"
            self.film = FilmConditioning(cond_dim=cond_dim, channels=num_features, data_format='NCHW')
        else:
            self.film = None

    def forward(self, observation: torch.Tensor, train: bool = True, cond_var = None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None

        assert (
            expecting_cond_var == received_cond_var
        ), 'Only pass in cond var iff model expecting cond var'
        
        x = normalize_image(observation, self.img_nomr_type)
        x = self.embedding(x)

        if self.use_film:
            x = self.film(x, cond_var)

        return x
    

class DynamicGroupNorm(nn.Module):
    """
    Flax's Groupnorm auto-picks groups; Pytorch needs num_groups: C
    This module picks the largest divisor <= max_groups (default 32)
    """    
    def __init__(self, num_channels, max_groups= 32, affine: bool = True, eps: float= 1e-5 ):
        super().__init__()
        self.affine = affine
        self.eps = eps
        self.max_groups = max_groups


class SmallStem(nn.Module):
    """
    Passes the image through a few light-weight convolutional layers,
    before patchifying the image. Empirically useful for many computers vision tasks.

    See Xiao et al: Early Convolutions Help Transformers See Better
    """
    def __init__(self, 
                 use_film: bool = False, 
                 patch_size: int = 32, 
                 kernel_size: tuple = (3,3,3,3),
                 strides: tuple = (2,2,2,2),
                 features: tuple = (32, 96, 192, 384),
                 padding: tuple = (1,1,1,1),
                 num_features: int = 512,
                 img_norm_type: str = 'default'):
        
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.features = features
        self.padding = padding
        self.num_features = num_features
        self.img_norm_type = img_norm_type



    def forward(self, observations: torch.Tensor, train: bool=True, cond_var = None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None

        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_image(observations, self.img_norm_type)
        for n, (kernel_size, stride, features, padding) in enumerate(
            zip(
                self.kernel_size,
                self.strides,
                self.features,
                self.padding
            )
        ):
            x = StdConv2d()