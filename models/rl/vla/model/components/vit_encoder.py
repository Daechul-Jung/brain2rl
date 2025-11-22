"""
Encoders more suitable for ViT architectures.

- PatchEncoder: Just patchifies the image
- SmallStem: 3 conv layers, then patchifies the image (from xiao et al. 2021)
- ViTResnet: ResNetv2, followed by patchification (from google-research/vision_transformer)
"""

import os, sys
import functools as ft
from typing import Callable, Sequence, TypeVar, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.film_conditioning_layer import FilmConditioning

T = TypeVar('T')

def normalize_image(img, img_norm_type = 'default'):
    """
    Normalize image based on the norm type of it
    """
    
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
    Conv2d layer with weight standardization (per out-channel)
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
        ## using nn.Conv2d rather than stdConv2d
        self.embedding = Conv(
            in_channels=in_channels,
            out_channels=num_features,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        ## Setting FiLMConditioning only conditional dimension exists
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
        ## Normalize image and embedding it
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
    def __init__(self, num_channels, max_groups= 32, affine: bool = True, eps: float= 1e-5):
        super().__init__()
        self.affine = affine
        self.eps = eps
        self.max_groups = max_groups
        self.gn = Optional[nn.GroupNorm] = None
        if num_channels is not None:
            self.__init_gn(num_channels)

    def __init_gn(self, C: int):
        """
        initializing group norm layer 
        """
        for g in range(min(self.max_groups, C), 0, -1):
            if C % g:
                self.gn = nn.GroupNorm(g, C, eps=self.eps, affine= self.affine)
                return 
        self.gn = nn.GroupNorm(1, C, eps=self.eps, affine = self.affine)
        
    def forward(self, x: torch.Tensor):
        if self.gn is None:
            self.__init_gn(x.shape[1])
        return self.gn(x)

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
                 features: tuple = (32, 96, 192, 384), ## each layer's output dimension
                 padding: tuple = (1,1,1,1),
                 num_features: int = 512, ## output channels of the final patchify conv for self.embedding 
                 in_channels = 3,
                 cond_dim: Optional[int] = None,
                 img_norm_type: str = 'default',
                 use_weight_standardized_conv: bool = True):
        
        super().__init__()
        self.use_film = use_film
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.strides = strides
        self.features = features
        self.padding = padding
        self.num_features = num_features
        self.img_norm_type = img_norm_type

        layers = []
        C_in = in_channels
        Conv = StdConv2d if use_weight_standardized_conv else nn.Conv2d
        ## setting layers first in the init. Originally, since flax does not have init, they do in __call__
        ## but I set layers in init
        for k, s, f, p in zip(kernel_size, strides, features, padding):
            layers.append(Conv(C_in, f, kernel_size = k, strides = s, padding = p, bias = True))
            layers.append(DynamicGroupNorm(num_channels=f))
            layers.append(nn.ReLU(inplace= True))
            C_in = f

        self.stem = nn.Sequential(*layers)
        
        k_patch = max(1, patch_size // 16) ## maybe 2
        ## Embedding on Conv2d
        self.embedding = nn.Conv2d(C_in, num_features, kernel_size=k_patch, stride= k_patch, padding = 0, bias= True)
        if use_film:
            self.film = FilmConditioning(cond_dim, num_features)
        
    def forward(self, observations: torch.Tensor, train: bool=True, cond_var = None):
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None

        assert (
            expecting_cond_var == received_cond_var
        ), "Only pass in cond var iff model expecting cond var"

        x = normalize_image(observations, self.img_norm_type)
        x = self.stem(x)
        x = self.embedding(x)
        
        if self.use_film:
            x = self.film
            
        return x
    
    
class ResidualUnit(nn.Module):
    """
    Bottleneck ResNet Block
        - 1x1 -> 3x3 (with stride) -> 1x1
        - GroupNorm + ReLU after first two convs
        - Final GroupNorm has zero-initialized scale (emulated by setting weight=0)
    """
    def __init__(self, features, strides: Sequence= (1,1), use_weight_standardized_conv: bool = True):
        super().__init__()
        self.features = features
        self.strides = strides
        
        Conv = StdConv2d if use_weight_standardized_conv else nn.Conv2d
        
        self.conv1 = Conv(in_channels=None, out_channels=None, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = Conv(in_channels=None, out_channels=None, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = Conv(in_channels=None, out_channels=None, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.gn1 = None
        self.gn2 = None 
        self.gn3 = None
        
        self.proj = Optional[nn.Conv2d]
        self.proj_gn = Optional[DynamicGroupNorm]
        self.use_wstd = use_weight_standardized_conv
        
    def _lazy_build(self, C_in: int):
        """
        building real layer lately since to build exact layers I want, I need input dimensionality
        """
        Conv = StdConv2d if self.use_wstd else nn.Conv2d
        
        self.conv1 = Conv(in_channels=C_in, out_channels=self.features, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn1 = DynamicGroupNorm(self.features)
        
        self.conv2 = Conv(in_channels=self.features, out_channels=self.features, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = DynamicGroupNorm(self.features)
        
        self.conv3 = Conv(self.features, self.features*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.gn3 = DynamicGroupNorm(self.features*4)
        
        with torch.no_grad():
            if hasattr(self.gn3.gn, 'weight') and self.gn3.gn.weight is not None:
                self.gn3.gn.weight.zero_()
                
        needs_proj = (C_in != self.features * 4) or (self.strides != (1,1))
        if needs_proj:
            self.proj = Conv(C_in, self.features * 4, kernel_size=1, stride=self.strides, padding=0, bias=False)
            self.proj_gn = DynamicGroupNorm(self.features*4)
            
    def forward(self, x: torch.Tensor):
        if self.gn1 is None:
            self._lazy_build(x.shape[1])
            
        residual = x
        if self.proj is not None:
            residual = self.proj(residual)
            residual = self.proj_gn(residual)
            
        y = self.conv1(x)
        y = self.gn1(y)
        y = F.relu(y, inplace = True)
        
        y = self.conv2(y)
        y = self.gn2(y)
        y = F.relu(y, inplace = True)
        
        y = self.conv3(y)
        y = self.gn3(y)
        ## After passing processes, combining with residual then take ReLU
        return F.relu(y + residual, inplace=True)
    

class ResNetStage(nn.Module):
    """
    A stack of ResnetUnit Blocks; first block may downsample via first stride
    """
    def __init__(self, block_size:int, n_out:int, first_stride: int, use_weight_standardized_conv: bool = True):
        self.block_size = block_size
        self.n_out = n_out
        self.first_stride = first_stride
        self.blocks = nn.ModuleList()
        self.blocks.append(ResidualUnit(n_out, self.first_stride, use_weight_standardized_conv))
        for _ in range(1, block_size):
            self.blocks.append(ResidualUnit(n_out, strides = (1,1), use_weight_standardized_conv= use_weight_standardized_conv))
        
    def forward(self, x):
        for b in self.blocks:
            x = b(x)
            
        return x
    
    
class ViTResNet(nn.Module):
    """
    Resnet-v2 architecture used in original ViT paper for hybrid (ResNet + ViT) architecture
    Mostly copied from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py
    
    There exist pre-trained parameters here: github.com/google-research/vision_transformer/
    """
    def __init__(self, 
                 use_film: bool = False, 
                 width: int = 1, 
                 num_layers: tuple = tuple(), 
                 img_norm_type: str = 'default',
                 cond_dim: Optional[int] = None,
                 in_channels: int = 3,
                 use_weight_standardized_conv: bool = True):
        super().__init__()
        self.use_film = use_film
        self.width = width
        self.num_layers = num_layers
        self.img_norm_type = img_norm_type
        
        Conv = StdConv2d if use_weight_standardized_conv else nn.Conv2d
        C = int(64 * width)
        self.conv1 = Conv(in_channels=in_channels, out_channels=C, kernel_size=7, stride=2, padding=3, bias = False)
        self.gn_root = DynamicGroupNorm(num_channels=C) 
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.stages = nn.ModuleList()
        if self.num_layers:
            block1 = ResNetStage(block_size=self.num_layers[0], n_out=C, first_stride=1, use_weight_standardized_conv=use_weight_standardized_conv)
            self.stages.append(block1)
            for i, block_size in enumerate(self.num_layers[1:], 1):
                self.stages.append(
                    ResNetStage(block_size=block_size, n_out=C*2**i, first_stride=2, use_weight_standardized_conv=use_weight_standardized_conv)
                )
                if self.use_film:
                     self.film = FilmConditioning(cond_dim, None)
                else: 
                    self.film = None
                    
            
    def forward(self, observation: torch.Tensor, cond_var = None, train: bool = True):
        """
        Normalize iamge -> conv -> groupNorm -> relu -> pooling -> (optional film conditioning)
        """
        expecting_cond_var = self.use_film
        received_cond_var = cond_var is not None
        assert (
            expecting_cond_var == received_cond_var
        ), 'Only pass in cond var iff model expecting cond var'
        
        x = normalize_image(observation, self.img_norm_type)
        x = self.conv1(x)
        x = self.gn_root(x)
        x = F.relu(x, inplace= True)
        x = self.pool(x)
        
        if len(self.num_layers) > 0:
            for i, stage in enumerate(self.stages):
                x = stage(x)
                if self.use_film and i == len(self.num_layers) - 1:
                    x = self.film(x, cond_var)
                    
        else: 
            if self.use_film:
                x = self.film(x, cond_var)
                
        return x
    
    
class SmallStem16(SmallStem):
    def __init__(self, **kwargs):
        super().__init__(patch_size=16, **kwargs)

class SmallStem32(SmallStem):
    def __init__(self, **kwargs):
        super().__init__(patch_size=32, **kwargs)

class ResNet26FILM(ViTResNet):
    def __init__(self, **kwargs):
        kwargs.setdefault("use_film", True)
        kwargs.setdefault("num_layers", (2, 2, 2, 2))
        super().__init__(**kwargs)

vit_encoder_configs = {
    "patchify-32-film": ft.partial(PatchEncoder, use_film=True, patch_size=32),
    "patchify-16-film": ft.partial(PatchEncoder, use_film=True, patch_size=16),

    "small-stem-8-film": ft.partial(
        SmallStem,
        use_film=True,
        patch_size=16,
        kernel_sizes=(3, 3, 3),
        strides=(2, 2, 2),
        features=(32, 96, 192),
        padding=(1, 1, 1),
    ),
    "small-stem-16": ft.partial(SmallStem, patch_size=16),
    "small-stem-16-film": ft.partial(SmallStem, use_film=True, patch_size=16),
    "small-stem-32-film": ft.partial(SmallStem, use_film=True, patch_size=32),

    "resnetv2-26-film": ft.partial(ViTResNet, use_film=True, num_layers=(2, 2, 2, 2)),
    "resnetv2-50-film": ft.partial(ViTResNet, use_film=True, num_layers=(3, 4, 6, 3)),
}