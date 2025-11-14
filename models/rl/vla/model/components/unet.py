from typing import Tuple

import torch
import torch.nn as nn
import os, sys
default_init = torch.nn.init.xavier_uniform_
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.vit_encoder import DynamicGroupNorm


@torch.jit
def mish(x):
    return x * torch.tanh(torch.nn.Softplus(x))


def unet_squaredcos_cap_v2(timesteps, s = 0.008):
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = torch.cos((t+s) / (1+s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor):
        """
        x: tensors of arbitrary shape (..., ) I broadcast over a new last dim
        returns (..., 2 * (features //2 ))
        """
        half_features = self.features//2
        div_term = torch.log(10000) / (half_features-1)
        freq = torch.exp(torch.arange(half_features, device=x.device)* -div_term)
        args = x[..., None] * freq
        embedding = torch.concat((torch.sin(args), torch.cos(args)), dim = -1)
        return embedding
    

class DownSample1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features= features
        self.conv = nn.Conv1d(in_channels=None, out_channels=None, kernel_size=3, stride=1, padding = 0)
        
    def _lazy_build(self, C_in: int, features: int):
        self.conv = nn.Conv1d(C_in, features, kernel_size=3, stride =2, padding = 1)

    def forward(self, x: torch.Tensor):
        if self.conv.in_channels is None:
            self._lazy_build(x.shape[1], features = self.features)
        return self.conv(x)


class UpSample1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features
        self.deconv = nn.ConvTranspose1d(in_channels = None, out_channels=None, kernel_size=4, stride = 2, padding = 0)
    
    def _lazy_build(self, C_in: int, features: int):
        self.deconv = nn.ConvTranspose1d(in_channels=C_in, out_channels=features, kernel_size=4, stride = 2, padding = 1)

    def forward(self, x: torch.tensor):
        if self.deconv.in_channels is None:
            self._lazy_build(x.shape[1], self.features)

        return self.deconv(x)
    

class Conv1dBlock(nn.Module):
    """
    Conv1d -> GroupNorm --> Mish
    """

    def __init__(self, features: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.features = features
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.conv = nn.Conv1d(in_channels=None, out_channels=None, kernel_size=kernel_size)
        self.gn = DynamicGroupNorm(num_channels=None)

    def _lazy_build(self, C_in: int, features: int):
        self.conv = nn.Conv1d(in_channels=C_in, out_channels=features, kernel_size=self.kernel_size, stride = 1, padding = self.kernel_size//2)
        self.gn = DynamicGroupNorm(C_in, self.n_groups)

    def forward(self, x: torch.Tensor):
        if self.conv.in_channels is None:
            self._lazy_build(x.shape[1], self.features)
        x = self.conv(x)
        x = self.gn(x)
        x = mish(x)
        return x
    

class ConditionalResidualBlock1d(nn.Module):
    """
    Assume that X: [B, C, L]
    cond: [B, cond_dim]
    Flow: Conv1dBlock -> FiLM -> Conv1dBlock -> residual
    """
    def __init__(self, features, kernel_size: int = 3, n_groups: int = 8, residual_proj: bool= False):
        super().__init__()
        self.features = features
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.block1 = Conv1dBlock(features, kernel_size, n_groups)
        self.cond_features = features * 2
        
        

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        residual = x
