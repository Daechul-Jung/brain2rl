from typing import Tuple

import torch
import torch.nn as nn

default_init = torch.nn.init.xavier_uniform_

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