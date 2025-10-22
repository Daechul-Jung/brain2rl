import math
import torch
from typing import Tuple

@torch.no_grad()
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor: 
    """
    cosine schedule from Nichol & Dhariwal 2021
    Returns:
        betas of shape (T, ) in range of (0, 1)
    """
    steps = torch.arange(T + 1, dtype = torch.float32) ### creating each step based on T
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2 ## cos( normalized? steps  * pi/2), first value would be nan
    f = f / f[0]
    alpha_bar = f  ### first value is nan
    betas = 1 - (alpha_bar[1: ] / alpha_bar[: -1]) 
    return betas.clamp(1e-6, 0.999)

@torch.no_grad()
def build_constants(betas: torch.Tensor):
    alphas = 1.0 - betas ## Get original alphas
    alphas_bar = torch.cumprod(alphas, dim = 0) ## torch.cumprod = returns cumulative product of elements of input in the dimension dim, So only first few elements have values and others are 0
    sqrt_alpha_bar = torch.sqrt(alphas_bar)
    sqrt_1mab = torch.sqrt(1 - alphas_bar)
    return alphas, alphas_bar, sqrt_alpha_bar, sqrt_1mab

