import math
import torch
from typing import Tuple


def cosine_beta_schedule(T: int, s: float = 0.008):
    steps = torch.arange(T + 1, dtype = torch.float32) ### creating each step based on T
    f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1: ] / alpha_bar[: -1])
    return betas.clamp(1e-6, 0.999)

def build_constants(betas: torch.Tensor):
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim = 0) ## torch.cumprod = returns cumulative product of elements of input in the dimension dim