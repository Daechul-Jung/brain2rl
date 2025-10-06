import math
import os, sys
import torch
from typing import Optional, Tuple
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.schdules import cosine_beta_schedule, build_constants

class Diffusion:
    def __init__(self, T: int = 1000, device: str = 'cuda'):
        self.T = T
        self.device = device
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas, self.alpha_bar, self.sqrt_alpha_bar, self.sqrt_1mab = build_constants(self.betas)

    def q_sample(self, x0:torch.Tensor, t: torch.LongTensor, noise:Optional[torch.Tensor] = None):
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_1mab[t].view(-1, 1, 1, 1)
        return sqrt_ab_t * x0 + sqrt_1mab * noise
    
    @torch.no_grad()
    def _predict_eps(self, model:nn.Module, x: torch.Tensor, t: torch.LongTensor,
                     cond: Optional[torch.Tensor], cfg_scale: float):
        
        if cond is None or cfg_scale == 1.0:
            return model(x, t, cond)

        eps_uncond = model(x, t, None)
        eps_cond = model(x, t, cond)
        return eps_uncond + cfg_scale(eps_cond - eps_uncond)

    @ torch.no_grad()
    def sample_ddpm(self, model: nn.Module, shape: Tuple[int, int, int, int], cond: Optional[torch.Tensor]= None,
                    cfg_scale: float = 1.0):
        x = torch.randn(shape, device = self.device)
        for i in reversed(self.T, -1, -1, -1):
            t = torch.full((shape[0], ), i, device=self.device, dtype=torch.long)
            eps = self._predict_eps(model, x, t, cond, cfg_scale)
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.betas[i]

            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)

            if i > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)

            else :
                x = mean
        return x

    @torch.no_grad()
    def sample_ddim(self, model:nn.Module, shape: Tuple[int, int, int, int], cond: Optional[torch.Tensor] = None,
                    steps: int = 50, eta:float = 0.0, cfg_scale: float = 1.0):
        T_full = self.T
        idx = torch.linspace(0, T_full - 1, steps).long().flip(0).to(self.device)
        x = torch.randn(shape, device= self.device)
        for k in range(len(idx) - 1):
            i, j = idx[k].item(), idx[k + 1].item()
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            eps = self._predict_eps(model, x, t, cond, cfg_scale)
            alpha_bar_i = self.alpha_bar[i]
            alpha_bar_j = self.alpha_bar[j]
            x0_pred = (x - torch.sqrt(1 - alpha_bar_i)* eps) /  torch.sqrt(alpha_bar_i)
            dir_term = torch.sqrt(1 - alpha_bar_j) * eps
            x = torch.sqrt(alpha_bar_j) * x0_pred + dir_term
            if eta > 0 and j > 0:
                sigma = eta * math.sqrt((1- alpha_bar_j) / (1- alpha_bar_i)) * math.sqrt(1 - alpha_bar_i/ alpha_bar_j)
                x = x + sigma * torch.randn_like(x)

        return x

    