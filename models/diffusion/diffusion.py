import math
import os, sys
import torch
from typing import Optional, Tuple
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.schdules import cosine_beta_schedule, build_constants

class Diffusion:
    """
    DDPM/DDIM utilities for action diffusion

    Treat X as set of actions over a short time horizon 
    """
    def __init__(self, T: int = 1000, device: str = 'cuda'):
        self.T = T
        self.device = device
        self.betas = cosine_beta_schedule(self.T).to(device)
        self.alphas, self.alpha_bar, self.sqrt_alpha_bar, self.sqrt_1mab = build_constants(self.betas)

    def q_sample(self, x0:torch.Tensor, t: torch.LongTensor, noise:Optional[torch.Tensor] = None):
        """
        x0: (B, H=time_horizon, action_dim), t: (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_ab_t = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_1mab = self.sqrt_1mab[t].view(-1, 1, 1)
        return sqrt_ab_t * x0 + sqrt_1mab * noise
    
    @torch.no_grad()
    def _predict_eps(self, model:nn.Module, x: torch.Tensor, t: torch.LongTensor,
                     cond_vec: Optional[torch.Tensor], cfg_scale: float):
        ### Standard classifier-free guidance (linear blend)
        if cond_vec is None or (cfg_scale == 1.0):
            return model(x, t, cond_vec)

        eps_uncond = model(x, t, None)
        eps_cond = model(x, t, cond_vec)
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    @ torch.no_grad()
    def sample_ddpm(self, model: nn.Module, shape: Tuple[int, int, int], 
                    cond_vec: Optional[torch.Tensor]= None, cfg_scale: float = 1.0):
        """
        Non-differentiable DDPM sampling for evaluation
        shape: (B, H, A) H=time_horizon, A=action_dim
        """
        B, H, A = shape 
        x = torch.randn((B, H, A), device = self.device)
        for i in reversed(self.T, -1, -1, -1):
            t = torch.full((shape[0], ), i, device=self.device, dtype=torch.long)
            eps = self._predict_eps(model, x, t, cond_vec, cfg_scale)
            alpha_t = self.alphas[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.betas[i]

            mean = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * eps)

            if i > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)

            else :
                x = mean
        return x

    def ddim_step(self, x: torch.Tensor, i:int, j:int, model: nn.Module, 
                    cond_vec: Optional[torch.Tensor] = None, eta:float = 0.0, cfg_scale: float = 1.0):
        """
        One differentiable DDIM step from i to j
        """
        B = x.size(0)
        t = torch.full((B, ), i, device=self.device, dtype=torch.long)
        eps = self._predict_eps(model, x, t, cond_vec, cfg_scale)
        alpha_bar_i = self.alpha_bar[i]
        alpha_bar_j = self.alpha_bar[j]

        x0_pred = (x - torch.sqrt(1 - alpha_bar_i) * eps) / torch.sqrt(alpha_bar_i)

        dir_term = torch.sqrt(1 - alpha_bar_j) * eps

        x_next = torch.sqrt(alpha_bar_j) * x0_pred + dir_term
        if eta >0 and j > 0:
            sigma = eta * math.sqrt((1 - alpha_bar_j) / (1 - alpha_bar_i)) * math.sqrt(1 - alpha_bar_i / alpha_bar_j)
            x_next = x_next + sigma * torch.randn_like(x)

        return x_next
    
    def sample_ddim_differentiable(self, model: nn.Module, shape: Tuple[int, int, int], 
                                   cond_vec: Optional[torch.Tensor] = None, 
                                   steps: int = 8, eta:float = 0.0 , cfg_scale: float = 1.0):
        """
        Differentiable DDIM(small number of steps) so I can backprop RL loss into the denoiser
        """

        B, H, A = shape 
        x = torch.randn((B, H, A), device = self.device)
        idx= torch.linspace(0, self.T - 1, steps).long().flip(0).to(self.device)
        for k in range(len(idx) - 1):
            i,j = idx[k].item(), idx[k + 1].item()
            x = self.ddim_step(x, i, j, model, cond_vec, eta=eta, cfg_scale=cfg_scale)
        return x


    