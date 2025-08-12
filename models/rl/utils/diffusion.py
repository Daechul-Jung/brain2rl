import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Any, Optional, Tuple, Union

class DiffusionPolicy(nn.Module):
    """Diffusion-based policy network for RL agents"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_timesteps=1000):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        
        # Noise schedule (linear β schedule)
        self.betas = self._linear_beta_schedule()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # U-Net style architecture for noise prediction
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Main U-Net blocks
        self.down1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
        self.down2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.mid = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.up2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2)
        )
        
        self.up1 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim * 2)  # mean + log_std
        )
        
    def _linear_beta_schedule(self):
        """Linear noise schedule"""
        scale = 1000 / self.num_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, self.num_timesteps)
    
    def forward(self, x, state, t):
        """Forward pass: predict noise given noisy action, state, and timestep"""
        # x: noisy action, state: environment state, t: timestep
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1).float())
        
        # State embedding
        s_emb = self.state_embed(state)
        
        # Combine embeddings
        h = torch.cat([s_emb, t_emb], dim=-1)
        
        # U-Net architecture
        h1 = self.down1(h)
        h2 = self.down2(h1)
        h_mid = self.mid(h2)
        h_up2 = self.up2(h_mid)
        h_up1 = self.up1(h_up2)
        
        return h_up1
    
    def sample_action(self, state, num_steps=50):
        """Sample action using diffusion process"""
        batch_size = state.shape[0]
        device = state.device
        
        # Start from pure noise
        x = torch.randn(batch_size, self.action_dim, device=device)
        
        # Reverse diffusion process
        for i in reversed(range(0, num_steps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            
            # Predict noise
            predicted_noise = self.forward(x, state, t)
            
            # Denoising step
            alpha_t = self.alphas_cumprod[i]
            alpha_t_prev = self.alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0)
            
            beta_t = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise
        
        # Return mean and log_std for action distribution
        mean = x[:, :self.action_dim]
        log_std = x[:, self.action_dim:]
        
        return mean, log_std
    
    def get_action(self, state, training=True):
        """Get action from diffusion policy"""
        if training:
            # During training, use the full diffusion process
            mean, log_std = self.sample_action(state, num_steps=self.num_timesteps)
        else:
            # During inference, use fewer steps for efficiency
            mean, log_std = self.sample_action(state, num_steps=20)
        
        # Create action distribution
        std = torch.exp(torch.clamp(log_std, -20, 2))
        dist = torch.distributions.Normal(mean, std)
        
        if training:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            action = mean
            log_prob = torch.zeros(action.shape[0])
        
        # Bound actions to [-1, 1]
        action = torch.tanh(action)
        
        return action, log_prob, mean, log_std