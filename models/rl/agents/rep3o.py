"""
Relative Entropy Proximal Pairwise Policy Optimization (My own Idea)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch.optim as optim
import copy
from torch.amp import GradScaler
import gymnasium as gym
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.reppo_network import *
from models.rl.utils.any_utils import (compute_gve, _ensure_batch, _env_shape, _split_step_return, _to_tensor, wrap_batch_dim)



class REP3OAgent:
    """
    Relative Entropy Proximal Pairwis Optimization Agent which clip between 
    This agent does not use GradScaler and Normalizer. Maybe later can use normalerizer for better performance
    """
    def __init__(self, observation_dim, action_dim, num_atoms = 151,  ### num_atoms (51 ~ 151)
                 vmin= -2000, vmax=4000, device='cuda',
                 lr = 3e-4, gamma = 0.99, kl_start = 0.05, entropy_start = 0.05,
                 lmbda = 0.95, obs_normalizer= None , critic_obs_normalizer = None):

        self.vmin = vmin
        self.vmax = vmax
        self.gamma = gamma

        self.kl_start = kl_start
        self.entropy_start = entropy_start
        self.lmbda = lmbda

        self.entropy_target = 0.3 * action_dim   # was 0.5 * action_dim (too high)
        self.kl_bound = 0.3                      # was 0.1 (too tight)
        self.kl_target = 0.01  #### previous 0.25 and try 0.5, 0.75 

        self.num_atoms = num_atoms
        self.observation_dim = observation_dim
        self.action_dim = action_dim 
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        
        self.aux_loss_mult = 1.0
        self.max_grad_norm = 0.6
        self.actor_kl_clip_mode = "clipped"
        # Actor(Policy) includes entropy and kl-regularization which are subject to optimization
        self.actor = Actor(observation_dim=observation_dim,
                            action_dim=action_dim, 
                            hidden_dim=512,
                           kl_start=kl_start, entropy_start=entropy_start, 
                           device=self.device)
        
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        
        self.critic = Critic(observation_dim=observation_dim, 
                             action_dim=action_dim,
                             hidden_dim=512, 
                             num_atoms=num_atoms, 
                             vmin=vmin, vmax=vmax, 
                             device=self.device)
        
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr = lr)#, betas=(0.9,0.999), eps=1e-5)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr = lr)#, betas=(0.9, 0.999), eps=1e-5)
        self.scaler = GradScaler(device=self.device)
        self.observation_normalizer = obs_normalizer
        self.critic_observation_normalizer = critic_obs_normalizer


    def _actor_forward(self, observation):

        return 

    def _old_actor_forward(self, observation):

        return 
    
    def _critic_forward(self, observation, action):

        return 
    
    def update_actor(self, batch):
        """
        Update actor with the method used in PPO but still use KL and Lagrangian parameters.
        """

        return 
    
    def update_critic(self, batch):

        return 
    
    def collect(self, observation, critic_observation, num_steps):

        return 