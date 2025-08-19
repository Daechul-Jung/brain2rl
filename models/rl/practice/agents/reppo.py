import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.optim as optim
import copy
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.fcnn import *

"""
This algorithm is Relative Entropy Pairwise Policy Optimization (RePPO)
This is on-policy and actor-critic alogrithm and train robust surrogate value function and effectively use pairwise policy gradient.
Build Maximum entropy framework and combines with a principled KL regularization, 
Categorical Q-Learning, Appropriately normalized neural network architecture and auxilary task
Show how a joint entropy and policy deviation tuning objective can address the twin problems of sufficient exploration and controlled policy update 
According to the paper, it does not utilize the replay buffer
"""

@dataclass(slots=True)
class TrainState:
    device: torch.device
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actor: Actor
    old_actor: Actor
    critic: Critic
    # normalizer: EmpiricalNormalization
    # critic_normalizer: EmpiricalNormalization
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    # scaler: GradScaler



class RePPOAgent:
    def __init__(self, observation_dim, action_dim, num_atoms = 101, 
                 vmin=-250, vmax=250, device='cuda',
                 lr = 3e-4, gamma = 0.99, kl_start = 0.1, entropy_start = 0.1,
                 lmbda = 0.95):
        super().__init__(observation_dim, action_dim, device="cuda")
        
        self.critic = Critic(observation_dim, action_dim, num_atoms, 
                             vmin=vmin, vmax=vmax, device=device)
        self.device = device
        # Actor(Policy) includes entropy and kl-regularization which are subject to optimization
        self.actor = Actor(observation_dim, action_dim, num_atoms, vmin=vmin, vmax=vmax, 
                           kl_start=kl_start, entropy_start=entropy_start, device=device)
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr = lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr = lr)
        
    def _actor_forward(self, observation):
        """
        Forward pass for the actor network
        
        Returns:
            pi: Action probabilities with transformed distributions
            mean: Mean of the action distribution
            log_temp: Log of entropy temperature
            log_lagrange: Log of KL regularization parameter
        """
        pi, mean, log_temp, log_lagrange = self.actor(observation)
        return pi, mean, log_temp, log_lagrange
    
    def _old_actor_forward(self, observation):
        old_pi, _, _, _ = self.old_actor(observation)
        return old_pi
    
    
    def _critic_forward(self, ):
        return 

    def update_actor(self):
        return 
    
    def update_critic(self):
        return 
    
    @torch.no_grad()
    def collect(self, env, observation, critic_observation, num_steps =10000):
        """
        On-policy rollout and return (transition tensordict, final_obs, final_critic_obs, infos)
        """
        transition = []
        info_list = []
        
        for _ in range(num_steps):
            ...
        
        return 
    
    def evaluate(self,):
        return 
    
    def load(self):
        return 
    
    def save(self, ):
        return 