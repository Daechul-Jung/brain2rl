import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.optim as optim
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.diffusion import *
from models.rl.utils.NeuralNetwork import *
from models.rl.utils.fcnn import *

"""
This algorithm is Relative Entropy Pairwise Policy Optimization (RePPO) written by Claas A Voelcker, Pieter Abbeel
This is on-policy and actor-critic alogrithm and train robust surrogate value function and effectively use pairwise policy gradient.
Build Maximum entropy framework and combines with a principled KL regularization, 
Categorical Q-Learning, Appropriately normalized neural network architecture and auxilary task
Show how a joint entropy and policy deviation tuning objective can address the twin problems of sufficient exploration and controlled policy update 
According to the paper, it does not utilize the replay buffer
"""

@dataclass(slots=True)


class RePPOAgent:
    def __init__(self, observation_dim, action_dim, num_atoms = 101, 
                 vmin=-250, vmax=250, device='cuda'):
        super().__init__(observation_dim, action_dim, device="cuda")
        
        self.value_net = Critic(observation_dim, action_dim, num_atoms, vmin=vmin, vmax=vmax, device=device)
        
        # Actor includes entropy and kl-regularization which are subject to optimization
        self.policy_net = Actor(observation_dim, action_dim, num_atoms, vmin=vmin, vmax=vmax, device=device)

        self.memory = TensorDict()
    def get_action(self, observation):
        
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, mean, log_std = self.policy_net(obs_tensor)
            
        action_info = {
            'log_prob': log_prob.item(),
            'mean': mean.squeeze().cpu().numpy(),
            'std': log_std.squeeze().cpu().numpy()
        }
        
        return action.squeeze().cpu().numpy(), action_info   
    def update_actor(self, observation,):
        return 
    
    def update_critic(self, ):
        return 
    
    def collect(self, observation, ):
        transition = []
        info_list = []
        
        return 
    
    def evaluate(self,):
        return 
    
    def load(self):
        return 
    
    def save(self, ):
        return 