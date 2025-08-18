import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from typing import Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.diffusion import *
from models.rl.utils.NeuralNetwork import *
from models.rl.utils.fcnn import *
from models.rl.practice.agents.ppo import PPOAgent

"""
This algorithm is Relative Entropy Pairwise Policy Optimization (RePPO) written by Claas A Voelcker, Pieter Abbeel
This is on-policy and actor-critic alogrithm and train robust surrogate value function and effectively use pairwise policy gradient.
Build Maximum entropy framework and combines with a principled KL regularization, 
Categorical Q-Learning, Appropriately normalized neural network architecture and auxilary task
Show how a joint entropy and policy deviation tuning objective can address the twin problems of sufficient exploration and controlled policy update 

"""

class RePPO(PPOAgent):
    def __init__(self, observation_dim, action_dim, device, alpha, beta):
        super().__init__(observation_dim, action_dim, device="cuda")
        self.value_net = NeuralNetwork(input_dim=observation_dim, output_dim=action_dim)
        self.policy_net = DiffusionPolicy(state_dim=observation_dim, action_dim=action_dim, device=device)
        self.temp_alpha = alpha
        self.temp_beta = beta
        
    def get_action(self, observation, training = True):
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, mean, log_std = self.policy_net.get_action(obs_tensor, training)
            
        action_info = {
            'log_prob': log_prob.item(),
            'mean': mean.squeeze().cpu().numpy(),
            'std': log_std.squeeze().cpu().numpy()
        }
        
        return action.squeeze().cpu().numpy(), action_info   
        
    
    