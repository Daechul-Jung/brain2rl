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

"""
This is an algorithm called CoMic: Complementary Task Learning & Mimicry for Reusable Skils written by Google DeepMind
"""
class CoMic:
    def __init__(self, observation_dim, action_dim, device):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = device
        self.value_net = NeuralNetwork(input_dim=observation_dim, output_dim=action_dim)
        self.policy_net = DiffusionPolicy(state_dim=observation_dim, action_dim=action_dim, device=device)

    def get_action(self):
        return 
    
    def update(self):
        return 
    

