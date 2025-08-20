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


class SleepingAgent:
    def __init__(self, observation_dim, action_dim, ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim