import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union

class NeuralNetwork(nn.Module):
    """
    Neural network for OpenArm or humanoid v-4
    Later I should fix this part more for utilizing information and complex tasks
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        """_summary_

        Args:
            input_dim (int): input dimension of the network which would be observation space dimension
            output_dim (int): output dimension of the network which would be action space dimension
            hidden_dims (List[int], optional): Hidden dimension of neural network. Defaults to [256, 256, 128].
        """
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)