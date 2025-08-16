import torch
import torch.nn as nn 
from torch.distributions import constraints
from torch.distributions import Normal

def get_activation(name):
    if name == 'relu':
        return nn.ReLU()
    elif name == 'swish':
        return nn.SiLU()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")
    
def normalized_activation_layer(in_features, out_features, use_norm = True, activation = 'swish', device = 'cuda'):
    layers = [nn.Linear(in_features, out_features, device=device)]
    if use_norm:
        layers.append(nn.RMSNorm([out_features], device=device))
    if activation is not None:
        layers.append(get_activation(activation))
        
class FCNN(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_dim = 256, 
                 hidden_activation = 'swish', output_activation = None,
                 use_norm = True, use_output_norm = False, layers = 3,
                 input_activation = False, device = None):
        super().__init__()
        net = []
        if layers == 1:
            net.append(normalized_activation_layer(in_feature, out_feature, use_norm = use_output_norm, 
                                                   activation= output_activation, device = device ))
        else:
            if input_activation:
                net.append(get_activation(hidden_activation))
            net.append(
                normalized_activation_layer(
                    in_feature, hidden_dim, use_norm=use_norm,
                    activation=hidden_activation, device=device
                )
            )
            for _ in range(layers-2):
                net.append(
                    normalized_activation_layer(
                        hidden_dim, hidden_dim, use_norm=use_norm,
                        activation=hidden_activation, device=device
                    )
                )
            net.append(
                normalized_activation_layer(
                    hidden_dim, out_feature, use_norm=use_output_norm,
                    activation=output_activation, device=device
                )
            )
        self.net = nn.Sequential(*net)
        
    def forward(self, x):
        return self.net(x)