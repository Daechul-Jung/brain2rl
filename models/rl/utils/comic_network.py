import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


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

    return nn.Sequential(*layers)
        
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


class ReferenceEncoder(nn.Module):
    """
    Reference encoder return latent variable z distribution
    pi_HL(z | s, s_ref) -> Normal distribution over latent z.
    """
    def __init__(self, observation_dim: int, ref_dim: int, z_dim: int, device: str):
        super().__init__()
        self.observation_dim = observation_dim
        self.ref_dim = ref_dim
        self.z_dim = z_dim
        self.encoder = FCNN(
            in_feature = observation_dim + ref_dim,
            out_feature = 2 * z_dim,
            hidden_dim= 512,
            hidden_activation='swish',
            output_activation=None,
            use_norm=True,
            use_output_norm=False, 
            layers = 2,
            device=device
        )
        self.min_std = 1e-3

    def forward(self, s: torch.Tensor, s_ref: torch.Tensor) -> Normal:
        
        x = self.encoder(torch.cat([s, s_ref], dim = -1))
        
        mean, log_std = torch.split(x, x.shape[-1]//2, dim=-1)
        
        std = torch.exp(log_std) + self.min_std
        pi = Normal(mean, std)
        return pi

class LowLevelPolicy(nn.Module):
    """pi_LL(a | s, z) — simple Gaussian policy. """
    def __init__(self, state_dim: int, z_dim: int, act_dim: int, device: str):
        super().__init__()
        self.model = FCNN(
            in_feature = state_dim + z_dim,
            out_feature = 2 * act_dim,
            hidden_dim = 512,
            hidden_activation = 'swish',
            output_activation=None,
            use_norm = True,
            use_output_norm = False,
            layers = 2,
            device = device
        )

    def dist(self, s: torch.Tensor, z: torch.Tensor) -> Normal:
        """construct Normal distribution."""
        # TODO
        pass

    def forward(self, s: torch.Tensor, z: torch.Tensor) -> Normal:
        """Alias for dist."""
        # TODO
        pass

class HighLevelPolicy(nn.Module):
    """Task-specific high-level policy: z ~ pi_HL^{task}(·|o)."""
    def __init__(self, obs_dim: int, z_dim: int):
        super().__init__()
        # TODO: declare mu_net, logstd param

    def forward(self, o: torch.Tensor) -> Normal:
        """TODO: return Normal over z."""
        # TODO
        pass


class MultiHeadValue(nn.Module):
    """Multiple value heads; total value is sum of heads."""
    def __init__(self, state_dim: int, n_heads: int):
        super().__init__()
        # TODO: create n value heads

    def forward(self, s: torch.Tensor):
        """TODO: return (per_head_values, total_value)."""
        # TODO
        pass


class MixtureOfGaussiansPolicy(nn.Module):
    """Optional: pi(a|s,z)=sum_i w_i(s,z) N(a; mu_i(s), o_i(s))."""
    def __init__(self, state_dim: int, z_dim: int, act_dim: int, num_components: int = 4):
        super().__init__()
        # TODO: declare torso for primitives (s), per-component, mixing head (s,z)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        """TODO: return mixture parameters (mus, stds, mix_logits)."""
        # TODO
        pass


class ProductOfGaussiansPolicy(nn.Module):
    """Optional: pi(a|s,z) ~ all_mult_i N(a; mu_i(s), o_i(s))^{w_i(s,z)}."""
    def __init__(self, state_dim: int, z_dim: int, act_dim: int, num_experts: int = 4):
        super().__init__()
        # TODO: declare experts and weighting head

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        """TODO: return experts' params and weights."""
        pass