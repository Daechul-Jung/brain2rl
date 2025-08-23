import torch
import torch.nn as nn 
from torch.distributions import constraints
from torch.distributions.transforms import Transform
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
    

class Critic(nn.Module):
    ## Value network
    def __init__(self,
        observation_dim,
        action_dim, 
        num_atoms: int,
        vmin: float,
        vmax: float,
        hidden_dim = 256,
        use_norm = True,
        use_encoder_norm = False,
        encoder_layers = 1,
        head_layers = 1,
        pred_layers = 1,
        device = None
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.vmin = vmin
        self.vmax = vmax
        self.hidden_dim = hidden_dim
        # Feature module network for getting features from the observation and action
        self.feature_module = FCNN(
            in_feature=observation_dim + action_dim,
            out_feature=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation='swish',
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=use_encoder_norm,
            layers=encoder_layers,
            device=device
        )
        ## Critic module network for getting logit from feature which is calculated from feature module 
        self.critic_module = FCNN(
            in_feature=hidden_dim,
            out_feature=num_atoms,
            hidden_dim=hidden_dim,
            hidden_activation='swish',
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False,
            input_activation=True,
            layers = head_layers,
            device = device
        )
        ## Prediction module network for getting next feature
        self.pred_module = FCNN(
            in_feature=hidden_dim,
            out_feature=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_activation='swish',
            output_activation=None,
            use_norm=use_norm,
            input_activation=True,
            use_output_norm=False,
            layers=pred_layers,
            device=device
        )

        self.values = torch.linspace(vmin, vmax, num_atoms, device=device, dtype = torch.float32)
        zeros = hl_gauss(torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms)
        zeros.requires_grad = True
        self.zero_dist = nn.Parameter(
            hl_gauss(torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms)
        )
        
    def forward(self, observation, action):
        # Concatenate observation and action first over last dimension. Dimension shape is (Batch, observation or action dim)
        input = torch.cat([observation, action], dim = -1)
        ## Learn features via feature network (Encoding)
        features = self.feature_module(input)
        ## Do prediction through prediction module
        next_pred_feature = self.pred_module(features)
        ## Getting logit through critic module
        logit = self.critic_module(features)

        ## Concatenate value
        value_cat = torch.softmax(logit, dim = -1)
        value = value_cat @ self.values

        return value, logit, next_pred_feature, features
    
class Actor(nn.Module):
    ### Policy network 
    def __init__(self, observation_dim, action_dim, 
                 entropy_start: float, kl_start: float,
                 hidden_dim = 256, use_norm = True, 
                 layers =3, min_std = 0.1, device = None):
        super().__init__()
        ### Actor model for getting probability including mean and std. By using this, making distribution and get action


        self.model = FCNN(
            in_feature=observation_dim,
            out_feature= 2 * action_dim,
            hidden_dim=hidden_dim,
            hidden_activation='swish',
            output_activation=None,
            use_norm=use_norm,
            use_output_norm=False, 
            layers = layers,
            device=device
        )


        ## Setting entropy as parameter for update
        self.log_temp = nn.Parameter(
            torch.log(torch.tensor(entropy_start, device=device, dtype = torch.float32))
        )


        ## Setting kl-regulation for lagrange as parameter for update. both temp and lagrange are updated in loss function 
        self.log_lagrange = nn.Parameter(
            torch.log(torch.tensor(kl_start, device = device, dtype = torch.float32))
        )

        self.min_std = min_std

    def forward(self, observation):
        x = self.model(observation)
        mean, log_std = torch.split(x, x.shape[-1]//2, dim=-1)
        std = torch.exp(log_std) + self.min_std
        pi = Normal(mean, std, validate_args=False)
        transformed_pi = torch.distributions.TransformedDistribution(
            pi, [torch.distributions.TanhTransform()]
        )
        return transformed_pi, torch.tanh(mean), torch.exp(self.log_temp), torch.exp(self.log_lagrange)
    
def hl_gauss(input, vmin, vmax, num_atoms):
    """
    """
    if input.dim() == 0:
        input = input.view(1, 1)
    elif input.dim() == 1:
        input = input.unsqueeze(-1)        # [B] -> [B,1]
    elif input.dim() == 2 and input.size(-1) == 1:
        pass                       # already [B,1]
    else:
        raise ValueError(f"hl_gauss expects [B] or [B,1]; got {tuple(input.shape)}. "
                         f"Did you pass a feature vector (e.g., 256-d)?")
    x = torch.clip(input, vmin, max = vmax)
    bin_width = (vmax - vmin) / (num_atoms - 1)
    sigma_to_final_sigma_ratio = 0.75
    support = torch.linspace(
        vmin - bin_width/2,
        vmax + bin_width/2,
        num_atoms + 1,
        device = input.device)
    sigma = bin_width * sigma_to_final_sigma_ratio
        
    cdf_evals = torch.erf(
        (support.unsqueeze(0) - x).squeeze() /
        (torch.sqrt(torch.tensor(2.0)) * sigma + 1e-6)
    )
    
    z = cdf_evals[..., -1] - cdf_evals[..., 0] 
    
    target_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
    target_probs = (target_probs / (z.unsqueeze(-1) + 1e-6)).reshape(
        *input.shape[:-1], num_atoms
    )
    return target_probs