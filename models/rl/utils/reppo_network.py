import torch
import torch.nn as nn 
from torch.distributions import constraints
from torch.distributions.transforms import Transform
from torch.distributions import Normal
import torch.nn.functional as F

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
        encoder_layers = 2,
        head_layers = 2,
        pred_layers = 2,
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
        ## Critic module network for getting logit from feature which is calculated from feature module, return critic value
        self.critic_module = FCNN(
            in_feature=hidden_dim,
            out_feature=num_atoms,  ## originally 1
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
            hl_gauss(torch.zeros(1, device=device), self.vmin, self.vmax, self.num_atoms) ### (1, num_atoms)
        )
        
    def forward(self, observation, action):
        # Concatenate observation and action first over last dimension. Dimension shape is (Batch, observation or action dim)
        input = torch.cat([observation, action], dim = -1) ## (Batch, observation_dim _action_dim)
        ## Learn features via feature network (Encoding)
        features = self.feature_module(input)  ### (hidden_dim, )
        ## Do prediction through prediction module
        next_pred_feature = self.pred_module(features)
        
        ## Getting logit through critic module
        ## Logit is nunormalized scores over num_atoms that supports value = linspace(vmin, vmax, num_atoms)
        logit = self.critic_module(features) + 50 * self.zero_dist  ### (1, num_atoms)

        ## This softmax is for getting categorical distribution P(s,a) over num_atoms
        value_cat = torch.softmax(logit, dim = -1) ### (Batch, num_atoms) = (1, num_atoms)
        
        ### Q-value = sum(p_i * z_i)
        value = value_cat @ self.values  ### (Batch, num_atoms) @ (num_atoms, 1) -> (Batch, 1) = [1]

        return value, logit, next_pred_feature, features
    
class Actor(nn.Module):
    ### Policy network 
    def __init__(self, observation_dim, action_dim, 
                 entropy_start: float, kl_start: float,
                 hidden_dim = 256, use_norm = True, 
                 layers = 4, min_std = 0.1, device = None):
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
        self.log_temp = nn.Parameter( #### Alpha
            torch.log(torch.tensor(entropy_start, device=device, dtype = torch.float32))
        )


        ## Setting kl-regulation for lagrange as parameter for update. both temp and lagrange are updated in loss function 
        self.log_lagrange = nn.Parameter(  #### Beta
            torch.log(torch.tensor(kl_start, device = device, dtype = torch.float32))
        )

        self.min_std = min_std

    def forward(self, observation):
        x = self.model(observation)
        mean, log_std = torch.split(x, x.shape[-1]//2, dim=-1)
        LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0      # tweakable; [-7, 1] also common
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std) + self.min_std

        pi = Normal(mean, std, validate_args=False)
        transformed_pi = torch.distributions.TransformedDistribution(
            pi, [torch.distributions.TanhTransform(cache_size=1)]
        )
        return transformed_pi, torch.tanh(mean), F.softplus(torch.exp(self.log_temp)) + 1e-6, F.softplus(torch.exp(self.log_lagrange)) + 1e-6
    

def hl_gauss(input, vmin, vmax, num_atoms):
    if input.dim() == 2 and input.size(-1) == 1:
        x = input.squeeze(-1)   # [B]
    elif input.dim() == 1:
        x = input               # [B]
    else:
        raise ValueError(f"hl_gauss expects [B] or [B,1]; got {tuple(input.shape)}")

    x = torch.clamp(x, vmin, vmax)                     # [B]
    bin_width = (vmax - vmin) / (num_atoms - 1)
    sigma = 0.75 * bin_width

    edges = torch.linspace(                            # [A+1]
        vmin - bin_width/2, vmax + bin_width/2,
        num_atoms + 1, device=x.device, dtype=x.dtype
    )
    import math
    denom = math.sqrt(2.0) * sigma + 1e-6
    diff = (edges.unsqueeze(0) - x.unsqueeze(1)) / denom   # [B, A+1]
    cdf  = torch.erf(diff)                                  # [B, A+1]

    z     = cdf[:, -1] - cdf[:, 0]                         # [B]
    probs = (cdf[:, 1:] - cdf[:, :-1]) / (z.unsqueeze(1) + 1e-8)  # [B, A]
    return probs


class EmpiricalNormalizer(nn.Module):
    """Normalize Mean and variance of values based on empirical values"""

    def __init__(self, shape, device, eps=1e-2, until = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device =device
        feat_dim = shape if isinstance(shape, int) else int(shape[-1])
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype = torch.long).to(device))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()
    
    @property
    def std(self):
        return self._std.squeeze(0).clone()
    
    def forward(self, x: torch.Tensor, center: bool = True) -> torch.Tensor:
        feat_dim = self._mean.shape[-1]

        if x.shape[-1] != feat_dim:
            raise ValueError(f"Expected input of shape (*, {feat_dim}), got {tuple(x.shape)}")

        # reshape to [B, D] for stats but preserve original shape for return
        orig_shape = x.shape
        x2d = x.view(-1, feat_dim)

        # if x.shape[-1] != self._mean.shape[-1 : ]:
        #     raise ValueError(
        #         f'Expected input of shape (*, {self._mean.shape[-1:]}), got {x.shape}'
        #     )
        
        if self.training:
            self.update(x2d)

        if center:
            y = (x2d - self._mean) / (self._std + self.eps)
        else:
            y = x2d / (self._std + self.eps)

        return y.view(orig_shape)
        
    @torch.jit.unused
    def update(self, x2d):
        if self.until is not None and self.count >= self.until:
            return
        B = x2d.shape[0]
        if B == 0:
            return

        new_count = self.count + B

        batch_mean = x2d.mean(dim=0, keepdim=True)                     # [1, D]
        batch_var  = x2d.var(dim=0, unbiased=False, keepdim=True)      # [1, D]

        delta = batch_mean - self._mean                                 # [1, D]
        m_a   = self._var * self.count                                  # [1, D]
        m_b   = batch_var * B                                           # [1, D]
        M2    = m_a + m_b + (delta**2) * (self.count * B / new_count.clamp_min(1))

        self._mean = self._mean + delta * (B / new_count.clamp_min(1))
        self._var  = M2 / new_count.clamp_min(1)
        self._std  = torch.sqrt(self._var + 1e-8)
        self.count = new_count


    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean
