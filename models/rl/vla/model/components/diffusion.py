# Mostly adapted from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
import logging
from typing import Callable, Sequence, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

default_init = nn.init.xavier_uniform


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule 
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """

    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps

    alphas_cumprod = torch.cos((t + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1: ]/ alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)


class ScoreActor(nn.Module):
    """
    This Diffusion model is used for DiffusionActionHead model.
    """
    def __init__(self, time_preprocess: nn.Module, cond_encoder: nn.Module, reverse_network: nn.Module):
        super().__init__()
        self.time_preprocess = time_preprocess  ### FourierFeatures
        self.cond_encoder = cond_encoder  ### MLP 
        self.reverse_netowrk = reverse_network  ### MLPResNet 

    def forward(self, obs_enc, action, time, train = False):
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        if time.ndim == 0:
            time = time[None]
        if time.shape[-1] != 1 and time.ndim >= 1:
            time = time.unsqueeze(-1)
        ## Firstly encode time vector 
        t_ff = self.time_preprocess(time)

        ## Secondly encoded time vector into condition encoder 
        try:
            cond_enc = self.cond_encoder(t_ff, train = train)
        except TypeError:
            cond_enc = self.cond_encoder(t_ff)

        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            logging.debug(
                "Broadcasting obs_enc from %s to %s", obs_enc, new_shape
            )
            ### broadcast encoded observation to new shape dimension
            obs_enc = torch.broadcast_to(obs_enc, new_shape)
        ### concatenate encoded time condition vector, encoded observation, and action 
        reverse_input = torch.concat([cond_enc, obs_enc, action], dim = -1)
        eps_pred = self.reverse_netowrk(reverse_input, train = train)
        return eps_pred
    
class FourierFeatures(nn.Module):
    """
    Learnable or fixed Fourier features, time_preprocess module for ScoreActor
    Input: x[*, input_dim]
    Output: [*, output_size] with cos/sin concatenated
    """
    def __init__(self, output_size: int, learnable: bool = True):
        super().__init__()
        self.output_size = output_size
        self.learnable = learnable
        self.weight: Optional[nn.Parameter] = None

    def _build(self, input_dim, device, dtype):
        half = self.output_size // 2
        if self.learnable:
            w = torch.empty(half, input_dim, device=device, dtype=dtype) ## torch way to initialize jax.param
            nn.init.normal_(w, mean = 0.0, std = 0.2)
            self.weight = nn.Parameter(w)
        else:
            div = math.log(10000.0) / max(half - 1, 1)
            freqs = torch.exp(torch.arange(half, device=device, dtype= dtype) * (-div))
            self.register_buffer('fixed_freqs', freqs, persistent=False)

    def forward(self, x: torch.Tensor):
        *_, input_dim = x.shape
        if (self.learnable and self.weight is None) or (not self.learnable and 'fixed_freqs' not in self._buffers):
            self._build(input_dim, x.device, x.dtype)

        if self.learnable:
            f = 2 * math.pi * (x @ self.weight.t())
        else:
            f = x * self.fixed_freqs

        return torch.cat([torch.cos(f), torch.sin(f)], dim = -1)
    
class MLP(nn.Module):
    """
    Feed forward MLP with optional LayerNorm and Dropout
    hidden_dims: e.g. (2 * time_dim, time_dim)
    activation: default swish (silu)
    activate_final: whether to activate on last layer too
    """
    def __init__(self, 
                 hidden_dims: Sequence[int],  ### (2 * time_dim, time_dim)
                 activation: Callable = F.silu,
                 activate_final: bool = False,
                 use_layer_norm: bool = False,
                 dropout_rate: Optional[float] = None 
                 ):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.activate_final = activate_final
        self.use_layer_norm = use_layer_norm
        self.dropout_rate = 0.0 if dropout_rate is None else dropout_rate

        layers = []
        norms = []

        for i, dim in enumerate(self.hidden_dims):
            lin = nn.Linear(0,0)
            layers.append(lin)
            norms.append(nn.LayerNorm(dim) if use_layer_norm else None)
        
        self.layers = nn.ModuleList(layers)
        self.norms = nn.ModuleList([n for n in norms if n is not None])
        self._built = False


    def _build(self, input_dim: int):
        prev = input_dim
        norm_idx = 0
        new_layers = nn.ModuleList()
        new_norms = nn.ModuleList()

        for i, dim in enumerate(self.hidden_dims):
            lin = nn.Linear(prev, dim)
            default_init(lin.weight)
            nn.init.normal_(lin.bias, std = 1e-6)
            new_layers.append(lin)
            if self.use_layer_norm:
                new_norms.append(nn.LayerNorm(dim))
            prev = dim
        
        self.layers = new_layers
        self.norms = new_norms
        self._built = True

    def forward(self, x: torch.Tensor, train: bool = False):

        if not self._built:
            self._build(x.shape[-1])

        norm_idx = 0

        for i, lin in enumerate(self.layers):
            x = lin(x)
            is_last = (i == len(self.layers) - 1)

            if (not is_last) or self.activate_final:
                if self.dropout_rate > 0:
                    x = F.dropout(x, p = self.dropout_rate, training=train)

                if self.use_layer_norm:
                    x = self.norms[norm_idx](x)
                    norm_idx += 1
                x = self.activation(x)

        return x
    
class MLPResNetBlock(nn.Module):
    def __init__(self, features: int, act: Callable, dropout_rate: float = None, use_layer_norm: bool = False):
        self.features = features
        self.act = act
        self.dropout_rate = 0.0 if dropout_rate is None else dropout_rate
        self.use_layer_norm = use_layer_norm
        self.layerNorm = nn.LayerNorm(features) if use_layer_norm else None
        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)
        self.proj: Optional[nn.Linear] = None

        default_init(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        default_init(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor, train: bool = False):
        residual = x

        if self.proj is None and residual.shape[-1] != self.features:
            self.proj = nn.Linear(residual.shape[-1], self.features)
            default_init(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

        if self.dropout_rate > 0:
            x = F.dropout(x, self.dropout_rate, training=train)
        
        if self.layerNorm is not None:
            x = self.layerNorm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)

        if self.proj is not None:
            residual = self.proj(residual)

        return residual + x
    
class MLPResNet(nn.Module):
    def __init__(self, num_blocks: int, 
                 output_dim: int, 
                 dropout_rate: float = None, 
                 use_layer_norm: bool = False, 
                 hidden_dim: int = 256,
                 activation: Callable = F.silu):
        super().__init__()
        self.num_blocks = num_blocks
        self.output_dim = output_dim
        self.dropout_rate = 0.0 if dropout_rate is None else dropout_rate
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.blocks = nn.ModuleList([ 
            MLPResNetBlock(self.hidden_dim, self.activation, dropout_rate=self.dropout_rate, use_layer_norm= self.use_layer_norm) 
            for _ in range(self.num_blocks)])
        self.in_proj = nn.Linear(0,0)
        self.out_act = activation
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._built = False
        default_init(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _build(self, input_dim: int):
        self.in_proj = nn.Linear(input_dim, self.hidden_dim)
        default_init(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        self._built = True

    def forward(self, x: torch.Tensor, train: bool = False):
        if not self._built:
            self._build(x.shape[-1])
        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x, train= train)
        x = self.out_act(x)
        x = self.out_proj(x)

        return x
    

def create_diffusion_model(
        out_dim: int,
        time_dim: int,
        num_blocks: int,
        dropout_rate: float,
        hidden_dim: int,
        use_layer_norm: bool
):
    return ScoreActor(
        FourierFeatures(time_dim, learnable=True),  ## time encoder
        MLP((2*time_dim, time_dim)),  ## condition encoder
        MLPResNet(  ### Reverse Network
            num_blocks,
            out_dim,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm
        )
    )