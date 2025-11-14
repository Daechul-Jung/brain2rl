from typing import Tuple

import torch
import torch.nn as nn
import os, sys
default_init = torch.nn.init.xavier_uniform_
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.vit_encoder import DynamicGroupNorm


@torch.jit
def mish(x):
    return x * torch.tanh(torch.nn.Softplus(x))


def unet_squaredcos_cap_v2(timesteps, s = 0.008):
    t = torch.linspace(0, timesteps, timesteps + 1) / timesteps
    alphas_cumprod = torch.cos((t+s) / (1+s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0, 0.999)

"""
Classes below takes output features and do lazy build when they firstly forward 
"""


class SinusoidalPosEmb(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features

    def forward(self, x: torch.Tensor):
        """
        x: tensors of arbitrary shape (..., ) I broadcast over a new last dim
        returns (..., 2 * (features //2 ))
        """
        half_features = self.features//2
        div_term = torch.log(10000) / (half_features-1)
        freq = torch.exp(torch.arange(half_features, device=x.device)* -div_term)
        args = x[..., None] * freq
        embedding = torch.concat((torch.sin(args), torch.cos(args)), dim = -1)
        return embedding
    

class DownSample1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features= features ## output features
        self.conv = nn.Conv1d(in_channels=None, out_channels=None, kernel_size=3, stride=1, padding = 0)
        
    def _lazy_build(self, C_in: int, features: int):
        self.conv = nn.Conv1d(C_in, features, kernel_size=3, stride =2, padding = 1)

    def forward(self, x: torch.Tensor):
        if self.conv.in_channels is None:
            self._lazy_build(x.shape[1], features = self.features)
        return self.conv(x)


class UpSample1d(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.features = features ## output features
        self.deconv = nn.ConvTranspose1d(in_channels = None, out_channels=None, kernel_size=4, stride = 2, padding = 0)
    
    def _lazy_build(self, C_in: int, features: int):
        self.deconv = nn.ConvTranspose1d(in_channels=C_in, out_channels=features, kernel_size=4, stride = 2, padding = 1)

    def forward(self, x: torch.tensor):
        if self.deconv.in_channels is None:
            self._lazy_build(x.shape[1], self.features)

        return self.deconv(x)
    

class Conv1dBlock(nn.Module):
    """
    Conv1d -> GroupNorm --> Mish
    """

    def __init__(self, features: int, kernel_size: int, n_groups: int):
        super().__init__()
        self.features = features  ## output features
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.conv = nn.Conv1d(in_channels=None, out_channels=None, kernel_size=kernel_size)
        self.gn = DynamicGroupNorm(num_channels=None)

    def _lazy_build(self, C_in: int, features: int):
        self.conv = nn.Conv1d(in_channels=C_in, out_channels=features, kernel_size=self.kernel_size, stride = 1, padding = self.kernel_size//2)
        self.gn = DynamicGroupNorm(C_in, self.n_groups)

    def forward(self, x: torch.Tensor):
        if self.conv.in_channels is None:
            self._lazy_build(x.shape[1], self.features)
        x = self.conv(x)
        x = self.gn(x)
        x = mish(x)
        return x
    

class ConditionalResidualBlock1d(nn.Module):
    """
    Assume that X: [B, C, L]
    cond: [B, cond_dim]
    Flow: Conv1dBlock -> FiLM -> Conv1dBlock -> residual
    Use Conv1dBlock(conv1d -> groupNorm -> mish)
    Put x into block1 and also put condtion vector into cond_mlp for embedding and then combine with output from block1 and embedded condtion vector
    Then put those into block2 and finally combine with residual 
    Simply Conv1dBlock -> cond_mlp -> Conv1dBlock
    """
    def __init__(self, features, kernel_size: int = 3, n_groups: int = 8, residual_proj: bool= False, cond_dim: int = 128):
        super().__init__()
        self.features = features
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.block1 = Conv1dBlock(features, kernel_size, n_groups)  ### Convolve and group norm and mish
        self.block2 = Conv1dBlock(features, kernel_size, n_groups)
        
        self.cond_mlp = nn.Linear(cond_dim, features * 2)
        self.residual_proj = residual_proj
        if residual_proj:
            self.residual = nn.Conv1d(in_channels=None, out_channels=None)

    def _lazy_build(self, C_in: int, features: int):
        self.residual = nn.Conv1d(C_in,features, features, kernel_size=1, stride = 1, padding = 0)


    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        residual = x

        x = self.block1(x)
        cond = self.cond_mlp(cond) ## [B, 2*features]
        
        scale, bias = torch.split(cond, 2, dim = -1) ## [B, features], [B, features]
        x = x * (1 + scale[:, :, None]) + bias[:, :, None] ## broadcast over time dimension

        x = self.block2(x)

        self._lazy_build(x.shape[1], self.features)
        if self.residual is not None:
            residual = self.residual(residual)

        return x + residual
    
class ConditionalUnet1D(nn.Module): 
    """
    Conditional Unet1D 
    Utilize 
        SinusoidalPosEmd:
        time_mlp1,2: 
        down_block1,2: ConditionalResidualBlock1D which include Conv1dBlock but use nn.Conv1d for residual 
        down and up sample: Up(Down)Sample1D
    """
    def __init__(self, down_features:Tuple[int] = (256, 512, 1024), mid_layer: int = 2, kernel_size: int = 3, n_groups: int = 8, time_features: int = 256):
        super().__init__()
        self.down_features = down_features
        self.mid_layer = mid_layer
        self.kernel_size = kernel_size
        self.n_groups = n_groups
        self.time_features = time_features
        ## time embedding: Sinusoidal -> Linear(4*D) -> Mish -> Linear(D)
        self.pos_embed = SinusoidalPosEmb(time_features)
        self.time_mlp1 = nn.Linear(time_features, 4 * time_features)
        self.time_mlp2 = nn.Linear(time_features * 4, time_features)

        self.down_block1 = nn.ModuleList()
        self.down_block2 = nn.ModuleList()
        self.downsample_layer = nn.ModuleList()

        self.mid_block = nn.ModuleList()

        self.up_block1 = nn.ModuleList()
        self.up_block2 = nn.ModuleList()
        self.upsample_layer = nn.ModuleList()


    def _build(self, obs_dim: int):
        cond_dim = obs_dim + self.time_features

        ## project down
        for i, features in enumerate(self.down_features):
            self.down_block1.append(
                ConditionalResidualBlock1d(features, kernel_size = self.kernel_size, n_groups = self.n_groups, residual_proj=True, cond_dim=cond_dim)
            )
            self.down_block2.append(
                ConditionalResidualBlock1d(features, kernel_size = self.kernel_size, n_groups = self.n_groups, cond_dim=cond_dim)
            )
            
            if i != len(self.down_features) - 1:
                self.downsample_layer.append(DownSample1d(features= features))

            else:
                self.downsample_layer.append(nn.Identity())


        ## Mid layers
        for _ in range(self.mid_layer):
            self.mid_block.append(
                ConditionalResidualBlock1d(features = self.down_features[-1], kernel_size=self.kernel_size, n_groups=self.n_groups)
            )

        ## project up

        for features in reversed(self.down_features):
            self.up_block1.append(
                ConditionalResidualBlock1d(features= features, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=True, cond_dim=cond_dim)
            )
            self.up_block2.append(
                ConditionalResidualBlock1d(features= features, kernel_size=self.kernel_size, n_groups=self.n_groups, residual_proj=False, cond_dim=cond_dim)
            )
            self.upsample_layer.append(
                UpSample1d(features = features)
            )

        self.final = Conv1dBlock(features=self.down_features[0], kernel_size=self.kernel_size, n_groups=self.n_groups)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, time: torch.Tensor, train: bool = False):
        """
        obs: [B, obs_dim]
        action: [B, action_dim]
        time: [B, ]
        """
        B = action.shape[0]

        if time.ndim > 1:
            time = time.squeeze(-1)
        
        time = self.pos_embed(time)
        time = self.time_mlp1(time)
        time = mish(time)
        time = self.time_mlp2(time)
        
        obs_dim = obs.shape[-1]
        self._build(obs_dim)

        cond = torch.cat([obs, time], dim = -1)  ## [B, obs_dim + time_dim]

        hidden_reps = []
        x = action
        for i, features in enumerate(self.down_features):
            x = self.down_block1[i](x, cond)
            x = self.down_block2[i](x, cond)

            if i != 0:
                hidden_reps.append(x) ## store skip, except first one

            if i != len(self.down_features) - 1:
                x = self.downsample_layer[i](x)  ## downsample, except the last layer
            
        for block in self.mid_block:
            x = block(x, cond)

        features4Up = list(self.down_features[:-1])
        skips = hidden_reps
        assert len(features4Up) == len(skips), "Miss match in skip connection"

        for feature, skip in reversed(list(zip(features4Up, skips))):
            x = torch.cat([x, skip], dim = 1)
            x = self.up_block1[0](x, cond)
            self.up_block1 = self.up_block1[1:]
            x = self.up_block2[0](x, cond)
            self.up_block2 = self.up_block2[1:]

            x =self.upsample_layer[0](x)
            self.upsample_layer = self.upsample_layer[1:]

        x = self.final(x)

        return x