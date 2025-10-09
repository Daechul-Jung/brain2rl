from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.utils.embedding import *
from models.diffusion.models.block import Down, Up, ResBlock, SelfAttention1d, SiLU

class UNetDenoiser(nn.Module):
    """
    Predict epsilon for x_t shaped as actions over horizon 
    x: (B, H, action_dim) This is set of actions 
    Internally, we use channel-first: (B, C=action_dim, L = H)

    Conditioning:
    -t : (B, ) timestep -> sinusoidal + MLP -> time_emb: (B, time_dim)
    -cond_vec: (B, cond_dim) (ex pooled EEG tokens + state conditioning)
    """
    def __init__(self, action_dim, base, time_dim, cond_dim, attn_res: List[int] = [8, 16, 32]):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU()) if cond_dim > 0 else None

        in_channels = action_dim
        self.in_conv = nn.Conv1d(in_channels, base, kernel_size=3, padding = 1)

        self.down1 = Down(base, base, time_dim, cond_dim, use_attn=(32 in attn_res))
        self.down2 = Down(base, base*2, time_dim=time_dim, cond_dim=cond_dim, use_attn=(16 in attn_res))
        self.down3 = Down(base*2, base*2, time_dim=time_dim, cond_dim=cond_dim, use_attn=(8 in attn_res))

        self.mid1 = ResBlock(base*2, base*2, time_dim=time_dim, cond_dim=cond_dim)
        self.mid_attn = SelfAttention1d(base*2)
        self.mid2 = ResBlock(base*2, base*2, time_dim, cond_dim)

        self.up3 = Up(base*2, skip_channel=base*2, output_channel=base*2, time_dim = time_dim, cond_dim=cond_dim, use_attn=(8 in attn_res))
        self.up2 = Up(base*2, skip_channel=base*2, output_channel= base, time_dim= time_dim, cond_dim= cond_dim, use_attn=(16 in attn_res))
        self.up1 = Up(base, skip_channel=base, output_channel=base, time_dim=time_dim, cond_dim=cond_dim, use_attn= (32 in attn_res))

        self.out_norm = nn.GroupNorm(8, base)
        self.out_act = SiLU()
        self.out_conv = nn.Conv1d(base, in_channels, kernel_size=3, padding = 1)

        self._time_dim = time_dim

    def forward(self, x, t: torch.LongTensor, cond_vec : Optional[torch.Tensor] = None):
        """
        x: (B, H, action_dim) set of actions , H = time horizon
        t: (B, ) integer time step
        cond_vec: (B, cond_dim) or None
        Returns:
            eps_prediction: (B, H, action_dim)
        """
        time_emb = timestep_embedding(t, self._time_dim)
        time_emb = self.time_mlp(time_emb)

        cond_emb = self.cond_proj(cond_vec) if (self.cond_proj is not None and cond_vec is not None) else None

        x = self.in_conv(x)

        x, skip1 = self.down1(x, time_emb, cond_emb)
        x, skip2 = self.down2(x, time_emb, cond_emb)
        x, skip3 = self.down3(x, time_emb, cond_emb)

        x = self.mid1(x, time_emb, cond_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, time_emb, cond_emb)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up3(x, skip3, time_emb, cond_emb)
        x = self.up2(x, skip2, time_emb, cond_emb)
        x = self.up1(x, skip1, time_emb, cond_emb)

        x = self.out_conv(self.out_act(self.out_norm(x)))

        return x.transpose(1,2).contiguous() ## (B, H, A)  this predict epsilon prime

