from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.utils.embedding import *
from models.diffusion.models.block import Down, Up, ResBlock, SelfAttention2d, SiLU

class UNetDenoiser(nn.Module):
    def __init__(self, in_channels, base, time_dim, cond_dim, attn_res: List[int]):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.cond_proj = nn.Sequential(nn.Linear(cond_dim, cond_dim), nn.SiLU()) if cond_dim > 0 else None

        self.in_conv = nn.Conv2d(in_channels, base, kernel_size=3, padding= 1)
        self.down1 = Down(base, base, time_dim, cond_dim, use_attn=(32 in attn_res))
        self.down2 = Down(base, base*2, time_dim=time_dim, cond_dim=cond_dim, use_attn=(16 in attn_res))
        self.down3 = Down(base*2, base*2, time_dim=time_dim, cond_dim=cond_dim, use_attn=(8 in attn_res))

        self.mid1 = ResBlock(base*2, base*2, time_dim=time_dim, cond_dim=cond_dim)
        self.mid_attn = SelfAttention2d(base*2)
        self.mid2 = ResBlock(base*2, base*2, time_dim, cond_dim)

        self.up3 = Up(base*2, skip_channel=base*2, output_channel=base*2, time_dim = time_dim, cond_dim=cond_dim, use_attn=(8 in attn_res))
        self.up2 = Up(base*2, skip_channel=base*2, output_channel= base, time_dim= time_dim, cond_dim= cond_dim, use_attn=(16 in attn_res))
        self.up1 = Up(base, skip_channel=base, output_channel=base, time_dim=time_dim, cond_dim=cond_dim, use_attn= (32 in attn_res))

        self.out_norm = nn.GroupNorm(32, base)
        self.out_act = SiLU()
        self.out_conv = nn.Conv2d(base, in_channels, kernel_size=3, padding = 1)

    def forward(self, x, t: torch.LongTensor, cond : Optional[torch.Tensor] = None):
        time_emb = self.time_mlp(timestep_embedding(t, self.time_mlp[0].in_features))  ## shape of (128, 256)
        cond_emb = self.cond_proj(cond) if (self.cond_proj is not None and cond is not None) else None  # None

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

        return x ## this predict epsilon prime

