import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class ResBlock(nn.Module):
    def __init__(self, input_channel, output_channel, time_dim, cond_dim = 0):
        super().__init__()
        ### Different convolution block from 2
        self.norm1 = nn.GroupNorm(input_channel, input_channel)
        self.activ1 = SiLU()
        self.conv1 = nn.Conv2d(in_channels=input_channel, output_channel=output_channel, kernel_size=3, padding= 1) ## Based on the kernel size sum of the h + w would not be changed but their values are slightly changed 

        self.norm2 = nn.GroupNorm(input_channel, output_channel)
        self.activ2 = SiLU()
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding= 1)

        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim + cond_dim, output_channel)
        )

        self.short = nn.Conv2d(input_channel, output_channel, 1) if input_channel != output_channel else nn.Identity()

    def forward(self, x, time_emb, cond_emb = None):
        h = self.conv1(self.activ1(self.norm1(x)))
        emb = time_emb if cond_emb is None else torch.cat([time_emb, cond_emb], dim = -1)
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(self.activ2(self.norm2))
        return h + self.short(x)
    

class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)

        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        q = self.q(x).view(b, self.num_heads, c // self.num_heads, h * w) ### convert (b, c, h, w) -> (b, c, h, w) -> (b, num_heads, channel // num_heads, h * w)
        v = self.v(x).view(b, self.num_heads, c // self.num_heads, h * w)
        k = self.k(x).view(b, self.num_heads, c // self.num_heads, h * 2)

        attn = torch.softmax((q.transpose(-2, -1))/ math.sqrt(c // self.num_heads), dim= -1)
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(b, c, h, w)
        return self.proj(out) + x
    
class Down(nn.Module):
    def __init__(self, input_channel, output_channel, time_dim, cond_dim = 0, use_attn = False):
        super().__init__()
        self.res1 = ResBlock(input_channel=input_channel, output_channel=output_channel, time_dim=time_dim, cond_dim=cond_dim)
        self.res2 = ResBlock(input_channel=output_channel, output_channel=output_channel, time_dim=time_dim, cond_dim=cond_dim)
        
