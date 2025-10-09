import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# - tokens: (B, num_tokens, token_dim)
# - cond_vec (what we pass to the denoiser): (B, cond_dim)
# - actions over a short horizon: (B, H, action_dim)
#   internally we use channels-first as (B, C=action_dim, L=H)

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class ResBlock(nn.Module):
    """
    ResNet block for up and down sampling with FiLM-style conditioning from (time_emb, cond_emb)
    in_channel, out_channel: channel dims over action feature channel 
    time_dim, cond_dim = dimensions of embedding I add in
    """
    def __init__(self, input_channel, output_channel, time_dim, cond_dim = 0, groups: int = 8):
        super().__init__()
        ### Different convolution block from 2
        self.norm1 = nn.GroupNorm(groups, input_channel) ## Applies group normalization over a mini-batch, similar to batch norm, first argument I used input channel, but it occurs error. so I replace to 32
        self.activ1 = SiLU()
        self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=3, padding= 1) ## Based on the kernel size sum of the h + w would not be changed but their values are slightly changed 

        self.norm2 = nn.GroupNorm(groups, output_channel) ## first argument I used input channel, but it occurs error. so I replace to 32
        self.activ2 = SiLU()
        self.conv2 = nn.Conv1d(output_channel, output_channel, 3, padding= 1)
        embed_in = time_dim + (cond_dim if cond_dim > 0 else 0)
        self.emb = nn.Sequential(
            ### Activation first and projection
            nn.SiLU(),
            nn.Linear(embed_in, output_channel)
        )

        self.short = nn.Conv1d(input_channel, output_channel, kernel_size=1) if input_channel != output_channel else nn.Identity()

    def forward(self, x, time_emb, cond_emb = None):
        """
        x: (B, channel, L)
        """
        h = self.conv1(self.activ1(self.norm1(x))) ## normalize -> activation -> convolution
        emb = time_emb if cond_emb is None else torch.cat([time_emb, cond_emb], dim = -1) ## Concatenate time and condition embedding 
        h = h + self.emb(emb)[:, :, None] ## embedding and largening the dimensions and adding with the normalized input
        h = self.conv2(self.activ2(self.norm2(h))) ## Same process as first h
        return h + self.short(x)
    

class SelfAttention1d(nn.Module):
    def __init__(self, channels, num_heads = 8):
        """
        Multi head self-attention over the temporal axis L
        Useful when H >1 
        """
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Conv1d(channels, channels, kernel_size=1)
        self.v = nn.Conv1d(channels, channels, 1) ### Even though (c, c, 1) shape of convolution, with (b, c, h, w), it returns (b, c, h, w) and does not impact on batch and channel. only impact on h, w 
        self.k = nn.Conv1d(channels, channels, 1)

        self.proj = nn.Conv1d(channels, channels, 1)
        self.scale = math.sqrt(channels // num_heads)


    def forward(self, x):
        """
        x : (B, C, L)
        """
        B, C, L = x.shape

        q = self.q(x).view(B, self.num_heads, C // self.num_heads, L) ### convert (b, c, h, w) -> (b, c, h, w) -> (b, num_heads, channel // num_heads, h * w)
        v = self.v(x).view(B, self.num_heads, C // self.num_heads, L) 
        k = self.k(x).view(B, self.num_heads, C // self.num_heads, L)

        attn = torch.softmax(torch.einsum('bhdl,bhdl->bhll', q.transpose(-2, -1), k)/ self.scale, dim=-1)  ### (B, H, L, L)
        out = torch.einsum('bhll,bhdl->bhdl', attn, v).reshape(B, C, L) # 
        return self.proj(out) + x
    
class Down(nn.Module):
    """
    Down sampling for UNet Denoiser
    """
    def __init__(self, input_channel, output_channel, time_dim, cond_dim = 0, use_attn = False):
        super().__init__()
        self.res1 = ResBlock(input_channel=input_channel, output_channel=output_channel, time_dim=time_dim, cond_dim=cond_dim)
        self.res2 = ResBlock(input_channel=output_channel, output_channel=output_channel, time_dim=time_dim, cond_dim=cond_dim)
        self.attn = SelfAttention1d(output_channel) if use_attn else nn.Identity()
        self.pool = nn.Conv1d(output_channel, output_channel, kernel_size=3, stride = 2, padding=1)

    def forward(self, x, time_emb, cond_emb = None):
        x = self.res1(x, time_emb, cond_emb) 
        x = self.res2(x, time_emb, cond_emb)
        x = self.attn(x)
        skip = x
        x = self.pool(x)
        return x, skip
    
class Up(nn.Module):
    """
    Up sampling for UNet Denoiser
    """
    def __init__(self, input_channel, skip_channel, output_channel, time_dim, cond_dim = 0, use_attn = False):
        super().__init__()
        self.res1 = ResBlock(input_channel + skip_channel, output_channel, time_dim, cond_dim)
        self.res2 = ResBlock(output_channel, output_channel, time_dim, cond_dim)
        self.attn = SelfAttention1d(output_channel) if use_attn else nn.Identity()
        self.upsample = nn.ConvTranspose1d(output_channel, output_channel, kernel_size=4, stride=2, padding=1)

    def forward(self, x, skip, time_emb, cond_emb = None):
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim = 1)
        x = self.res1(x, time_emb, cond_emb)
        x = self.res2(x, time_emb, cond_emb)
        x = self.attn(x)
        self.upsample(x)

        return x