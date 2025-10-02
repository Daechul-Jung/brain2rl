import torch
import math


def timestep_embedding(t: torch.LongTensor, dim: int):
    device = t.device
    half = dim // 2

    freqs = torch.exp(-torch.arange(half, device=device) * math.log(10000.0) / half)  ## exp( (1,..., half) * log(10000) / half) values are from 0 to approximately 10
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim = -1) 

    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb

