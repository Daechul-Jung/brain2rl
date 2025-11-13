import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.utils.typing import Shape, Sequence


@dataclass(frozen=True)
class TokenGroup:
    """
    A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)
    
    Attributes:
        tokens: torch.tensor array of shape (..., n_tokens, token_dim)
        mask:  torch.tensor array of shape (..., n_tokens) indicating which tokens are valid (1) or padding (0)
    """
    tokens = torch.Tensor
    mask = torch.Tensor

    @classmethod
    def create(cls, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None, **kwargs):
        ## cls is TokenGroup itself
        if mask is None:
            mask = torch.ones((tokens.shape[:-1]), device=tokens.device, dtype=tokens.dtype)
        assert mask.ndim == tokens.ndim - 1

        return cls(tokens, mask, **kwargs)
    
    @classmethod
    def concatenate(cls, group_list: Sequence['TokenGroup'], axis = -2):
        data = torch.concat([t.tokens for t in group_list], dim = axis)
        mask = torch.concat([t.mask for t in group_list], dim = axis + 1)
        return cls(data, mask)