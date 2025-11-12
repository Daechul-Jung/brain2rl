import torch
import torch.nn as nn
from dataclasses import dataclass
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.utils.typing import Shape, Sequence
@dataclass

class TokenGroup:
    """
    A group of tokens that have semantic meaning together (e.g. the tokens for a single observation)
    
    Attributes:
        tokens: torch.tensor array of shape (..., n_tokens, token_dim)
        mask:  torch.tensor array of shape (..., n_tokens) indicating which tokens are valid (1) or padding (0)
    """
    