import logging
import re
from typing import Dict, Optional, Sequence
import os, sys

import torch
import torch.nn as nn
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.vla.model.components.transformer import MAPHead, AddPositionEmbs
from models.rl.vla.utils.spec import ModuleSpec
from models.rl.vla.model.components.base import TokenGroup


EPS = 1e-6


def generate_proper_pad_mask(
        tokens: torch.Tensor,
        pad_mask_dict: Optional[Dict[str, torch.Tensor]],
        keys: Sequence[str]
) -> torch.Tensor:
    if pad_mask_dict is None:
        logging.warning("No pad_mask_dict found. Nothing will be masked")
        return torch.ones(tokens.shape[:-1])
    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}"
            "Nothing will be masked"
        )
        return torch.ones(tokens.shape[:-1])
    
    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], dim = -1)
    pad_mask = torch.any(pad_mask, dim = -1)
    pad_mask = pad_mask.to(dtype = tokens.dtype)

    return pad_mask 


class TokenLearner(nn.Module):
    """
    Learns to map fixed-length sequence of tokens into specified number of tokens

    Args:
        num_tokens(int): Number of output tokens
        bottleneck_dim(int): Size of hidden layers of the mapping MLP
        dropout_rate(float): Rate of dropout applied in the mapping MLP. Default to no dropout
    """

    def __init__(self, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.map_head = MAPHead(num_readout=num_tokens)
        self.layerNorm = nn.LayerNorm(-1)
        self._lazy_build = False
        self.pos_emb = AddPositionEmbs(posemb_init=0.02)

    def _build(self, token_dim: int):
        self.layerNorm(token_dim)
        self._lazy_build = True

    def forward(self, inputs: torch.Tensor, train: bool = True):
        *_, time_dim, token_dim = inputs.shape
        if not self._lazy_build:
            self._build(token_dim=token_dim)
        x = self.pos_emb(inputs)
        x = self.layerNorm(x)
        return self.map_head(x, train)
    

def regex_match(regex_keys, x):
    return any([re.match(r_key, x) for r_key in regex_keys])

def regex_filter(regex_keys, xs):
    return list(filter(lambda x: regex_match(regex_keys, x), xs))

class ImageTokenizer(nn.Module):
    """
    Image tokenizer that encodes image stack into tokens with optional FiLM conditioning

    Args:
        encoder(ModuleSpec): Encoder classes
        use_token_learner(bool): whether to use token learner. Default to False
        num_tokens(int): Number of output tokens, only enforced when use_token_learner is True
        obs_stack_keys(Sequence[str]): Which spatial observation inputs get stacked for encoder input. Support regex
        task_stack_keys(Sequence[str]): Which spatial task inputs get stacked for encoder input. Support regex
        task_film_keys(Sequence[str]): Which non-spatial task keys get passed into FiLM conditioning. Support regex 
    """
    def __init__(self, 
                 encoder: ModuleSpec, 
                 use_token_learner: bool = False, 
                 num_tokens: int = 8, 
                 conditioning_type:str = None, 
                 obs_stack_keys: Sequence[str] = ('image_.*', 'depth_.*'),
                 task_stack_keys: Sequence[str] = tuple(),
                 task_film_keys: Sequence[str] = tuple()
                ):
        super().__init__()
        self.encoder = encoder
        self.use_token_learner = use_token_learner
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys


    def forward(self, observation: torch.Tensor, 
                tasks = None, train: bool = True):
        
        def extract_inputs(keys, inputs, check_spatial = False):
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])            
            return torch.concatenate(extracted_outputs, dim = -1)