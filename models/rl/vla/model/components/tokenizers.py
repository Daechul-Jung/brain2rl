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
    ## check every keys are in the pad mask key dictionary
    if not all([key in pad_mask_dict for key in keys]):
        logging.warning(
            f"pad_mask_dict missing keys {set(keys) - set(pad_mask_dict.keys())}"
            "Nothing will be masked"
        )
        return torch.ones(tokens.shape[:-1])
    ## Stack over the last dimension
    pad_mask = torch.stack([pad_mask_dict[key] for key in keys], dim = -1)
    ## make it as bool over the last dimension
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
        ## Do positional embeddin -> layer norm -> multihead attention
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
                 task_film_keys: Sequence[str] = tuple(),
                 proper_pad_mask: bool = True
                ):
        super().__init__()
        self.encoder = encoder
        self.use_token_learner = use_token_learner
        self.num_tokens = num_tokens
        self.conditioning_type = conditioning_type
        self.obs_stack_keys = obs_stack_keys
        self.task_stack_keys = task_stack_keys
        self.task_film_keys = task_film_keys
        self.proper_pad_mask = proper_pad_mask

        self.token_learner = TokenLearner(num_tokens) if use_token_learner else None

    def forward(self, observations, 
                tasks = None, train: bool = True):
        """
        Sequences
        1. From observation stack keys, do regex filter -> obs_stack_key
        2. extract input from observation stack key -> encoder input
        3. From task stack keys, with observation, do regex filter and create task_stack_key and extract input based on tasks key -> task_inputs
        4. concatenate enc_input(task stack keys) and task input -> enc_input
        5. If plan to use film conditioning, from task_film_keys with regex filter, concatenate over last dimension
        6. enc_input(concatenated with observation and task) used as image and film_conditioning used as cond_var(conditioning variables)
        7. Put it into encoder and reshape output based on dimensions and finally put it into TokenGroup
        """
        def extract_inputs(keys, inputs, check_spatial = False):
            """
            Extract inputs based on keys and concatenate over the last dimension 
            """
            extracted_outputs = []
            for key in keys:
                if check_spatial:
                    assert len(inputs[key].shape) >= 4
                extracted_outputs.append(inputs[key])            
            return torch.concatenate(extracted_outputs, dim = -1)
        
        obs_stack_keys = regex_filter(self.obs_stack_keys, sorted(observations.keys()))
        if len(obs_stack_keys) == 0:
            logging.info(
                f'No image inputs matching {self.obs_stack_keys} were found'
                'Skipping tokenizer entirely'
            )
            assert self.proper_pad_mask, "Cannot skip unless using proper_pad_mask"
            return None
        
        ## Stack all spatial observation and task inputs
        enc_inputs = extract_inputs(obs_stack_keys, observations, True)
        if self.task_stack_keys:
            needed_task_key = regex_filter(self.task_stack_keys, observations.keys())
            ## if any task inputs are missing, replace with zero padding (TODO: more flexible)
            for key in needed_task_key:
                if key not in tasks:
                    logging.info(
                        f'No task inputs matching {key} were found. Replacing with zero padding'
                    )
                    tasks[key] = torch.zeros_like(observations[key][:, 0]) ## [B, H, W, C]
            task_stack_keys = regex_filter(self.task_stack_keys, sorted(tasks.keys()))
            if len(task_stack_keys) == 0:
                raise ValueError(
                    f'No task inputs are matching {self.task_stack_keys} were found'
                )
            task_inputs = extract_inputs(task_stack_keys, tasks, True)
            task_inputs = task_inputs[:, None].repeat(enc_inputs.shape[1], dim = 1)

            enc_inputs = torch.concatenate([enc_inputs, task_inputs], dim = -1)

        b, time_dim, h, w, c_total = enc_inputs.shape 
        imgs = enc_inputs.permute(0, 1, 4, 2, 3).reshape(b * time_dim, c_total, h, w)

        ## None spatial FiLM encoding 
        encoder_kwargs = {}
        if self.task_film_keys:
            film_inputs = torch.cat([tasks[k] for k in regex_filter(self.task_film_keys, tasks.keys())], dim = -1)
            if film_inputs.ndim == 2:
                film_inputs = film_inputs[:, None].repeat(1, time_dim, 1) ## (Batch, time_dim, D_film)

            encoder_kwargs['cond_var'] = film_inputs.reshape(b * time_dim, -1)

        ## Run visual encoder, Encode -> tokens
        image_tokens = self.encoder(imgs, **encoder_kwargs)

        if image_tokens.ndim == 4:
            ## [B * time_dim, Channel', H', W'] -> flatten spatial to time_dim
            image_tokens = image_tokens.permute(0, 2, 3, 1).reshape(image_tokens.shape[0], -1, image_tokens.shape[1])
        
        ## Unfold time back: (B, time_dim, time_tokens , token_dim)
        time_tokens = image_tokens.shape[1]
        token_dim = image_tokens.shape[2]
        image_tokens = image_tokens.reshape(b, time_dim, time_tokens, token_dim)

        if self.use_token_learner:
            image_tokens = self.token_learner(image_tokens, train= train)

        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens = image_tokens,
                pad_mask_dict=observations.get('pad_mask_dict', None),
                keys = obs_stack_keys
            )
        else:
            pad_mask = torch.ones(image_tokens, pad_mask)

        return TokenGroup(image_tokens, pad_mask)