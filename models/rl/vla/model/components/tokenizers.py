import logging
import re
from typing import Dict, Optional, Sequence
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
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
    

class LanguageTokenizer(nn.Module):
    """
    Language Tokenizer that embeds text inputs IDs into continuous language embeddings. Support pre-trained Hugging Face model

    Args:
        num_tokens (int): Number of output tokens (not enforced)
        encoder (str, optional): Optional HuggingFace Automodel name for encoding input IDs
        finetune_encoder (bool, optional): Optional finetune last layers of the language model
    """
    def __init__(self, 
                 encoder: str = None,
                 finetune_encoder: bool = False,
                 proper_pad_mask: bool = True):
        super().__init__()
        self.encoder = encoder
        self.finetune_encoder = finetune_encoder
        self.proper_pad_mask = proper_pad_mask
        self.hf_model = None

        if self.encoder is not None:
            try:
                from transformers import AutoConfig, AutoModel, T5EncoderModel

                config = AutoConfig.from_pretrained(self.encoder)
                if "t5" in self.encoder:
                    self.hf_model = T5EncoderModel(config)
                else:
                    self.hf_model = AutoModel(config)

            except Exception as e:
                raise RuntimeError(f'Failed to initialize HF model {encoder}: {e}')
    

    def forward(self, observations: Dict[str, torch.Tensor], tasks = None, train: bool = True):
        tasks = {} if tasks is None else tasks
        if "language_instruction" not in tasks:
            logging.warning("No language inputs found. Skipping tokenizer entirely")
            assert self.proper_pad_mask, "Cannot skip unless using proper pad mask"
            return None 
        
        if not isinstance(tasks['language_instruction'], (torch.Tensor, torch.Tensor)):
            assert (
                self.encoder is not None
            ), "Received language tokens but no encoder specified"
            outputs = self.hf_model(**{K: (V if isinstance(V, torch.Tensor) else torch.tensor(V)) for K, V in tasks['language_instruction']})
            tokens = outputs.last_hidden_state
        else:
            tokens = tasks['language_instruction']
            if tokens.ndim == 2:
                tokens = tokens[:, None, :]

        if not self.finetune_encoder and tokens.requires_grad:
            tokens = tokens.detach()

        ### Build pad mask
        if self.proper_pad_mask:
            pad_mask = generate_proper_pad_mask(
                tokens, pad_mask_dict=tasks.get('pad_mask_dict', None), keys = ('language_instruction',)
            )
        else:
            pad_mask = torch.ones(tokens.shape[:-1], dtype = tokens.dtype, device = tokens.device)

        return TokenGroup(tokens, pad_mask)
    

def _ndtri(p: torch.Tensor):
    """
    Inversed standard Normal CDF 
    """
    import math
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)

class BinTokenizer(nn.Module):
    """
    Tokenize continuous inputs via dimension-wise binning in given range
    Used for discrete loss and decode the tokens 

    Args:
        n_bins (int): Number of discrete bins per dimension
        bin_type (str): Type of binning ['uniform', 'normal' = Gaussian]
        low (float): Lower bound for bin range
        high (float): Upper bound for bin range
    """
    def __init__(self, n_bins: int = 256, bin_type: str = 'uniform', low: float = 0.0, high: float = 1.0):
        super().__init__()
        self.n_bins = n_bins
        self.bin_type = bin_type
        self.low = low
        self.high = high
        self.eps = 1e-6
        self.register_buffer('thresholds', torch.empty(0), persistent=False)

    def _build_thresholds(self, device, dtype):
        if self.bin_type == "uniform":
            th = torch.linspace(self.low, self.high, self.n_bins + 1, device=device, dtype=dtype)
        elif self.bin_type == "normal":
            p = torch.linspace(self.eps, 1 - self.eps, self.n_bins + 1, device=device, dtype=dtype)
            th = _ndtri(p)
        else:
            raise ValueError(f"Binning type {self.bin_type} not supported.")
        self.thresholds = th

    def forward(self, inputs: torch.Tensor):
        if self.thresholds.numel() == 0 or self.thresholds.device != inputs.device or self.thresholds.dtype != inputs.dtype:
            self._build_thresholds(inputs.device, inputs.dtype)

        x = inputs
        if self.bin_type == "uniform":
            x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x.unsqueeze(-1) 

        left = self.thresholds[:-1]
        right = self.thresholds[1:]
        token_bool = (x >= left) & (x < right)
        tokens = token_bool.max(dim=-1).indices  # argmax over bins
        return tokens  # (...,)

    def decode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Map discrete bin indices back to bin midpoints.
        inputs (torch.Tensor): predicted labels with shape of (batch, )
        """
        if self.thresholds.numel() == 0:
            self._build_thresholds(inputs.device, torch.float32)
        bin_avgs = (self.thresholds[1:] + self.thresholds[:-1]) / 2
        return bin_avgs[inputs]
    

class LowdimObsTokenizer(BinTokenizer):
    """
    Tokenizer for non-spatial observation 
    Optionally discretizes into bins per dimension (see BinTokenzier)

    Args:
        obs_keys(Sequence[str]): List of non-spatial keys to concatenate & tokenize. support regex
        discretize (bool): If True, discretizes inputs per dimension, see BinTokenizer
    """

    def __init__(self, obs_keys: Sequence[str] = tuple(), discretize: bool = False, proper_pad_mask: bool = True, **bin_kwargs):
        super().__init__(**bin_kwargs)
        self.obs_keys = tuple(obs_keys)
        self.discretize = discretize
        self.proper_pad_mask = proper_pad_mask

    def forward(self, observations: Dict[str, torch.Tensor], *unused_args, **unused_kwargs):
        assert self.obs_keys, "Need to specify observation keys to tokenize."

        if len(regex_filter(self.obs_keys, sorted(observations.keys()))) == 0:
            logging.warning(
                f'No observation inputs matching {self.obs_keys} were found.'
                'Skipping Tokenizer entirely.'
            )
            assert self.proper_pad_mask, 'Cannot skip unless using proper pad mask'
            return None 
        tokenizer_inputs = []
        for o_key in self.obs_keys:
            for key in filter(re.compile(o_key).match, sorted(observations.keys())):
                assert(
                    len(observations[key].shape) == 3
                ), f'Only support non-spatial inputs but {key} has shape {observations[key].shape}'
                tokenizer_inputs.append(observations[key])
        tokenizer_inputs = torch.concat(tokenizer_inputs, dim = -1)
        
        if self.discretize:
            tokenizer_inputs = super().forward(tokenizer_inputs)
            tokens = F.one_hot(tokenizer_inputs, num_classes=self.n_bins).to(tokenizer_inputs.dtype)
        else:
            tokens = tokenizer_inputs.unsqueeze(-1)

        masks = torch.ones(tokens.shape[:-1], dtype=tokens.dtype, device = tokens.device)
        return TokenGroup(tokens, masks)
