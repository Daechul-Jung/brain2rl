from __future__ import annotations
from abc import ABC, abstractmethod
import logging 
from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributions import Categorical
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.base import TokenGroup
from models.rl.vla.model.components.diffusion import cosine_beta_schedule, create_diffusion_model
from models.rl.vla.model.components.tokenizers import BinTokenizer
from models.rl.vla.model.components.transformer import MAPHead
from models.rl.vla.model.components.unet import ConditionalUnet1D, unet_squaredcos_cap_v2
from models.rl.vla.utils.typing import PRNGkey

def masked_mean(x: torch.Tensor, mask: torch.Tensor):
    mask = torch.broadcast_to(mask, x.shape)
    return torch.mean(x * mask) / torch.clamp(torch.mean(mask), min = 1e-5, max = None)

def continuous_loss(
        pred_value: torch.Tensor,
        ground_truth_value: torch.Tensor,
        mask: torch.Tensor,
        loss_type: str = 'mse'
):
    """
    Args:
        pred_value: shape(batch_dims...)
        ground_truth_value: continuous values with shape (batch_dims...)
        mask: broadcastable to ground_truth
    """

    if loss_type == 'mse':
        loss = torch.square(pred_value - ground_truth_value)

    elif loss_type == 'l1':
        loss = torch.abs(pred_value - ground_truth_value)

    else:
        raise ValueError(f'Invalid loss type: {loss_type}')
    
    loss = masked_mean(loss, mask)
    mse = torch.square(pred_value - ground_truth_value)
    mse = masked_mean(mse, mask)

    return loss, {
        'loss': loss,
        'mse': mse
    }

def discrete_loss(
        discrete_tokenizer: BinTokenizer,
        logits: torch.Tensor,
        ground_truth_value: torch.Tensor,
        mask: torch.Tensor
):
    """
    Args:
        discrete_tokenizer: BinTokenizer to use on ground_truth_value
        logits: shape (batch_dims..., vocab_size)
        ground_truth_value: continuous values in with shape (batch_dims..., )
        mask: broadcastable to ground_truth_value 
    """
    labels = discrete_tokenizer(ground_truth_value) ## Through BinTokenizer, making ground_truth_value into token values
    labels_one_hot = F.one_hot(labels, num_classes=logits.shape[-1]).to(logits.dtype)  ## only corresponding labels would be 1, otherwise 0

    logprobs = F.log_softmax(logits, dim=-1) ## shape of (batch_dims, vocab_size)
    nll = -(logprobs * labels_one_hot).sum(dim = -1) ### only maximum values remain
    loss = masked_mean(nll, mask)

    pred_label = logprobs.argmax(dim=-1)
    acc = masked_mean((pred_label == labels).to(logits.dtype), mask)

    pred_value = discrete_tokenizer.decode(pred_label).to(logits.dtype) ## decode predicted labels to value of probability
    mse = masked_mean(torch.square(pred_value - ground_truth_value), mask)

    return loss, {
        'loss': loss,
        'mse' : mse,
        'accuracy': acc
    }

class ActionHead(ABC, nn.Module):
    """
    Action Prediction modules that take in the transformer token outputs and predict actions 

    Each action head here does chunked action prediction: i.e. at every timestep, it tries to predict the next 
    'action horizon' actions into the future from that timestep. Setting 'action_horizon = 1' corresponds to 
    the typical action prediction setup.  
    """
    @abstractmethod
    def loss(
        self, 
        transformer_outputs: Dict[str, TokenGroup],
        actions: torch.Tensor,
        timestep_pad_mask: torch.Tensor,
        action_pad_mask: torch.Tensor,
        train: bool = True
    ):
        raise NotImplementedError
    
    @abstractmethod
    def predict_action(
        self, 
        transformer_outputs: Dict[str, TokenGroup],
        argmax: bool = False,
        sample_shape: Tuple[int, ...] = (),
        rng: Optional[PRNGkey] = None,
        temperature: float = 1.0,
        train: bool = False,
        embodiment_action_dim: Optional[int] = None
    ):
        raise NotImplementedError
    

class ContinuousActionHead(ActionHead):
    """
    Predicts continuous actions (as opposed to discretized)

    Continuous actions are predicted y tanh squashing the model output to [-max_action, max_action], and then
    optimized using a standard regression loss. 
    token_group.tokens: [Batch, horizon, n_tokens, token_dim ]

    You may create an embedding by either mean-pooling across tokens (use_map = False) or using multi-head
    attention pooling (use_map = True). It is recommended to use MAP when decoding from the observation token stream.
    """

    def __init__(
            self,
            readout_key: str,
            use_map: bool = False,
            action_horizon: int = 1,
            action_dim: int = 7,
            max_action: float = 5.0,
            loss_type: str = 'mse'
    ):
        super().__init__()
        self.readout_key = readout_key
        self.use_map = use_map
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.max_action = max_action
        self.loss_type = loss_type

        if use_map:
            self.map_head = MAPHead(num_readout=1)
        self.mean_proj = nn.Linear(-1, -1, bias=True)
        self._built = False

    def _build(self, token_dim):
        self.mean_proj = nn.Linear(token_dim, self.action_horizon * self.action_dim)
        nn.init.xavier_uniform_(self.mean_proj.weight)
        nn.init.zeros_(self.mean_proj.bias)
        self._built = True

    def forward(self, transformer_outputs: Dict[str, TokenGroup], train: bool = True):
        ...