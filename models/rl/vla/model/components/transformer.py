# adapted from https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py

from typing import Callable, Optional
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.base import TokenGroup
from models.rl.vla.utils.typing import Dtype, PRNGkey, Shape, Union

class AddPositionEmbs(nn.Module):
    """
    Adds learned positional embeddings to the inputs

    Attributes: 
        posemb_init: positional embedding initializer
    """
    def __init__(self, posemb_init:Callable[[PRNGkey, Shape, Dtype], torch.Tensor]):
        super().__init__()

        self.posemb_init = posemb_init
        
    def forward(self, inputs: torch.Tensor):
        assert inputs.ndim == 3, (
            "Number of dimensions should be 3," " but it is: %d" % inputs.ndim 
        )

        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = torch.empty(1, inputs.shape[1], inputs.shape[2], device = inputs.device, dtype = inputs.dtype)
        self.posemb_init(pe)
        self.pos_embedding = nn.Parameter(pe)

        return inputs + self.pos_embedding
    

class MlpBlock(nn.Module):
    """
    Transformer MLP / feed-forward block.
    """
    def __init__(self, mlp_dim: int, 
                 dtype: Dtype = torch.float32, 
                 out_dim: Optional[int] = None,
                 dropout_rate: float = 0.1,
                 kernel_init: Callable[[PRNGkey, Shape, Dtype], torch.Tensor]= nn.init.xavier_uniform(),
                 bias_init: Callable[[PRNGkey, Shape, Dtype], torch.Tensor]= nn.init.normal(std=1e-6)):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dtype = dtype
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(-1, -1)
        self.fc2 = nn.Linear(-1, -1)

    def _build(self, D_in: int):
        """
        build neural network necessary for MlpBlock
        """
        self.fc1 = nn.Linear(D_in, self.mlp_dim)
        self.fc2 = nn.Linear(self.mlp_dim, self.out_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias)
        nn.init.normal_(self.fc2.bias)

    def forward(self, inputs, *, deterministic: bool):
        """
        Applies Transformer MlpBlock module
        Assume input shape is [B, T, D]
        """
        D_in = inputs.shape[-1]
        self._build(D_in)
        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.dropout(x) if not deterministic else x
        x = self.fc2(x)
        x = self.dropout(x) if not deterministic else x

        return x
    

class MAPHead(nn.Module):
    """
    Multihead Attention Pooling.

    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """

    def __init__(self, mlp_dim: int, num_heads: int = 8, num_readout: int = 1, dropout: float = 0.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_readout = num_readout
        self.attn = None
        self.probe: Optional[nn.Parameter] = None
        self.layerNorm = nn.LayerNorm(-1)
        self.mlp: Optional[MlpBlock] = None
        self.dropout = dropout


    def _build(self, d_model: int):
        """
        Build lazy model 
        """
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=self.num_heads, dropout=self.dropout, batch_first=True)
        self.probe = nn.Parameter(torch.empty(1, self.num_readout, d_model))
        nn.init.xavier_uniform_(self.probe)
        self.layerNorm = nn.LayerNorm(d_model)
        self.mlp = MlpBlock(self.mlp_dim, out_dim=d_model, dropout_rate=self.dropout)

    def forward(self, x: Union[torch.Tensor, TokenGroup], train = True):
        if isinstance(x, TokenGroup):
            tokens, mask = x.tokens, x.mask
        else:
            mask = None 

        *batch_dims, T, D = tokens.shape
        tokens = tokens.reshape(-1, T, D)
        self._build(D)

        batch_size = tokens.shape[0]


