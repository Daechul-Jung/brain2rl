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

        pe = torch.empty(1, inputs.shape[1], inputs.shape[2], device = inputs.device, dtype = inputs.dtype)
        self.posemb_init(pe)
        self.pos_embedding = nn.Parameter(pe)

        return inputs + self.pos_embedding
    

class MlpBlock(nn.Module):
    """
    Transformer MLP / feed-forward block. 
    Fully connected layers -> gelu -> dropout -> fully connected layers -> dropout 
    """
    def __init__(self, mlp_dim:Optional[int], 
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
        self._lazy_build = False

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
        if not self._lazy_build:
            self._build(D_in)
            self._lazy_build = True
        x = self.fc1(inputs)
        x = F.gelu(x)
        x = self.dropout(x) if not deterministic else x
        x = self.fc2(x)
        x = self.dropout(x) if not deterministic else x

        return x
    

class MAPHead(nn.Module):
    """
    Multihead Attention Pooling.
    Attention -> layer norm -> add attention output and layer normed attention output and then reshape
    From https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py
    """

    def __init__(self, mlp_dim: Optional[int] = None, num_heads: int = 8, num_readout: int = 1, dropout: float = 0.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.num_readout = num_readout
        self.attn = None
        self.probe: Optional[nn.Parameter] = None
        self.layerNorm = nn.LayerNorm(-1)
        self.mlp: Optional[MlpBlock] = None
        self.dropout = dropout
        self._lazy_build = False

    def _build(self, d_model: int):
        """
        Build lazy model 
        """
        if self.mlp_dim is None:
            self.mlp_dim = d_model * 4
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
        ### Expecting x(token shape) would be (batch_size, horizon, num_tokens, token_dim)
        ## T: horizon(time_dim), D: token_dim
        *batch_dims, T, D = tokens.shape
        
        tokens = tokens.reshape(-1, T, D)
        if not self._lazy_build:
            self._build(D)
            self._lazy_build = True

        batch_size = tokens.shape[0]

        probe = self.probe.expand(batch_size, -1, -1)
        key_padding_mask = None
        
        if mask is not None:
            mask = mask.reshape(-1, T)
            key_padding_mask = (mask == 0)

        out, _ = self.attn(query = probe, ## [B, num_readout, Token_dim]
                           key = tokens, ## [B, time_dim, Token_dim]
                           value = tokens, ## [B, time_dim, Token_dim]
                           key_padding_mask = key_padding_mask,  ## 
                           need_weight = False)  ### Return [B, R, D]
        y = self.layerNorm(out)
        out = out + self.mlp(y, deterministic = not train)
        
        ### Reshape into [..., num_readouts, Token_dim]
        out = out.reshape(*batch_dims, self.num_readout, D)

        return out
    

class Encoder1DBlock(nn.MOdule):
    """
    Transformer encoder layer
    Attributes:
        inputs: [B, Time_dim, Token_dim]
        mlp_dim: dimension of the mlp on top of attention block
        dtype: the dtype of the computation (default: float32)
        dropout_rate: dropout rate
        attention_dropout_rate: dropout for attention heads
        deterministic: bool, deterministic or not (to apply dropout)
        num_heads: number of heads in nn.MultiHeadAttention
        attention mask: Optional [B, Time_dim] with 1 = valid, 0 = pad(key padding mask)
    """

    def __init__(self, mlp_dim: int, num_heads: int, dtype: Dtype = torch.float32, dropout_rate: float = 0.1, attention_dropout_rate: float = 0.1):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.attention_dropout_rate = attention_dropout_rate
        self.dtype = dtype
        self.layer_norm = nn.LayerNorm()
        self.attn = nn.MultiheadAttention()
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_res = nn.Dropout(attention_dropout_rate)
        self.layer_norm2 = nn.LayerNorm()
        self.mlp = MlpBlock(mlp_dim=mlp_dim, dtype = dtype, out_dim = None)
        self._lazy_build = False

    def _build(self, d_model):
        """
        Lazy build for NNs with the input of token dimension
        """
        self.layer_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=self.num_heads,
            dropout=self.attention_dropout_rate,
            batch_first=True
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = MlpBlock(self.mlp_dim, dtype= self.dtype, out_dim=d_model, dropout_rate=self.dropout_rate)

    def forward(self, inputs: torch.Tensor, attention_mask: Optional[torch.Tensor], *, deterministic: bool):
        """
        Applies Encoder1DBlock module used for Transformer

        LayerNorm -> attention -> dropout -> (attn_out + original input) -> second layerNorm -> mlp block
        Args:
            inputs: inputs to the layer
            deterministic: Dropout will not be applied when set to True

        Returns:
            outputs after transformer encoder block 
        """
        assert inputs.ndim == 3, f'Expected (batch. seq, hidden) got {inputs.shape}'
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        B, T, D = inputs.shape
        if not self._lazy_build:
            self._build(D)
            self._lazy_build = True

        x = self.layer_norm(inputs)
        attn_out, _ = self.attn(query = x, key = x, value = x, key_padding_mask = key_padding_mask, need_weights = False, attn_mask = attention_mask)
        attn_out = self.dropout_res(attn_out) if not deterministic else attn_out
        x = attn_out + inputs

        y = self.layer_norm2(x)
        y = self.mlp(y,deterministic)

        return x + y
    

class Transformer(nn.Module):
    """
    Transformer model Enccoder for sequence to sequence 

    Attributes:
        num_layers: number of layers
        mlp_dim: dimension of the mlp on top of attention block 
        num_heads: Number of heads in nn.MultiheadAttention
        dropout_rate: dropout_rate
        attention_dropout_rate: dropout_rate in self attention
    """

    def __init__(self, 
                 num_layers: int, 
                 mlp_dim: int, 
                 num_heads: int,
                 dropout_rate: float,
                 attention_dropout_rate: float,
                 add_position_embedding: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.add_position_embedding = add_position_embedding
        self.pos_embedding = AddPositionEmbs(0.02) if add_position_embedding else None
        self.dropout = nn.Dropout(dropout_rate)
        ### Using Encoder1DBlock for sequential data
        self.blocks = nn.ModuleList([
            Encoder1DBlock(mlp_dim=mlp_dim, num_heads=num_heads, dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate) for _ in range(self.num_layers)
        ])
        self.layer_norm = nn.LayerNorm(-1)
        self._norm_build = False

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor], *, train: bool):
        """
        Applies Transformer model on the inputs

        Create moduleList of Encoder1DBlock(having multihead attention in here)
        positional embedding -> block passing -> layer norm
        Args:
            x: Inputs to the layer 
            train: Set to True when training
        """
        assert x.ndim == 3, f"Must get [B, T, D] but got {x.shape}"

        B, T, D = x.shape
        if self.add_position_embedding:
            x = self.pos_embedding(x)
            x = self.dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask, deterministic = not train)

        if not self._norm_build:
            self.layer_norm = nn.LayerNorm(D)
            self._norm_build = True

        x = self.layer_norm(x)

        return x

def common_transformer_sizes(transformer_size: str):
    """
    Args:
        transformer_size (str): The size of the transformer. One of "dummy", "vanilla", "vit_s", "vit_b", "vit_l", "vit_h"

    Returns:
            token_embedding_size (int): The size of the token embeddings
            transformer_kwargs (dict): The kwargs to pass to the transformer

    """
    assert transformer_size in [
        "dummy",
        "vanilla",
        "vit_t",
        "vit_s",
        "vit_b",
        "vit_l",
        "vit_h",
    ]
    default_params = {
        "attention_dropout_rate": 0.0,
        "add_position_embedding": False,
    }

    TRANSFORMER_SIZES = {
        "dummy": dict(
            num_layers=1,
            mlp_dim=256,
            num_attention_heads=2,
            dropout_rate=0.1,
        ),
        "vanilla": dict(
            num_layers=4,
            mlp_dim=1024,
            num_attention_heads=8,
            dropout_rate=0.1,
        ),
        "vit_t": dict(
            num_layers=12,
            mlp_dim=768,
            num_attention_heads=3,
            dropout_rate=0.0,
        ),
        "vit_s": dict(
            num_layers=12,
            mlp_dim=1536,
            num_attention_heads=6,
            dropout_rate=0.0,
        ),
        "vit_b": dict(
            num_layers=12,
            mlp_dim=3072,
            num_attention_heads=12,
            dropout_rate=0.0,
        ),
        "vit_l": dict(
            num_layers=24,
            mlp_dim=4096,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
        "vit_h": dict(
            num_layers=32,
            mlp_dim=5120,
            num_attention_heads=16,
            dropout_rate=0.1,
        ),
    }

    TOKEN_DIMS = {
        "dummy": 256,
        "vanilla": 256,
        "vit_t": 192,
        "vit_s": 384,
        "vit_b": 768,
        "vit_l": 1024,
        "vit_h": 1280,
    }

    return TOKEN_DIMS[transformer_size], {
        **default_params,
        **TRANSFORMER_SIZES[transformer_size],
    }