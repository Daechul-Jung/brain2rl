from enum import Enum
from fnmatch import fnmatch
from dataclasses import dataclass
import logging 
from typing import Any, Dict, Mapping, Sequence, Tuple, Union, Optional, List

import einops
import torch
import torch.nn as nn
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rl.vla.model.components.base import TokenGroup
from models.rl.vla.model.components.transformer import Transformer

class AttentionRule(Enum):
    """
    Enum describing when to attend to another token group 
    For most use cases, you should use WhenToAttend.CAUSAL or WhenToAttend.NEVER
    """
    NEVER = 'never'
    CAUSAL = 'other.timestep <= self.timestep'
    CURRENT = 'other.timestep == self.timestep'
    STRICT_PAST = 'other.timestep < self.timestep'
    ALL = 'all' ## Breaks causal structure! Be careful


@dataclass(frozen = True)
class PrefixGroup(TokenGroup):
    """
    A group of tokens that will be at the beginning of the token sequence (e.g. task tokens)

    Add a name identifying the group, and a dictionary indicating what groups it should attend to

    name(str): Name of the group, which other groups will look at when deciding whether to attend to this group
    attention_rulse (Dict[str, AttentionRule]): A dictionary of {pattern: AttentionRule} where the attention rule 
        is recovered by fnmatch-ing the name of the other group unitl a match is found (or the end)
    """
    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        assert (
            len(self.tokens.ndim) == 3
        ), 'Prefixgroup tokens must be (batch, n_tokens, Token_dim)'
        assert len(self.mask.ndim) == 2, 'PrefixGroup mask must be (batch,  n_tokens)'

@dataclass(frozen = True)
class TimestepGroup(TokenGroup):
    """
    A group of tokens that is repeated for each timestep. (e.g. observation tokens)

    See PrefixGroup for details on the name and attention_rules fields.
    """
    name: str
    attention_rules: Mapping[str, AttentionRule]

    def __post_init__(self):
        assert (
            len(self.tokens.ndim) == 4
        ), 'TimestepGroup tokens must be (batch, Time_dim, n_tokens, Token_dim)'
        assert len(self.mask.ndim) == 3, 'TimestepGroup tokens must be (batch, Time_dim, n_tokens, Token_dim)'

def find_match(pattern_dict: Dict[str, Any], name: str, default: Any):
    """
    Find the first pattern in the dictionary, or return the default value 
    """

    for pattern, value in pattern_dict.items():
        if fnmatch(name, pattern):
            return value 
    return default 

@dataclass(frozen = True)
class TokenMetadata:
    """
    Attention mask logic supported by AttentionRule. Note that all tokens with the same 
    group at the same timestep always attend to each other unless you explicitly have 
    attention_rules[self.name] = AttentionRule.NEVER
    """

    name: str
    timestep: int # -1 for prefix token
    attention_rules: Mapping[str, AttentionRule]

    @classmethod
    def create(cls, group: Union[PrefixGroup, TimestepGroup], timestep: int):
        return cls(
            name = group.name, 
            timestep=timestep,
            attention_rules = group.attention_rules
        )
    
    def should_attend_to(self, other_metadata: "TokenMetadata"):
        """
        To check whether the token should attend or not 
        Everythings are boolean
        """
        attention_rule = find_match(
            self.attention_rules, other_metadata.name, AttentionRule.NEVER
        )

        if attention_rule == AttentionRule.CAUSAL:
            return other_metadata.timestep <= self.timestep
        elif attention_rule == AttentionRule.CURRENT:
            return other_metadata.timestep == self.timestep
        elif attention_rule == AttentionRule.STRICT_PAST:
            return other_metadata.timestep < self.timestep
        elif attention_rule == AttentionRule.ALL:
            return True
        elif attention_rule == AttentionRule.NEVER:
            return False
        else:
            raise ValueError(f"Invalid attention rule: {attention_rule}")

def split_tokens(x: torch.Tensor, n_tokens_per_group: Sequence[int], axis: int):
    if sum(n_tokens_per_group) == 0:
        return []
    idxs = np.cumsum(n_tokens_per_group)[:-1]
    return torch.tensor_split(x, indices=idxs.tolist(), dim = axis)


class BlockTransformer(nn.Module):
    """
    A transformer that acts on multiple groups of tokens, which may attend to each other (in complex patterns).
    """
    def __init__(self, transformer_kwargs: Dict, enforce_causal: bool = True, use_correct_attention: bool = False):
        super().__init__()
        self.transformer_kwargs = transformer_kwargs
        self.enforce_causal = enforce_causal
        self.use_correct_attention = use_correct_attention
        #self.transformer: Optional[nn.Module] = None
        self.transformer = Transformer(transformer_kwargs)

    def forward(self, 
                prefix_groups: Sequence[PrefixGroup],
                timestep_groups: Sequence[TimestepGroup],
                train: bool,
                verbose: bool = False
                ):
        """
        Args:
            prefix_groups: A list of PrefixGroup objects
                each group has 
                    - tokens with shape (batch, n_tokens, token_embedding_size)
                    - mask with shape (batch, n_tokens) indicating which tokens are padding 
                    - name identifying the group 
                    - dictionary of attention patterns dictating which other groups it will attend to.

            timestep_groups: A list of TimestepGroup objects
                each group has 
                    - tokens with shape (batch, time_dim, n_tokens, token_embedding_size)
                    - mask with shape (batch, time_dim, token_embedding_size)
                    - name identigying the group
                    - dictionary of attention patterns dictating which other groups it will attend to. 
            train: whether to use dropout

        Returns:
            prefix_outputs: A list of PrefixGroup objects containing the output embedding for each token group.
            timestep_outputs: A list of TimestepGroup objects containing the output embedding for each token group.  
        """
        if verbose:
            self.pretty_print_attention_mask(prefix_groups, timestep_groups)

        horizon  = timestep_groups[0].tokens.shape[1]  ## time dimension
        assert all([group.tokens.shape[1] == horizon for group in timestep_groups])

        token_dim = timestep_groups[0].tokens.shape[-1] ## token dimension
        assert all([group.tokens.shape[-1] == token_dim for group in timestep_groups])
        assert all([group.tokens.shape[-1] == token_dim for group in prefix_groups])

        ## Assemble input tokens (batch, total_tokens, token_embedding_size)
        input_tokens = self.assemble_input_tokens(prefix_groups, timestep_groups)

        ## Create correct attention mask for transformer using group attention rules and masks 
        ## shape: (batch, 1, total_tokens, total_tokens). This is shape of combination of causal and pad_mask 
        attention_mask = self.generate_attention_mask(prefix_groups, timestep_groups)
        attention_mask = attention_mask.squeeze(1).bool()
        ## Sows attention mask for ease of retrieval when debugging -> This part exists in original code(flax) but not in pytorch

        if self.transformer is None:  ## Need to figure out how to replace this code
            self.transformer = Transformer(**self.transformer_kwargs)

        output = self.transformer(input_tokens, attention_mask, train = train)

        all_prefix_outputs, all_timestep_outputs = self.split_output_tokens(
            output, prefix_groups, timestep_groups
        )

        return all_prefix_outputs, all_timestep_outputs
    
    def assemble_input_tokens(self, prefix_groups: Sequence[PrefixGroup], timestep_groups: Sequence[TimestepGroup]):
        """
        - Concatenate all timestep tokens together 
        - Fold horizon dim into token sequence dim
        - Prepend task tokens
        
        prefix_groups: shape of (batch, n_tokens, token_embedding_size)
        timestep_groups: shape of (batch, horizon, n_tokens, token_embedding_size )

        Returns:
            tokens: A tensor of shape (batch, total_tokens, token_embedding_size)
        """
        if len(prefix_groups)> 0: 
            all_prefix_tokens = torch.concatenate([group.tokens for group in prefix_groups], dim = 1) 
            ## This is shape of [batch, total_prefix_tokens, token_dim]
        else:
            batch = prefix_groups[0].tokens.shape[0]
            token_dim = prefix_groups[0].tokens.shape[-1]
            all_prefix_tokens = torch.zeros((batch, 0, token_dim), device=prefix_groups[0].tokens.device, dtype=prefix_groups[0].tokens.dtype)
        
        all_timestep_tokens = torch.concatenate([group.tokens for group in timestep_groups], dim = 2)
        batch, time_dim, n_tokens, token_dim = all_timestep_tokens.shape
        all_timestep_tokens = all_timestep_tokens.reshape(batch, time_dim * n_tokens, token_dim) ## [batch, total_tokens, token_dim] 

        tokens = torch.concatenate([all_prefix_tokens, all_timestep_tokens], dim = 1) ## [batch, total_prefix + total_timestep, token_dim]
        return tokens
    
    def split_output_tokens(self, 
                            output_tokens: torch.Tensor,
                            prefix_groups: Sequence[PrefixGroup],
                            timestep_groups: Sequence[TimestepGroup]
                            ):
        """
        prefix_groups: shape of (batch, n_tokens, token_embedding_size)
        timestep_groups: shape of (batch, horizon, n_tokens, token_embedding_size)

        after splited (batch, total_tokens, token_embedding_size), where total_tokens = horizon * n_tokens

        Reverses the process of assemble_input_tokens
        """
        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups] ### list of each group's number of tokens 

        n_prefix_tokens = sum(tokens_per_prefix_group) ## sum of all 

        prefix_embeddings, timestep_embeddings = torch.split(output_tokens, [n_prefix_tokens, output_tokens.shape[1] - n_prefix_tokens], dim = 1)

        if len(prefix_groups) > 0:
            prefix_embeddings_split = split_tokens(prefix_embeddings, tokens_per_prefix_group, axis=1)
            all_prefix_output = [
                group.replace(tokens = embeddings)
                for group, embeddings in zip(prefix_groups, prefix_embeddings_split)
            ]
        
        else:
            all_prefix_output = []

        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]
        batch, total_timestep_token, token_embedding_size = timestep_embeddings.shape 
        time_dim = timestep_groups[0].tokens.shape[1]
        ## Process timestep group outputs
        timestep_embeddings = timestep_embeddings.reshape(batch, time_dim, -1, token_embedding_size)
        timestep_embeddings_split = split_tokens(timestep_embeddings, tokens_per_timestep_group, axis=2)
        all_timestep_output = [
            group.replace(tokens = embeddings)
            for group, embeddings in zip(timestep_groups, timestep_embeddings_split)
        ]

        return all_prefix_output, all_timestep_output
    

    def generate_attention_mask(self,
                                prefix_groups: Sequence[PrefixGroup],
                                timestep_groups: Sequence[TimestepGroup],
                                ):
        """
        Args:
            prefix_groups: A list of PrefixGroup objects
            timestep_groups: A list of TimestepGroup objects

        Returns:
            attention_mask: A boolean mask of shape (batch, 1, total_tokens, total_tokens)
        
        We use the attention rules specified by each group to determine the transformer attention mask 
        We then combine this with the padding mask to ensure that padding tokens are not attended to.
        """
        if self.enforce_causal:
            self.verify_causality(prefix_groups, timestep_groups)

        if not self.use_correct_attention:
            ## No longer used in new models, but keeping for backward compatability with models released in December
            logging.warning(
                "Using old attention computation from released Decemeber models"
            )
            side = 'left'
        else:
            side = 'right'

        def _get_position(i, tokens_per_group):
            """
            i (int): index of tokens -> whether it is prefix token or timestep token
            tokens_per_group (List): tokens_per_prefix_group or tokens_per_timestep_group to calculate cumulative sum 


            """
            return np.searchsorted(np.cumsum(tokens_per_group), i, side=side)
        
        horizon = timestep_groups[0].tokens.shape[1]
        tokens_per_prefix_group = [group.tokens.shape[1] for group in prefix_groups]
        tokens_per_timestep_group = [group.tokens.shape[2] for group in timestep_groups]

        tokens_for_prefix = sum(tokens_per_prefix_group)
        tokens_per_timestep = sum(tokens_per_timestep_group)

        total_tokens = tokens_for_prefix + tokens_per_timestep * horizon 

        attention_mask = np.zeros((total_tokens, total_tokens), dtype = int) ### This attention_mask includes all prefix and timestep 

        def get_token_metadata(i):
            ## Separates the areas between tokens for prefix and timestep 
            ## if index of token is under prefix, 
            if i < tokens_for_prefix: 
                position = _get_position(i, tokens_per_prefix_group)
                return TokenMetadata.create(prefix_groups[position], timestep=-1)  ### take last timestep 
            
            i -= tokens_for_prefix  ## to check the position of timestep token to remove prefix token index
            timestep, i = divmod(i, tokens_per_timestep)
            position = _get_position(i, tokens_per_timestep_group)  ## get the position of timestep token 
            return TokenMetadata.create(timestep_groups[position], timestep)
        
        for i in range(total_tokens): ## Token Attending -> Current 
            for j in range(total_tokens): ## Token being attended to -> Past
                metadata_i = get_token_metadata(i)  ## return attention rules which are boolean 
                metadata_j = get_token_metadata(j)  ## return attention rules which are boolean 
                mask = int(metadata_i.should_attend_to(metadata_j)) ### to check whether token_i should attend token_j or not 
                attention_mask[i, j] = mask

        pad_attention_mask = self.generate_pad_attention_mask(prefix_groups, timestep_groups)
        ### combine
        full_attn_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(timestep_groups[0].tokens.shape[0], 1, total_tokens, total_tokens)
        full_attn_mask = attention_mask & pad_attention_mask
        return full_attn_mask
    
    def verify_causality(self,
                         prefix_groups: Sequence[PrefixGroup],
                         timestep_groups: Sequence[TimestepGroup],
                         ):
        """
        Ensures that no token can attend to another token in a future timestep
        """
        
        ## First verify that prefix group isn't attending to any timestep group

        for prefix_group in prefix_groups:
            for ts_group in timestep_groups:
                rule = find_match(prefix_group.attention_rules, ts_group.attention_rules, AttentionRule.NEVER)
                assert (
                    prefix_group.attention_rules.get(ts_group.name, AttentionRule.NEVER) == AttentionRule.NEVER
                ), f'Causality broken! Prefix group {prefix_group.name} is attending to timestep group {ts_group.name}'

        ## Next, make sure that nothing is attending to future timesteps 
        for group in prefix_groups + timestep_groups:
            for other_group in prefix_groups + timestep_groups:
                rule = find_match(
                    group.attention_rules, other_group.name,AttentionRule.NEVER
                )
                assert (
                    rule != AttentionRule.ALL
                ), f'Causality broken! WhenToAttend.ALL attends to future timesteps too.'

    def pretty_print_attention_mask(self, prefix_groups: Sequence[PrefixGroup], timestep_groups: Sequence[TimestepGroup]):
        """
        Visualizes the attention patterns 
        """
        horizon = timestep_groups[0].tokens.shape[1]
        cols = []
        metas: List[TokenMetadata] = []
        for pg in prefix_groups:
            cols.append(f"{pg.name} ({pg.tokens.shape[1]} tok)")
            metas.append(TokenMetadata.create(pg, timestep=-1))
        for ts in range(horizon):
            for tg in timestep_groups:
                cols.append(f"t={ts} {tg.name} ({tg.tokens.shape[2]} tok)")
                metas.append(TokenMetadata.create(tg, timestep=ts))

        rows = []
        for j, mj in enumerate(metas):
            row = [cols[j]]
            for mi in metas:
                row.append("x" if mi.should_attend_to(mj) else " ")
            rows.append(row)

        logging.warning("Attention layout:")
        logging.warning(" | ".join([" "] + cols))
        for r in rows:
            logging.warning(" | ".join(r))