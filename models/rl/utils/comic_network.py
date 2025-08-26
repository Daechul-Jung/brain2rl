import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden=(256, 256)):
        """TODO: define layers only; no logic beyond structure."""
        super().__init__()
        # TODO: define layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: forward pass skeleton."""
        # TODO
        pass


class ReferenceEncoder(nn.Module):
    """π_HL(z | s, s_ref) -> Normal distribution over latent z."""
    def __init__(self, state_dim: int, ref_dim: int, z_dim: int):
        super().__init__()
        # TODO: declare submodules/params (e.g., MLP for mu, logstd param)

    def forward(self, s: torch.Tensor, s_ref: torch.Tensor) -> Normal:
        """TODO: return Normal(mu, std) without implementing details."""
        # TODO
        pass

class HighLevelPolicy(nn.Module):
    """Task-specific high-level policy: z ~ π_HL^{task}(·|o)."""
    def __init__(self, obs_dim: int, z_dim: int):
        super().__init__()
        # TODO: declare mu_net, logstd param

    def forward(self, o: torch.Tensor) -> Normal:
        """TODO: return Normal over z."""
        # TODO
        pass


class MultiHeadValue(nn.Module):
    """Multiple value heads; total value is sum of heads."""
    def __init__(self, state_dim: int, n_heads: int):
        super().__init__()
        # TODO: create n value heads

    def forward(self, s: torch.Tensor):
        """TODO: return (per_head_values, total_value)."""
        # TODO
        pass


class MixtureOfGaussiansPolicy(nn.Module):
    """Optional: π(a|s,z)=∑_i w_i(s,z) N(a; μ_i(s), σ_i(s))."""
    def __init__(self, state_dim: int, z_dim: int, act_dim: int, num_components: int = 4):
        super().__init__()
        # TODO: declare torso for primitives (s), per-component μ/σ, mixing head (s,z)

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        """TODO: return mixture parameters (mus, stds, mix_logits)."""
        # TODO
        pass


class ProductOfGaussiansPolicy(nn.Module):
    """Optional: π(a|s,z) ∝ ∏_i N(a; μ_i(s), σ_i(s))^{w_i(s,z)}."""
    def __init__(self, state_dim: int, z_dim: int, act_dim: int, num_experts: int = 4):
        super().__init__()
        # TODO: declare experts and weighting head

    def forward(self, s: torch.Tensor, z: torch.Tensor):
        """TODO: return experts' params and weights."""
        # TODO
        pass