import torch
from torch.distributions.kl import kl_divergence


class OnPolicyAC:
    """Minimal on-policy actor-critic with latent KL regularizer."""
    def __init__(
        self,
        encoder, # ReferenceEncoder
        low_level_policy, # LowLevelPolicy (or mixture/product)
        value_fn, # MultiHeadValue
        enc_opt, low_opt, v_opt,
        beta: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        ):
        # TODO: store handles and hyperparams
        pass

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        """TODO: return (advantages, returns)."""
        # TODO
        pass

    def _actor_loss(self, logp: torch.Tensor, advantages: torch.Tensor):
        """TODO: compute policy loss skeleton."""
        # TODO
        pass

    def _critic_loss(self, values: torch.Tensor, targets: torch.Tensor):
        """TODO: compute value loss skeleton."""
        # TODO
        pass

    def _latent_kl(self, qz, pz) -> torch.Tensor:
        """TODO: KL(q||p) over latents."""
        # TODO
        pass

    def update_tracking(self, batch: dict) -> dict:
        """TODO: single update step for tracking phase; return logs."""
        # TODO
        pass

    def update_transfer(self, batch: dict, high_level_policy) -> dict:
        """TODO: update step for task transfer with frozen low-level."""
        # TODO
        pass

    def update_joint(self, batch: dict, high_level_heads: dict) -> dict:
        """TODO: update step for joint training across tasks."""
        # TODO
        pass