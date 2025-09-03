import torch
from torch.distributions.kl import kl_divergence


class OnPolicyAC:
    """Minimal on-policy actor-critic with latent KL regularizer."""
    def __init__(
        self,
        encoder, ## ReferenceEncoder
        low_level_policy, ## LowLevelPolicy (or mixture/product)
        value_fn, ## MultiHeadValue (critic)
        enc_opt, low_opt, v_opt, ## Encoder, low-level, value optimizer
        beta: float = 1e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        ):
        self.encoder = encoder
        self.ll_policy = low_level_policy
        self.critic = value_fn
        self.encoder_opt = enc_opt
        self.ll_policy_opt = low_opt
        self.critic_opt = v_opt
        self.beta = beta
        self.gamma = gamma
        self.lmbda = lam

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor):
        """Calculating Generalized Advantage Estimation (GAE-Lambda).
        td = r_t + gamma * (1- done_t) * v(s_t+l) - v(s_t)
        """
        adv = torch.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            td = rewards[t] + self.gamma * (1 - dones[t]) * values[t+1] - values[t]
            gae = td + self.gamma * self.lmbda * (1)
        return adv

    def _actor_loss(self, logp: torch.Tensor, advantages: torch.Tensor):
        """This loss is based on 'on-policy variant of Maximum a Posteriori Policy Optimization' """
        
        pass

    def _critic_loss(self, values: torch.Tensor, targets: torch.Tensor):
        """ MSE loss between values and targets """
        
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