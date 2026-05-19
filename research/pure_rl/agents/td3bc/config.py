from dataclasses import dataclass


@dataclass
class TD3BCConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5
    hidden_dim: int = 256
