from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def _mlp(input_dim: int, output_dim: int, hidden_dim: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.net = _mlp(observation_dim, action_dim, hidden_dim)
        low = torch.as_tensor(action_low, dtype=torch.float32)
        high = torch.as_tensor(action_high, dtype=torch.float32)
        self.register_buffer("action_scale", (high - low) / 2.0)
        self.register_buffer("action_bias", (high + low) / 2.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.action_bias + self.action_scale * torch.tanh(self.net(obs))


class Critic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = observation_dim + action_dim
        self.q1 = _mlp(input_dim, 1, hidden_dim)
        self.q2 = _mlp(input_dim, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1))
