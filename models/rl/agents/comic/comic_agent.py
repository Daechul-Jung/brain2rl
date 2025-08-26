import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from dataclasses import dataclass
from typing import Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


"""
This is an algorithm called CoMic: Complementary Task Learning & Mimicry for Reusable Skils written by Google DeepMind
"""
@dataclass
class AgentConfig:
    z_dim: int = 32
    beta: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    device: str = "cpu"


class CoMicAgent:
    def __init__(self, encoder, low_level, value_fn, algo, cfg: AgentConfig):
        # TODO: store modules and config
        pass

    def rollout_tracking(self, env, T: int) -> Dict:
        """TODO: collect T steps; must populate keys needed by algo.update_tracking."""
        # TODO
        pass

    def train_tracking(self, env, updates: int, T_per_update: int) -> None:
        """TODO: loop: rollout -> update -> log."""
        # TODO
        pass

    def freeze_low_level(self) -> None:
        """TODO: set requires_grad(False) for low-level policy params."""
        # TODO
        pass

    def act_transfer(self, high_level, obs) -> torch.Tensor:
        """TODO: sample z from high_level and action from low_level."""
        # TODO
        pass

    def train_transfer(self, env, high_level, updates: int, T_per_update: int) -> None:
        """TODO: rollout + update for transfer phase."""
        # TODO
        pass

    def train_joint(self, env_tracking, task_envs: Dict[str, object], task_probs: Dict[str, float], updates: int, T_per_update: int) -> None:
        """TODO: interleave tracking and complementary tasks using probabilities."""
        # TODO
        pass

    def save(self, path: str) -> None:
        """TODO: serialize state dicts and config."""
        # TODO
        pass

    def load(self, path: str) -> None:
        """TODO: restore state dicts and config."""
        # TODO
        pass