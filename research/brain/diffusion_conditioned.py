"""
DiffusionConditioned — PLACEHOLDER BrainConditioner (not yet implemented).

Idea spec: research/ideas/002_diffusion_conditioned.md

Planned design:
  - A DDPM-style diffusion model takes (noisy_action, EEG_token, timestep) as input.
  - Conditioned on the EEG token at the current RL step, it denoises an action directly.
  - Output is a full action sample, not a delta — so alpha = 1.0 and base_action is ignored.
  - Requires multiple denoising steps at inference; use DDIM for speed.

This placeholder raises NotImplementedError so tests catch it early.
Implement this class when ready to run idea #002.
"""

from __future__ import annotations

import torch

from research.brain.base import BrainConditioner


class DiffusionConditioned(BrainConditioner):
    """Placeholder — implement when ready. See research/ideas/002_diffusion_conditioned.md."""

    def __init__(self, token_dim: int, action_dim: int):
        super().__init__()
        self._token_dim = token_dim
        self._action_dim = action_dim

    @property
    def token_dim(self) -> int:
        return self._token_dim

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def forward(self, token_seq: torch.Tensor, t: int) -> dict[str, torch.Tensor]:
        raise NotImplementedError(
            "DiffusionConditioned is not yet implemented. "
            "See research/ideas/002_diffusion_conditioned.md for the design spec."
        )
