"""
BrainConditioner — abstract interface for pluggable EEG conditioning modules.

How to add a new idea (ADR-004):
  1. Write a spec in research/ideas/NNN_your_idea.md
  2. Subclass BrainConditioner in research/brain/your_idea.py
  3. Implement forward(), token_dim, and action_dim
  4. Pass an instance to EEGRePPOAgent(brain=YourConditioner(...))
  5. Run: pytest tests/test_02_brain.py to verify the interface contract

Interface contract:
  - forward(token_seq, t) consumes EEG tokens accumulated up to step t.
  - It returns a dict that MUST contain 'delta_action' (B, action_dim).
  - It MAY also return 'alpha' (B, 1) in [0, 1] — a confidence gate.
    If absent, the agent defaults alpha = 1.0 (full delta applied).
  - The module must be an nn.Module so optimizers can reach its parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BrainConditioner(nn.Module, ABC):
    """
    Abstract base class for all EEG brain conditioning modules.

    Subclasses receive the EEG token sequence accumulated so far in the
    episode and produce an action modification signal for the current step.
    """

    @abstractmethod
    def forward(self, token_seq: torch.Tensor, t: int) -> dict[str, torch.Tensor]:
        """
        Compute action conditioning signal from EEG tokens.

        Args:
            token_seq : (B, t+1, token_dim)  — all EEG tokens from step 0 to t
            t         : current RL timestep index (0-based)

        Returns:
            dict with at minimum:
                'delta_action' : (B, action_dim)  — action perturbation
            optionally:
                'alpha'        : (B, 1)  — gate in [0, 1]; defaults to 1.0 if absent
        """

    @property
    @abstractmethod
    def token_dim(self) -> int:
        """Dimensionality of input EEG tokens — must match EEGTokenizer.token_dim."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimensionality of the output delta_action — must match the environment."""

    def check_output(self, out: dict, B: int) -> None:
        """Validate that forward() output meets the interface contract."""
        assert "delta_action" in out, "BrainConditioner.forward() must return 'delta_action'"
        assert out["delta_action"].shape == (B, self.action_dim), (
            f"delta_action shape mismatch: got {out['delta_action'].shape}, "
            f"expected ({B}, {self.action_dim})"
        )
        if "alpha" in out:
            assert out["alpha"].shape == (B, 1), (
                f"alpha shape mismatch: got {out['alpha'].shape}, expected ({B}, 1)"
            )
