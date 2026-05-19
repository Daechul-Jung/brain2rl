"""
EEG Tokenizer — encodes a raw EEG segment into one token per RL timestep.

Design (ADR-002, ADR-003):
  - Conv1D trunk extracts temporal features from the EEG signal.
  - adaptive_avg_pool1d resamples the feature map to exactly T_rl frames.
  - One output token per RL timestep, so token[t] is fed to the brain
    conditioner at step t.
  - Trained end-to-end with the RL loss (ADR-008).

Shapes:
  Input  : (B, C, T_eeg)  — batch, EEG channels, raw time samples
  Output : (B, T_rl, token_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EEGTokenizer(nn.Module):
    """
    Encodes a raw EEG segment (B, C, T_eeg) into a token sequence (B, T_rl, token_dim).

    Args:
        n_channels  : number of EEG electrode channels
        token_dim   : output token dimensionality (default 128)
        hidden_dim  : Conv1D channel width (doubled on each layer)
        n_layers    : number of Conv blocks (each halves the time dimension)
        dropout     : dropout probability after the linear projection
    """

    def __init__(
        self,
        n_channels: int,
        token_dim: int = 128,
        hidden_dim: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = token_dim

        layers: list[nn.Module] = []
        in_ch = n_channels
        out_ch = hidden_dim
        for _ in range(n_layers):
            layers.append(_ConvBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)

        self.trunk = nn.Sequential(*layers)

        self.proj = nn.Sequential(
            nn.Linear(in_ch, token_dim),
            nn.LayerNorm(token_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, T_rl: int) -> torch.Tensor:
        """
        Args:
            x    : (B, C, T_eeg) raw EEG segment
            T_rl : number of RL timesteps — the output token count

        Returns:
            tokens : (B, T_rl, token_dim)
        """
        h = self.trunk(x)                          # (B, ch, T')
        h = F.adaptive_avg_pool1d(h, T_rl)         # (B, ch, T_rl)
        h = h.permute(0, 2, 1)                     # (B, T_rl, ch)
        return self.proj(h)                        # (B, T_rl, token_dim)

    def load_pretrained_trunk(self, classifier_ckpt: str, strict: bool = False) -> None:
        """
        Warm-start the Conv trunk from a trained ActionClassifier checkpoint.
        Keys with prefix 'trunk.' are mapped; mismatched keys are skipped.
        """
        ckpt = torch.load(classifier_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt)  # support both raw and wrapped checkpoints
        trunk_sd = {
            k[len("trunk."):]: v
            for k, v in state.items()
            if k.startswith("trunk.")
        }
        if trunk_sd:
            self.trunk.load_state_dict(trunk_sd, strict=strict)
