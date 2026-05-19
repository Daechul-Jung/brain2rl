"""
ExperimentConfig — single source of truth for all experiment hyperparameters.

Usage:
    cfg = ExperimentConfig(env_id="HalfCheetah-v4", total_steps=500_000)
    cfg.save("output/my_run/config.json")
    cfg2 = ExperimentConfig.load("output/my_run/config.json")
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Optional


@dataclasses.dataclass
class ExperimentConfig:
    # ---- Experiment identity ----
    name: str = "eeg_reppo"
    seed: int = 42
    device: str = "cuda"

    # ---- Environment ----
    env_id: str = "HalfCheetah-v4"
    env_kwargs: dict = dataclasses.field(default_factory=dict)

    # ---- EEG ----
    eeg_csv: Optional[str] = None          # path to processed EEG CSV
    eeg_n_channels: int = 8               # number of EEG channels
    eeg_token_dim: int = 128              # EEGTokenizer output dim
    eeg_hidden_dim: int = 64             # Conv1D channel width
    eeg_n_layers: int = 3               # number of Conv blocks
    eeg_dropout: float = 0.1

    # ---- Brain conditioner ----
    brain_type: str = "transformer_delta"  # "transformer_delta" | "diffusion_conditioned"
    brain_d_model: int = 256
    brain_n_heads: int = 4
    brain_n_layers: int = 2
    brain_dropout: float = 0.1
    brain_stochastic: bool = False
    brain_lr: float = 3e-4

    # ---- REPPO ----
    obs_dim: int = 17                  # set to match env.observation_space
    action_dim: int = 6                # set to match env.action_space
    reppo_lr: float = 3e-4
    reppo_gamma: float = 0.99
    reppo_lmbda: float = 0.95
    reppo_num_atoms: int = 151
    reppo_vmin: float = -2500.0
    reppo_vmax: float = 2500.0

    # ---- Training ----
    total_steps: int = 500_000
    rollout_steps: int = 128           # steps per collect() call  == T_rl
    eval_every: int = 10_000
    save_every: int = 50_000
    output_dir: str = "output"
    log_to_tensorboard: bool = True

    # ---- Data ----
    task_label: int = 0               # EEG task label to condition on

    # ------------------------------------------------------------------

    def output_path(self, *parts: str) -> str:
        return os.path.join(self.output_dir, self.name, *parts)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def __post_init__(self) -> None:
        # T_rl == rollout_steps by convention
        self.T_rl = self.rollout_steps
