import os
from typing import Dict

import numpy as np
import torch


class OfflineReplayBuffer:
    """
    Fixed dataset buffer for offline RL.

    Expected arrays: observations, actions, rewards, next_observations, terminals.
    Common aliases (obs/act/rew/next_obs/done) are also accepted.
    """

    KEY_ALIASES = {
        "observations": ("observations", "obs", "states", "state"),
        "actions": ("actions", "act", "action"),
        "rewards": ("rewards", "rew", "reward"),
        "next_observations": ("next_observations", "next_obs", "next_states", "next_state"),
        "terminals": ("terminals", "dones", "done", "timeouts"),
    }

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_observations: np.ndarray,
        terminals: np.ndarray,
        device: str = "cuda",
        normalize_observations: bool = True,
    ):
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")

        observations = np.asarray(observations, dtype=np.float32)
        next_observations = np.asarray(next_observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        terminals = np.asarray(terminals, dtype=np.float32).reshape(-1, 1)

        if observations.shape != next_observations.shape:
            raise ValueError("observations and next_observations must have the same shape.")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("observations and actions must contain the same number of samples.")

        self.obs_mean = observations.mean(axis=0, keepdims=True)
        self.obs_std = observations.std(axis=0, keepdims=True) + 1e-6
        if normalize_observations:
            observations = (observations - self.obs_mean) / self.obs_std
            next_observations = (next_observations - self.obs_mean) / self.obs_std
        else:
            self.obs_mean = np.zeros_like(self.obs_mean, dtype=np.float32)
            self.obs_std = np.ones_like(self.obs_std, dtype=np.float32)

        self.observations = torch.as_tensor(observations, device=self.device)
        self.actions = torch.as_tensor(actions, device=self.device)
        self.rewards = torch.as_tensor(rewards, device=self.device)
        self.next_observations = torch.as_tensor(next_observations, device=self.device)
        self.terminals = torch.as_tensor(terminals, device=self.device)
        self.size = int(observations.shape[0])

    @classmethod
    def from_file(
        cls,
        path: str,
        device: str = "cuda",
        normalize_observations: bool = True,
    ) -> "OfflineReplayBuffer":
        ext = os.path.splitext(path)[1].lower()
        if ext == ".npz":
            with np.load(path) as data:
                arrays = {k: data[k] for k in data.files}
        elif ext in {".h5", ".hdf5"}:
            try:
                import h5py
            except ImportError as exc:
                raise ImportError("Install h5py to load HDF5 offline RL datasets.") from exc
            arrays = {}
            with h5py.File(path, "r") as data:
                for key in data.keys():
                    if hasattr(data[key], "shape"):
                        arrays[key] = np.asarray(data[key])
        else:
            raise ValueError(f"Unsupported dataset extension: {ext}. Use .npz, .h5, or .hdf5.")

        picked = {}
        for canonical, aliases in cls.KEY_ALIASES.items():
            key = next((alias for alias in aliases if alias in arrays), None)
            if key is None:
                raise KeyError(f"Dataset is missing '{canonical}'. Accepted aliases: {aliases}")
            picked[canonical] = arrays[key]

        return cls(
            observations=picked["observations"],
            actions=picked["actions"],
            rewards=picked["rewards"],
            next_observations=picked["next_observations"],
            terminals=picked["terminals"],
            device=device,
            normalize_observations=normalize_observations,
        )

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "terminals": self.terminals[idx],
        }

    def normalize_obs(self, observation: np.ndarray) -> np.ndarray:
        return ((observation - self.obs_mean.squeeze(0)) / self.obs_std.squeeze(0)).astype(np.float32)
