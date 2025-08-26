import torch
import random
import numpy as np


def set_seed(seed: int) -> None:
    """TODO: set Python/NumPy/PyTorch seeds."""
    # TODO
    pass


def to_tensor(x, device: str):
    """TODO: convert input to torch.Tensor on device."""
    # TODO
    pass


def to_numpy(x: torch.Tensor):
    """TODO: detach+cpu+numpy."""
    # TODO
    pass


def make_optimizer(params, lr: float):
    """TODO: return an optimizer instance for params."""
    # TODO
    pass


def log(metrics: dict, step: int) -> None:
    """TODO: simple stdout or hook for logger."""
    # TODO
    pass