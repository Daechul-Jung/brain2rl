import torch
from typing import Any, Dict
from tensordict import TensorDict

class RolloutBuffer:
    """Time-major storage for on-policy rollouts."""
    def __init__(self, capacity: int, device: str = "cuda"):
        # TODO: allocate placeholders
        pass

    def add(self, transition: Dict[str, Any]):
        """TODO: push one transition (dict of tensors)."""
        # TODO
        pass

    def sample(self) -> Dict[str, torch.Tensor]:
        """TODO: return stacked tensors for update."""
        # TODO
        pass

    def clear(self):
        """TODO: reset buffer pointers."""
        # TODO
        pass