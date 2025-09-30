import torch
import torch.nn as nn

class EMA:
    """
    EMA: exponential moving average, keep a shadow copy of the weights that updates as a smooth average of recent checkpoint
    """
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.requires_grad}

    def update(self, model:nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha = 1 - self.decay)

    def copy_to(self, model:nn.Module):
        model.state_dict({**model.state_dict(), **self.shadow})
