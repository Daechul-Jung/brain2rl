from typing import Any, Mapping, Sequence, Union, MutableMapping, List, Tuple, Dict
from __future__ import annotations
import torch

PRNGkey = torch.Generator
TensorLike = Union[torch.Tensor, int, float, bool]
PyTree = Union[torch.Tensor, Mapping[str, 'PyTree'], Sequence['PyTree']]
Config = Union[Any, Mapping[str, "Config"]]

Params = Mapping[str, PyTree]
Data   = Mapping[str, PyTree]

Shape = Sequence[int]
Dtype = torch.dtype