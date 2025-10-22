from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple
import torch
import torch.nn as nn
import numpy as np


@dataclass
class HeuristicCfg:
    hover_h: float = 0.10
    descend_d: float = 0.005
    lift_h: float = 0.10
    tol_xy: float = 0.01
    tol_z: float = 0.01
    grip_open: float = 0.0451
    grip_close: float = 0.0 

