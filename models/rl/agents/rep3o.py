"""
Relative Entropy Proximal Pairwise Policy Optimization (My own Idea)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch.optim as optim
import copy
from torch.amp import GradScaler
import gymnasium as gym
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.reppo_network import *
from models.rl.utils.any_utils import (compute_gve, _ensure_batch, _env_shape, _split_step_return, _to_tensor, wrap_batch_dim)