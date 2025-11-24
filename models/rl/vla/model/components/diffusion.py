# Mostly adapted from: https://raw.githubusercontent.com/rail-berkeley/bridge_data_v2/main/jaxrl_m/networks/diffusion_nets.py
import logging
from typing import Callable, Sequence, Optional

import torch
import torch.nn as nn

default_init = nn.init.xavier_uniform