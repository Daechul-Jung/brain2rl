"""
In this script, setting robot from maniskills and also importing tasks, but trying to do multiple tasks according to the thinking process
"""

from copy import deepcopy
import numpy as np
import sapien 
import sapien.physx as physx
import torch

import mani_skill
import mani_skill.sensors as sensors
from models.rl.mani_skill.tasks.pick_cube_openarm import *


