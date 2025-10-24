import os, sys, importlib
import numpy as np
from pathlib import Path
import gymnasium as gym
from typing import Dict, Optional, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # adjust if needed
sys.path.insert(0, str(PROJECT_ROOT))
from models.rl.mani_skill.thinkers.policyThinker import PolicyThinker
from models.rl.mani_skill.thinkers.thinkingAgent import ThinkingAgent

MS_REPO_ROOT = Path('~/openarm_maniskill_simulation').expanduser()  # the folder that contains 'mani_skill/'
assert (MS_REPO_ROOT / 'mani_skill').exists(), "Repo root must contain mani_skill/"
sys.path.insert(0, str(MS_REPO_ROOT))

if 'mani_skill' in sys.modules:
    del sys.modules['mani_skill']
importlib.invalidate_caches()

import mani_skill
print("mani_skill loaded from:", mani_skill.__file__)



