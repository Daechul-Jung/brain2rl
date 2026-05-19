# Compatibility shim — the canonical implementation lives in research/pure_rl/agents/td3bc/.
# Import from there instead of this file.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from research.pure_rl.agents.td3bc.config import TD3BCConfig  # noqa: F401
from research.pure_rl.agents.td3bc.networks import Actor, Critic  # noqa: F401
from research.pure_rl.agents.td3bc.replay_buffer import OfflineReplayBuffer  # noqa: F401
from research.pure_rl.agents.td3bc.agent import TD3BCAgent  # noqa: F401

__all__ = ["TD3BCConfig", "Actor", "Critic", "OfflineReplayBuffer", "TD3BCAgent"]
