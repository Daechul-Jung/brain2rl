"""
Thinking agent with pluggable Thinker module. 
 - BaseThinker: interface for any ML/planning policy that decides the next high level intention. 
 - HeuristicThinker: baseline finite-state machine (no learning) for pick-and-place
 - PolicyThinker: placeholder showing where to plug a learned model

The agent converts high-level intents (e.g., target pose above cube/goal, gripper open/close)
into low-level actions compatible with ManiSkill control_mode="pd_ee_delta_pose".

Action format expected by env (OpenArm config):
[dx, dy, dz, dRx, dRy, dRz, gripper_target]
where per-step deltas are clipped to ±0.1 meters/radians per your PDEEPose controller.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple
import torch
import torch.nn as nn
import numpy as np

def clamp(x: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(x, low), high)

def pose_from_obs(raw_pose):
    """
    Split Maniskill 7D pose [px, py, pz, qx, qy, qz, qw] into (p, q)
    """
    arr = np.asarray(raw_pose)
    p = arr[..., :3]
    q = arr[..., 3:7]
    return p, q

@dataclass
class Intent:
    """
    High level command that the agent should realize this step
    """
    target_pos_world: Optional[np.ndarray]= None 
    use_absolute_target: bool = True
    
    # Optional direct per-step deltas (x, y, z) when not using absolute target 
    dpos: Optional[np.ndarray] = None
    # Desired gripper opening (position command for PD gripper )
    gripper: Optional[float] = None


class BaseThinker(Protocol):
    def reset(self,): ...
    def step(self, obs: Dict, info: Dict)-> Intent:... 

@dataclass
class Controlcfg:
    """
    Config for control 
    """
    dpos_clip: float = 0.1
    drot_clip: float = 0.1
    ee_gain_xy: float = 0.5
    ee_gain_z: float = 0.5


class ThinkingAgent:
    """
    Wrap Thinking model and convert intents into low-level actions. Main body of thinker 
    """

    def __init__(self, thinker, ctrl: Controlcfg, ):
        self.thinker = thinker ## Any thinker can be here but 
        self.ctrl = ctrl

    def reset(self):
        self.thinker.reset()

    def _delta_to(self, curr: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        d = np.zeros(3, dtype=np.float32)
        diff = tgt - curr
        d[:2] = self.ctrl.ee_gain_xy * diff[:2]
        d[2] = self.ctrl.ee_gain_z * diff[2]
        return clamp(d, -self.ctrl.dpos_clip, self.ctrl.dpos_clip)
    
    def act(self, obs: Dict, info: Dict):
        intent = self.thinker.step(obs, info)
        tcp_p, _ = pose_from_obs(obs['tcp_pose']) ## TCP: Tool Center Point 

        if intent.use_absolute_target and intent.target_pos_world is not None:
            dpos = self._delta_to(tcp_p, np.asarray(intent.target_pos_world, dtype=np.float32))

        elif intent.dpos is not None:
            dpos = np.asarray(intent.dpos, dtype=np.float32)
            dpos = clamp(dpos, -self.ctrl.dpos_clip, self.ctrl.dpos_clip)

        else:
            dpos = np.zeros(3, dtype= np.float32)

        drot = np.zeros(3, dtype=np.float32)
        gripper = intent.gripper if intent.gripper is not None else 0.0451

        action = np.concatenate([dpos, drot, np.array(gripper, dtype = np.float32)], axis=0)
        return action 


    