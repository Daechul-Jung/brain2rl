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
from typing import Dict, Optional, Protocol, Tuple, List, Any
import os, sys
import json, re
from huggingface_hub import login, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pipelines import pipeline
import torch
import torch.nn as nn
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.agents.reppo import *
from models.rl.agents.ppo import *


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

    def __init__(self, planner, actor, obs_dim, action_dim, ctrl: Controlcfg, prompt: str, env: Any ,device= 'cuda'):
        self.planner = planner ## Any thinker can be here but VLA for later
        self.actor = actor(obs_dim, action_dim) ### This is RL actor for performing actual action distribution
        self.device = device
        self.ctrl = ctrl
        self.env = env
        self.tasks = self.setting_tasks(prompt) 
        self.todo_list = {}
        self.setting_todo_list()
        self.actor = self._initialize_actor(actor)

    def setting_tasks(self, prompt):
        plan = self.planner.plan(prompt)
        return plan

    def setting_todo_list(self):
        for task in self.tasks.keys():
            self.todo_list[task] = False


    def _initialize_actor(self, actor):
        obs_dim = self.env.action_space.shape[0]
        action_dim = self.env.observation_space.shape[0]

        if hasattr(actor, PPOAgent):
            self.actor = PPOAgent()

        elif hasattr(actor, RePPOAgent):
            self.actor = RePPOAgent(obs_dim, action_dim, device= self.device)


    def act(self, obs, info, train = True):
        """
        Get action distribution from actor 
        """
        if not train:
            self.actor

    def learn(self, env, total_steps = 200000, num_step = 128, num_epoch = 16, num_mini_batch = 8):
        """
        Train with the given env
        """
        N_envs = getattr(env, 'num_envs', 1)
        batch_size = (N_envs * num_step)
        total_updates = total_steps * 10 // (N_envs * num_step) + 1
        
        eval_interval = max(1, total_updates // 5)
        reset_return, info = env.reset()
        observation = reset_return[0]


        if hasattr(env.action_space, 'low') and isinstance(env.action_space, gym.spaces.Box):
            low, high = env.action_space.low, env.action_space.high 

    def _episode_stats_from_rollout(transition):
        return 