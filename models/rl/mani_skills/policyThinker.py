import torch
import torch.nn as nn
import os, sys
from typing import Dict
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.rl.mani_skills.thinkingAgent import *

class PolicyThinker:
    """
    Slot for thinking model and can be replaced with any ML model for 
    Plug your network here and map obs -> Intent. You can predict either:
        (A) absolute world target (above cube/goal), or
        (B) direct per-step delta (dpos).


        Example forward signature:
        intent = self.model.forward(obs_tensor) -> dict with keys {"target_pos", "gripper"}
    """
    def __init__(self, model: nn.Module, device: str = 'cuda', use_absolute_target: bool = True):
        self.device = device
        self.model = model.to(device)
        self.use_absolute_target = use_absolute_target

    def reset(self):
        pass

    def _obs_to_tensor(self, obs: Dict):
        tcp_p = pose_from_obs(obs['tcp_pose'])
        obj_p = pose_from_obs(obs['obj_pose'])

        goal_p = np.asarray(obs['goal_pose'], dtype = np.float32)
        x = np.concatenate([tcp_p, obj_p, goal_p], axis =0).astype(np.float32)
        x = torch.from_numpy(x)[None] ## Extend dimension

        return x.to(self.device)

    @ torch.no_grad()
    def step(self, obs: Dict, info: Dict):
        x = self._obs_to_tensor(obs)
        out = self.model(x)

        if self.use_absolute_target:
            target = out.get('target_pos')
            target = target.squeeze(0).detach().cpu().numpy().astype(np.float32)
            grip = out.get('gripper')
            grip_val = float(grip.squeeze().item()) if grip is not None else None
            return Intent(target_pos_world=target, use_absolute_target=True)
        
        else:
            dpos = out.get('target_pos')
            dpos = dpos.squeeze(0).detach().cpu().numpy().astype(np.float32)
            grip = out.get("gripper")
            grip_val = float(grip.squeeze().item()) if grip is not None else None
            return Intent(dpos=dpos, use_absolute_target=False, gripper=grip_val)