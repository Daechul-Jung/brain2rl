from dataclasses import dataclass
import torch
import time
import gymnasium as gym
import os, sys
from typing import Optional
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TASK_TO_TEXT = {
    "push":  "push the red cube into the push target region",
    "pull":  "pull the red cube into the pull target region",
    "pick":  "pick the red cube and place it at the green goal sphere",
    "stack": "stack the red cube on top of the green cube",
}


@dataclass
class RunConfig:
    max_step: int = 50000
    render: bool = True
    print_every: int = 20


class SequentialVLAAgent:
    """
    Runs OpenVLA (or any compatible policy) on the Combined-v1 ManiSkill
    environment, executing tasks in the order planned by the LLM planner.

    Policy interface
    ----------------
    The policy must be callable as:
        action = policy(rgb_image_np, instruction_str)
    where
        rgb_image_np : np.ndarray  uint8  (H, W, 3)
        instruction_str : str
        action : np.ndarray  (action_dim,)

    OpenVLAPolicy already implements __call__ with this signature.

    Environment requirement
    -----------------------
    The env must be created with  obs_mode='rgbd'  so that observations
    contain an 'rgb' key. State-only obs_mode will not provide images.
    """

    def __init__(self, env: gym.Env, policy, planner, device,
                 run_config: Optional[RunConfig] = None):
        self.env = env
        self.policy_model = policy  
        self.planner = planner
        self.device = device
        self.cfg = run_config if run_config is not None else RunConfig()

    def _order_task_from_prompt(self, prompt: str) -> list:
        tasks = self.planner.plan(prompt)
        ordered = [t for t, _ in sorted(tasks.items(), key=lambda kv: kv[1])]
        if not ordered:
            ordered = ['push', 'pull', 'pick', 'stack']
        return ordered

    def _get_rgb_from_obs(self, obs) -> np.ndarray:
        """
        Extract a uint8 (H, W, 3) image from the environment observation.

        Works with both dict observations (rgbd mode) and batched outputs.
        """
        if isinstance(obs, dict):
            if 'sensor_data' in obs:
                for cam_key in obs['sensor_data']:
                    cam = obs['sensor_data'][cam_key]
                    if 'rgb' in cam:
                        rgb = cam['rgb']
                        if isinstance(rgb, torch.Tensor):
                            rgb = rgb.cpu().numpy()
                        if rgb.ndim == 4:
                            rgb = rgb[0]   # take first env
                        return np.asarray(rgb, dtype=np.uint8)
            if 'rgb' in obs:
                rgb = obs['rgb']
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:
                    rgb = rgb[0]
                return np.asarray(rgb, dtype=np.uint8)
        raise KeyError(
            "Could not find 'rgb' in observation. "
            "Create the environment with obs_mode='rgbd'.")

    def run_episode(self, prompt: str) -> dict:
        ordered_task = self._order_task_from_prompt(prompt)
        print(f"Task order: {ordered_task}")

        observation, info = self.env.reset(
            seed=100, options={'task_sequence': ordered_task})

        logs = {
            'prompt': prompt,
            'task_sequence': ordered_task,
            'steps': 0,
            'terminated': False,
            'truncated': False,
            'stage_rewards': [],
        }

        for t in range(self.cfg.max_step):
            rgb = self._get_rgb_from_obs(observation)

            curr_stage = getattr(self.env.unwrapped, 'curr_stage', 'push')
            sub_instruction = TASK_TO_TEXT.get(curr_stage, curr_stage)

            # OpenVLAPolicy.__call__(rgb, instruction) → scaled action
            action = self.policy_model(rgb, sub_instruction)

            observation, reward, terminated, truncated, info = self.env.step(action)

            if self.cfg.render:
                try:
                    self.env.render()
                    time.sleep(1 / 60)
                except Exception:
                    pass

            # if t % self.cfg.print_every == 0:
            #     reward = []
            #     rew_val = float(reward[0]) if hasattr(reward, '__len__') else float(reward)
            #     print(f"[t={t:04d}] stage={info.get('curr_stage','?')}  "
            #           f"reward={rew_val:.3f}  success={info.get('success')}")

            # if terminated or truncated:
            #     logs['terminated'] = bool(terminated) if not hasattr(terminated, '__len__') else bool(terminated.any())
            #     logs['truncated'] = bool(truncated) if not hasattr(truncated, '__len__') else bool(truncated.any())
            #     logs['steps'] = t + 1
            #     break

        return logs