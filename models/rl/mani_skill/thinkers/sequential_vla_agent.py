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
    def __init__(self, env: gym.Env, policy, planner, device, run_config: Optional[RunConfig] = None):
        self.env = env
        self.policy_model = policy
        self.planner = planner
        self.device = device
        self.cfg = run_config if run_config is not None else RunConfig

    def _order_task_from_prompt(self, prompt: str) -> dict:
        tasks = self.planner.plan(prompt)
        tasks = [t for t, _ in sorted(tasks.items(), key = lambda kv: kv[1])]
        
        if len(tasks) == 0:
            tasks = ['push', 'pull', 'pick', 'place']
        return tasks
    
    def _get_iamge_from_obs(self, obs: Optional[dict]):
        if 'rgb' in obs:
            rgb = obs["rgb"]
        else:
            raise KeyError("No 'rgb' in obs. Use FlattenRGBDObservationWrapper or adapt this function.")
        if rgb.ndim == 4:
            rgb = rgb[0]
        return np.asarray(rgb, dtype=np.uint8)

    def run_episode(self, prompt: str):
        ordered_task = self._order_task_from_prompt(prompt)
        observation, info = self.env.reset(seed = 100 , options = {'task_sequence': ordered_task})
        print(observation)
        logs ={
            'prompt': prompt,
            'task_sequence': ordered_task,
            'steps': 0,
            'terminated': False,
            'truncated': False
        }

        for t in range(self.cfg.max_step):
            rgb = self._get_iamge_from_obs(observation)

            curr_stage = getattr(self.env.unwrapped, 'curr_stage', 'push')

            sub_instruction = TASK_TO_TEXT.get(curr_stage, curr_stage)

            action = self.policy_model.predict_action(rgb, sub_instruction)
            observation, reward, terminated, truncated, info = self.env.step(action)
            if self.cfg.render:
                try:
                    self.env.render()
                    time.sleep(1/60)
                except Exception:
                    pass

            if (t % self.cfg.print_every) == 0:
                print(f"[t={t:04d}] stage={info.get('curr_stage')} reward={float(reward):.3f} success={info.get('success')}")

            if terminated or truncated:
                logs["terminated"] = bool(terminated)
                logs["truncated"] = bool(truncated)
                logs["steps"] = t + 1
                break
        return logs