# scripts/eval_rl_tokenizer.py
import gymnasium as gym
import mani_skill.envs
import numpy as np
from models.rl.mani_skill.tasks.multiple_tasks_env import CombinedTaskEnv  # noqa

from core.tokenizer_rl_pipeline import RLTokenizerPipeline

pipe = RLTokenizerPipeline({})
pipe.build_token_pool(X_eeg, y_labels)   # or load from .npz
pipe.build_agent()
pipe.load('output/rl_tokenizer/rl_tokenizer.pth')

env = gym.make('Combined-v1', obs_mode='state',
               control_mode='pd_joint_delta_pos', render_mode='human')
obs, _ = env.reset()

for _ in range(500):
    action = pipe.get_action(obs, task_label=0)   # uses get_action() at line 597
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        obs, _ = env.reset()

env.close()
