import os, sys
import torch
import gymnasium as gym
import argparse
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _PROJECT_ROOT)

from models.rl.mani_skill.tasks.multiple_tasks_env import CombinedTaskEnv  # noqa: register env
from models.rl.mani_skill.thinkers.openvla_policy import OpenVLAPolicy
from models.rl.mani_skill.thinkers.sequential_vla_agent import SequentialVLAAgent, RunConfig
from models.rl.mani_skill.thinkers.task_thinker import TaskPlannerLLM


def make_env(obs_mode = 'rgbd', control_mode = 'pd_joint_delta_pos', render_mode = 'human'):
    env = gym.make(
        'Combined-v1',
        obs_mode = obs_mode,
        control_mode = control_mode,
        robot_uids = 'so100',
        render_mode = render_mode
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True)
    return env

def main(args):
    LLM_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_ID = 'openvla/openvla-7b'
    HF_TOKEN = os.getenv("HF_TOKEN")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = TaskPlannerLLM(LLM_MODEL_ID, HF_TOKEN, device_map='auto')
    instruction = args.prompt
    
    
    vla = OpenVLAPolicy(model_id=MODEL_ID, 
                        device=device, 
                        use_flash_attn=True)
    env = make_env(render_mode='human')
    agent = SequentialVLAAgent(env, vla, planner, device)
    log = agent.run_episode(instruction)
    env.close()
    return log


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prompt for task sequence')
    parser.add_argument('--prompt', type=str, default = 'push and pull and pick then stack')
    args = parser.parse_args()
    main(args)