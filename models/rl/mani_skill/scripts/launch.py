import os, sys
import sapien
import torch
import torch.nn as nn
import gymnasium as gym

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.mani_skill.tasks.multiple_tasks_env import *
from models.rl.mani_skill.thinkers.policyThinker import PolicyThinker
from models.rl.mani_skill.thinkers.thinkingAgent import ThinkingAgent
from models.rl.mani_skill.thinkers.task_thinker import *


def make_env(obs_mode = 'state', control_mode = 'pd_ee_delta_pose'):
    env = gym.make(
        'Combined-v1',
        robot_uids = 'so100'
    )
    return env

def main():
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")

    planner = TaskPlannerLLM(MODEL_ID, HF_TOKEN, )
    instruction = "push first and pull and pick then stack"
    env = make_env()

    
    return 


if __name__ == '__main__':
    main()