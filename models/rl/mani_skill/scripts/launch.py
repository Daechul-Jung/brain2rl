import os, sys
import sapien
import torch
import torch.nn as nn
import gymnasium as gym
from transformers import pipeline, AutoModelForVision2Seq
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.mani_skill.tasks.multiple_tasks_env import *
from models.rl.mani_skill.thinkers.policyThinker import PolicyThinker
from models.rl.mani_skill.thinkers.thinkingAgent import ThinkingAgent
from models.rl.mani_skill.thinkers.task_thinker import *
from models.rl.agents.ppo import PPOAgent
from models.rl.agents.reppo import RePPOAgent

def make_env(obs_mode = 'state', control_mode = 'pd_joint_delta_pos', render_mode = 'human'):
    env = gym.make(
        'Combined-v1',
        obs_mode = obs_mode,
        control_mode = control_mode,
        robot_uids = 'so100',
        render_mode = render_mode
    )
    return env

def main():
    MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
    HF_TOKEN = os.getenv("HF_TOKEN")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    planner = TaskPlannerLLM(MODEL_ID, HF_TOKEN, )
    instruction = "push first and pull and pick then stack"
    env = make_env()
    
    vla_actor = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b",
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True)
    reppo_actor = RePPOAgent
    ppo_actor = PPOAgent
    action_dim = env.action_space
    obs_dim = env.observation_space
    
    agent = ThinkingAgent(planner, vla_actor, obs_dim, action_dim, instruction, env, device)
    
    agent.learn(env)

    return 


if __name__ == '__main__':
    main()