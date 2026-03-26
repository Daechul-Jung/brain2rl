from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

import mani_skill.envs  # noqa: register built-in envs
import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper

# Register our custom multi-task environment
from models.rl.mani_skill.tasks.multiple_tasks_env import CombinedTaskEnv  # noqa
from models.rl.agents.reppo import RePPOAgent, EmpiricalNormalizer
from models.rl.utils.train import train_reppo


def make_env(num_envs: int = 1, obs_mode: str = 'state',
             control_mode: str = 'pd_joint_delta_pos',
             render: bool = False, robot_uids: str = 'panda') -> gym.Env:
    """
    Create the Combined-v1 multi-task ManiSkill environment.
    """
    if render and num_envs > 1:
        print("Warning: render=True forces num_envs=1 (GUI constraint).")
        num_envs = 1

    render_mode = 'human' if render else None

    env = gym.make(
        'Combined-v1',
        num_envs=num_envs,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        robot_uids=robot_uids,
    )

    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)

    return env

def main(args: argparse.Namespace):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n=== REPPO Training on ManiSkill Combined-v1 ===")
    print(f"Device:      {device}")
    print(f"Num envs:    {args.num_envs}")
    print(f"Obs mode:    {args.obs_mode}")
    print(f"Control:     {args.control_mode}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"Render:      {args.render}")

    # --- Build environment ---
    env = make_env(
        num_envs=args.num_envs,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render=args.render,
        robot_uids=args.robot_uids,
    )

    obs_space = env.observation_space
    act_space = env.action_space
    print(f"\nObservation space: {obs_space.shape}")
    print(f"Action space:      {act_space.shape}")

    if not isinstance(act_space, gym.spaces.Box):
        raise ValueError("REPPO requires a continuous (Box) action space.")

    obs_dim    = obs_space.shape[-1]
    action_dim = act_space.shape[-1]

    obs_normalizer  = EmpiricalNormalizer(shape=obs_dim,    device=device)
    cobs_normalizer = EmpiricalNormalizer(shape=obs_dim,    device=device)

    agent = RePPOAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        num_atoms=args.num_atoms,
        vmin=args.vmin,
        vmax=args.vmax,
        device=device,
        lr=args.lr,
        gamma=args.gamma,
        kl_start=args.kl_start,
        entropy_start=args.entropy_start,
        lmbda=args.lmbda,
        obs_normalizer=obs_normalizer,
        critic_obs_normalizer=cobs_normalizer,
    )

    if args.resume_from and os.path.exists(args.resume_from):
        step = agent.load(args.resume_from)
        print(f"Resumed from {args.resume_from} (step={step})")

    print(f"\nStarting REPPO training...")
    all_returns = train_reppo(
        env=env,
        agent=agent,
        total_steps=args.total_steps,
        num_step=args.num_step,
        num_epoch=args.num_epoch,
        num_mini_batch=args.num_mini_batch,
    )

    # --- Save checkpoint ---
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    agent.save(args.output, extra={'all_returns': all_returns})
    print(f"\nSaved checkpoint to: {args.output}")

    if all_returns:
        print(f"Mean return (last 20 eps): {np.mean(all_returns[-20:]):.2f}")
        print(f"Max return:                {max(all_returns):.2f}")

    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Train REPPO agent on ManiSkill Combined-v1 environment')

    # Environment
    p.add_argument('--num-envs',     type=int,   default=1,
                   help='Number of parallel environments (>1 disables GUI)')
    p.add_argument('--obs-mode',     type=str,   default='state',
                   choices=['state', 'rgbd', 'pointcloud'],
                   help='Observation modality')
    p.add_argument('--control-mode', type=str,   default='pd_joint_delta_pos',
                   help='Robot control mode')
    p.add_argument('--robot-uids',   type=str,   default='panda',
                   choices=['panda', 'so100'],
                   help='Robot to use in simulation')
    p.add_argument('--render',       action='store_true',
                   help='Render with GUI (forces num_envs=1, much slower)')

    # Training
    p.add_argument('--total-steps',    type=int,   default=200_000,
                   help='Total environment steps')
    p.add_argument('--num-step',       type=int,   default=128,
                   help='Steps per rollout collection')
    p.add_argument('--num-epoch',      type=int,   default=8,
                   help='SGD epochs per rollout')
    p.add_argument('--num-mini-batch', type=int,   default=4,
                   help='Mini-batches per epoch')

    # REPPO hyperparameters
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--gamma',         type=float, default=0.99)
    p.add_argument('--lmbda',         type=float, default=0.95)
    p.add_argument('--kl-start',      type=float, default=0.01)
    p.add_argument('--entropy-start', type=float, default=0.01)
    p.add_argument('--num-atoms',     type=int,   default=151)
    p.add_argument('--vmin',          type=float, default=-50.0)
    p.add_argument('--vmax',          type=float, default=50.0)

    # Checkpointing
    p.add_argument('--output',      type=str, default='output/reppo_maniskill/checkpoint.pth',
                   help='Path to save the trained agent')
    p.add_argument('--resume-from', type=str, default=None,
                   help='Path to checkpoint to resume training from')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
