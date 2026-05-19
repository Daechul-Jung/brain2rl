"""
Visualize a trained agent rolling out on any Gymnasium / MuJoCo environment.

Works with:
  - PureRePPOAgent checkpoints (--agent reppo)
  - TD3BCAgent checkpoints   (--agent td3bc)
  - Random policy            (--random)

Usage:
    # Visualize trained REPPO on Pendulum
    python scripts/simulate_mujoco.py \
        --env Pendulum-v1 \
        --checkpoint output/reppo/final.pth \
        --agent reppo \
        --episodes 5

    # Visualize random policy
    python scripts/simulate_mujoco.py --env HalfCheetah-v4 --random --episodes 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.envs.registry import make_env


def _obs_action_dims(env):
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


def _load_reppo(checkpoint: str, obs_dim: int, action_dim: int, device: str):
    from research.pure_rl.agents.reppo_agent import PureRePPOAgent
    agent = PureRePPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(checkpoint)
    return agent


def _load_td3bc(checkpoint: str, obs_dim: int, action_dim: int, device: str):
    from models.rl.agents.td3_bc import TD3BCAgent
    agent = TD3BCAgent(observation_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(checkpoint)
    return agent


@torch.no_grad()
def _reppo_action(agent, obs_np: np.ndarray) -> np.ndarray:
    device = agent.device
    obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    norm = agent.agent.observation_normalizer(obs_t)
    pi, _, _, _ = agent.agent._actor_forward(norm)
    act = pi.sample().clamp(-1 + 1e-6, 1 - 1e-6)
    return act.squeeze(0).cpu().numpy().astype(np.float32)


@torch.no_grad()
def _td3bc_action(agent, obs_np: np.ndarray) -> np.ndarray:
    return agent.get_action(obs_np)


def simulate(args: argparse.Namespace) -> None:
    env = make_env(args.env, render_mode="human")
    obs_dim, action_dim = _obs_action_dims(env)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    agent = None
    get_action = None

    if not args.random:
        if not args.checkpoint:
            raise ValueError("--checkpoint required unless --random is set")
        if args.agent == "reppo":
            agent = _load_reppo(args.checkpoint, obs_dim, action_dim, device)
            get_action = lambda obs: _reppo_action(agent, obs)
        elif args.agent == "td3bc":
            agent = _load_td3bc(args.checkpoint, obs_dim, action_dim, device)
            get_action = lambda obs: _td3bc_action(agent, obs)
        else:
            raise ValueError(f"Unknown agent type: {args.agent}")
        print(f"Loaded {args.agent} checkpoint: {args.checkpoint}")
    else:
        get_action = lambda obs: env.action_space.sample()
        print("Using random policy")

    total_reward_list = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0
        done = False

        while not done and step < args.max_steps:
            obs_np = np.asarray(obs, dtype=np.float32).flatten()
            action = get_action(obs_np)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
            step += 1
            if args.delay > 0:
                time.sleep(args.delay)

        total_reward_list.append(ep_reward)
        success = info.get("success", info.get("is_success", "N/A"))
        print(f"  Episode {ep+1}/{args.episodes}  reward={ep_reward:.2f}  steps={step}  success={success}")

    env.close()
    print(f"\nMean episode reward: {np.mean(total_reward_list):.2f} ± {np.std(total_reward_list):.2f}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Pendulum-v1")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--agent", choices=["reppo", "td3bc"], default="reppo")
    p.add_argument("--random", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--delay", type=float, default=0.0, help="Seconds between steps (slow down for visualization)")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    simulate(_parse())
