#!/usr/bin/env python3
import argparse
import os, sys
import numpy as np

# Make sure the ROS package (installed by colcon) is on PYTHONPATH
HOME = os.path.expanduser("~")
ROS_INSTALL = os.path.join(HOME, "ros2_ws", "install")
if ROS_INSTALL not in sys.path:
    sys.path.insert(0, ROS_INSTALL)

# Common ROS overlay site-packages pattern
for root in (ROS_INSTALL,):
    for sub in ("brain2rl_openarm/lib/python3.11/site-packages",
                "brain2rl_openarm/lib/python3.10/site-packages",
                "lib/python3.11/site-packages",
                "lib/python3.10/site-packages"):
        p = os.path.join(root, sub)
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

from brain2rl_openarm.envs.openarm_env import OpenArmEnv

# Add your RL agents
B2R = os.path.join(HOME, "brain2rl")
if B2R not in sys.path:
    sys.path.insert(0, B2R)
RL_DIR = os.path.join(B2R, "models", "rl")
if RL_DIR not in sys.path:
    sys.path.insert(0, RL_DIR)

Agent = None
train_fn = None
try:
    from rl_agents_prac import PPOAgent, SACAgent, train_agent_with_buffer
    Agent = PPOAgent
    train_fn = train_agent_with_buffer
except Exception:
    try:
        from rl_agents_prac import PPOAgent, SACAgent, train_agent
        Agent = PPOAgent
        train_fn = train_agent
    except Exception as e:
        print("[train_online] Warning: couldn't import your RL code:", e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', choices=['ppo','sac'], default='ppo')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--save', type=str, default='openarm_ppo.pt')
    args = ap.parse_args()

    env = OpenArmEnv(action_scale=0.03, horizon=args.steps)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if Agent is None or train_fn is None:
        print("[train_online] RL code not found; running a random-policy dry run.")
        for ep in range(3):
            obs, _ = env.reset()
            ret = 0.0
            for t in range(args.steps):
                a = env.action_space.sample()
                obs, r, term, trunc, _ = env.step(a)
                ret += r
                if term or trunc:
                    break
            print(f"[train_online] ep {ep+1} return={ret:.3f}")
        env.close()
        return

    # select algo
    if args.algo == 'ppo':
        agent = PPOAgent(obs_dim, act_dim)
    else:
        agent = SACAgent(obs_dim, act_dim)

    # Try your training helper if it exists
    try:
        info = train_fn(env, agent, num_episodes=args.episodes, max_steps=args.steps, render=False)
    except TypeError:
        # fallback simple loop
        for ep in range(args.episodes):
            obs, _ = env.reset()
            ret = 0.0
            for t in range(args.steps):
                a = agent.act(obs)
                next_obs, r, term, trunc, _ = env.step(a)
                if hasattr(agent, "remember"):
                    agent.remember(obs, a, r, next_obs, term or trunc)
                if hasattr(agent, "update"):
                    agent.update()
                obs = next_obs
                ret += r
                if term or trunc:
                    break
            print(f"[train_online] ep {ep+1} return={ret:.3f}")

    if hasattr(agent, "save"):
        agent.save(args.save)
        print(f"[train_online] saved -> {args.save}")
    env.close()

if __name__ == "__main__":
    main()
