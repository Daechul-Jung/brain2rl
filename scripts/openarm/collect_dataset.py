#!/usr/bin/env python3
import argparse
import os, sys
import numpy as np

HOME = os.path.expanduser("~")
ROS_INSTALL = os.path.join(HOME, "ros2_ws", "install")
if ROS_INSTALL not in sys.path:
    sys.path.insert(0, ROS_INSTALL)

from brain2rl_openarm.envs.openarm_env import OpenArmEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--outfile', type=str, default='openarm_rollouts.npz')
    ap.add_argument('--policy', choices=['random'], default='random')
    args = ap.parse_args()

    env = OpenArmEnv(action_scale=0.03, horizon=args.steps)
    obs_buf, act_buf, rew_buf, next_obs_buf, done_buf = [], [], [], [], []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        for t in range(args.steps):
            if args.policy == 'random':
                act = env.action_space.sample()
            obs2, r, term, trunc, _ = env.step(act)
            done = bool(term or trunc)
            obs_buf.append(obs)
            act_buf.append(act)
            rew_buf.append(r)
            next_obs_buf.append(obs2)
            done_buf.append(done)
            obs = obs2
            if done:
                break
        print(f"[collect_dataset] ep {ep+1}/{args.episodes} steps={t+1}")

    env.close()
    np.savez_compressed(
        args.outfile,
        obs=np.asarray(obs_buf, dtype=np.float32),
        actions=np.asarray(act_buf, dtype=np.float32),
        rewards=np.asarray(rew_buf, dtype=np.float32),
        next_obs=np.asarray(next_obs_buf, dtype=np.float32),
        dones=np.asarray(done_buf, dtype=np.bool_)
    )
    print(f"[collect_dataset] wrote {args.outfile}")

if __name__ == "__main__":
    main()
