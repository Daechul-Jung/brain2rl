"""
Collect offline demonstrations from a trained REPPO agent on ManiSkill pick-and-place tasks.

Saves data in .npz format compatible with OfflineReplayBuffer.from_file().

Usage:
    # Collect 50k transitions from a trained checkpoint
    python scripts/generate_offline_data.py \
        --env PickCube-v1 \
        --checkpoint output/pure_rl/PickCube_run/final.pth \
        --n-steps 50000 \
        --out data/offline/PickCube.npz

    # Collect from a random policy (for sanity checks)
    python scripts/generate_offline_data.py \
        --env PickCube-v1 --random --n-steps 10000 --out data/offline/PickCube_random.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from research.envs.registry import make_env, maniskill_available
from research.pure_rl.agents.reppo_agent import PureRePPOAgent


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _obs_action_dims(env):
    import gymnasium as gym
    obs_sp = env.observation_space
    act_sp = env.action_space
    obs_dim = int(np.prod(obs_sp.shape))
    action_dim = int(np.prod(act_sp.shape))
    return obs_dim, action_dim


def collect(args: argparse.Namespace) -> None:
    is_maniskill = maniskill_available() and args.env in {
        "PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1",
    }
    env = make_env(args.env, obs_mode="state", num_envs=1) if is_maniskill else make_env(args.env)
    obs_dim, action_dim = _obs_action_dims(env)

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    agent = None
    if not args.random:
        if not args.checkpoint:
            raise ValueError("--checkpoint required unless --random is set")
        agent = PureRePPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")

    print(f"Collecting {args.n_steps} steps from {args.env} ({'random' if args.random else 'policy'})...")

    observations, actions, rewards, next_observations, terminals = [], [], [], [], []

    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret

    collected = 0
    while collected < args.n_steps:
        obs_np = _to_numpy(obs).reshape(1, obs_dim) if obs_dim > 1 else _to_numpy(obs).reshape(1, -1)

        if args.random or agent is None:
            action = env.action_space.sample()
            action_np = np.asarray(action, dtype=np.float32).flatten()
        else:
            with torch.no_grad():
                obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                norm_obs = agent.agent.observation_normalizer(obs_t)
                pi, _, _, _ = agent.agent._actor_forward(norm_obs)
                act_t = pi.sample().clamp(-1 + 1e-6, 1 - 1e-6)
            action_np = act_t.squeeze(0).cpu().numpy().astype(np.float32)
            action = action_np

        step = env.step(action)
        next_obs, rew, done, trunc, _ = step

        next_np = _to_numpy(next_obs).reshape(1, obs_dim) if obs_dim > 1 else _to_numpy(next_obs).reshape(1, -1)
        rew_val = float(_to_numpy(rew).flat[0])
        done_val = bool(_to_numpy(done).flat[0]) or bool(_to_numpy(trunc).flat[0])

        observations.append(obs_np.squeeze(0))
        actions.append(action_np)
        rewards.append(rew_val)
        next_observations.append(next_np.squeeze(0))
        terminals.append(float(done_val))

        collected += 1
        if collected % 5000 == 0:
            print(f"  {collected}/{args.n_steps}")

        if done_val:
            ret = env.reset()
            obs = ret[0] if isinstance(ret, tuple) else ret
        else:
            obs = next_obs

    env.close()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        next_observations=np.array(next_observations, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
    )
    print(f"Saved {collected} transitions to {args.out}")
    print(f"  obs shape: {np.array(observations).shape}")
    print(f"  act shape: {np.array(actions).shape}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PickCube-v1")
    p.add_argument("--checkpoint", default="", help="Path to trained REPPO checkpoint")
    p.add_argument("--random", action="store_true", help="Use random policy instead of trained agent")
    p.add_argument("--n-steps", type=int, default=50_000)
    p.add_argument("--out", default="data/offline/PickCube.npz")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    return p.parse_args()


if __name__ == "__main__":
    collect(_parse())
