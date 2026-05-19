"""
Visualize a trained REPPO agent rolling out on ManiSkill pick-and-place tasks.

Requires ManiSkill 3.0. Uses the GUI renderer (render_mode='human') for
interactive visualization, or saves frames to video with --video-dir.

Usage:
    python scripts/simulate_maniskill.py \
        --env PickCube-v1 \
        --checkpoint output/pure_rl/PickCube_run/final.pth \
        --episodes 5

    # Random policy (no checkpoint needed)
    python scripts/simulate_maniskill.py --env PushCube-v1 --random --episodes 3

    # Save video frames instead of live display
    python scripts/simulate_maniskill.py \
        --env PickCube-v1 \
        --checkpoint output/pure_rl/PickCube_run/final.pth \
        --video-dir output/videos/PickCube
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import gymnasium as gym
    from mani_skill.utils.wrappers.record import RecordEpisode
    _MS_AVAILABLE = True
except ImportError:
    _MS_AVAILABLE = False


def _require_maniskill():
    if not _MS_AVAILABLE:
        print("ManiSkill3 not installed. Install with: pip install mani-skill")
        sys.exit(1)


def _obs_action_dims(env) -> tuple[int, int]:
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    return obs_dim, action_dim


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


@torch.no_grad()
def _reppo_action(agent, obs_t: torch.Tensor) -> np.ndarray:
    obs_t = obs_t.to(agent.device)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)
    norm = agent.agent.observation_normalizer(obs_t)
    pi, _, _, _ = agent.agent._actor_forward(norm)
    act = pi.sample().clamp(-1 + 1e-6, 1 - 1e-6)
    return act.squeeze(0).cpu().numpy().astype(np.float32)


def simulate(args: argparse.Namespace) -> None:
    _require_maniskill()

    render_mode = "human" if args.video_dir is None else "rgb_array"
    env = gym.make(args.env, obs_mode="state", num_envs=1, render_mode=render_mode)

    if args.video_dir is not None:
        os.makedirs(args.video_dir, exist_ok=True)
        env = RecordEpisode(env, output_dir=args.video_dir, save_video=True)
        print(f"Recording videos to {args.video_dir}")

    obs_dim, action_dim = _obs_action_dims(env)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    agent = None
    if not args.random:
        if not args.checkpoint:
            raise ValueError("--checkpoint required unless --random is set")
        from research.pure_rl.agents.reppo_agent import PureRePPOAgent
        agent = PureRePPOAgent(obs_dim=obs_dim, action_dim=action_dim, device=device)
        agent.load(args.checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using random policy")

    success_count = 0
    reward_list = []

    for ep in range(args.episodes):
        ret = env.reset()
        obs = ret[0] if isinstance(ret, tuple) else ret
        ep_reward = 0.0
        step = 0
        done = False

        while not done and step < args.max_steps:
            if agent is None:
                action = env.action_space.sample()
            else:
                obs_t = torch.as_tensor(_to_numpy(obs), dtype=torch.float32)
                action = _reppo_action(agent, obs_t)

            step_ret = env.step(action)
            obs, rew, terminated, truncated, info = step_ret

            rew_val = float(_to_numpy(rew).flat[0])
            ep_reward += rew_val

            terminated_val = bool(_to_numpy(terminated).flat[0])
            truncated_val = bool(_to_numpy(truncated).flat[0])
            done = terminated_val or truncated_val
            step += 1

        # Extract success from info (ManiSkill returns tensor)
        success = info.get("success", None)
        if success is not None:
            success_val = bool(_to_numpy(success).flat[0])
            success_count += int(success_val)
        else:
            success_val = "N/A"

        reward_list.append(ep_reward)
        print(f"  Episode {ep+1}/{args.episodes}  reward={ep_reward:.3f}  steps={step}  success={success_val}")

    env.close()
    print(f"\nSuccess rate: {success_count}/{args.episodes}")
    print(f"Mean episode reward: {np.mean(reward_list):.3f} ± {np.std(reward_list):.3f}")


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PickCube-v1",
                   choices=["PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1"])
    p.add_argument("--checkpoint", default="")
    p.add_argument("--random", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--video-dir", default=None, help="Save video frames here instead of GUI display")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    simulate(_parse())
