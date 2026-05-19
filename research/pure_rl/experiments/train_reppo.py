"""
Pure RL REPPO training script for ManiSkill pick-and-place tasks.

Usage:
    python research/pure_rl/experiments/train_reppo.py \
        --env PickCube-v1 --total-steps 500000 --num-envs 4

Checkpoints and TensorBoard logs are saved to output/pure_rl/<run_name>/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, asdict
import json

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from research.envs.registry import make_env, maniskill_available
from research.pure_rl.agents.reppo_agent import PureRePPOAgent

import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
try:
    import contextlib, io as _io
    with contextlib.redirect_stderr(_io.StringIO()):
        from torch.utils.tensorboard import SummaryWriter
    _TB = True
except Exception:
    _TB = False


@dataclass
class TrainConfig:
    env_id: str = "PickCube-v1"
    num_envs: int = 1
    obs_mode: str = "state"
    total_steps: int = 500_000
    steps_per_update: int = 128
    num_atoms: int = 151
    vmin: float = -2500.0
    vmax: float = 2500.0
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    device: str = "cuda"
    seed: int = 0
    log_interval: int = 10
    save_interval: int = 100
    run_name: str = ""
    output_dir: str = "output/pure_rl"


def _parse() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PickCube-v1")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--obs-mode", default="state")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--steps-per-update", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lmbda", type=float, default=0.95)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="output/pure_rl")
    p.add_argument("--run-name", default="")
    a = p.parse_args()

    run_name = a.run_name or f"{a.env}_{int(time.time())}"
    return TrainConfig(
        env_id=a.env,
        num_envs=a.num_envs,
        obs_mode=a.obs_mode,
        total_steps=a.total_steps,
        steps_per_update=a.steps_per_update,
        lr=a.lr,
        gamma=a.gamma,
        lmbda=a.lmbda,
        device=a.device,
        seed=a.seed,
        output_dir=a.output_dir,
        run_name=run_name,
    )


def _obs_action_dims(env) -> tuple[int, int]:
    """Extract flat obs_dim and action_dim from env."""
    import gymnasium as gym
    obs_sp = env.observation_space
    act_sp = env.action_space
    if isinstance(obs_sp, gym.spaces.Box):
        obs_dim = int(np.prod(obs_sp.shape))
    else:
        raise ValueError(f"Unsupported obs space: {obs_sp}")
    action_dim = int(np.prod(act_sp.shape))
    return obs_dim, action_dim


def _info_success(infos) -> float:
    """Extract mean success rate from info dict (ManiSkill or Gymnasium)."""
    if isinstance(infos, (list, tuple)):
        successes = [float(i.get("success", 0.0)) for i in infos if isinstance(i, dict)]
        return float(np.mean(successes)) if successes else 0.0
    if isinstance(infos, dict):
        s = infos.get("success", infos.get("is_success", None))
        if s is None:
            return 0.0
        if isinstance(s, torch.Tensor):
            return float(s.float().mean().item())
        return float(np.mean(s))
    return 0.0


def train(cfg: TrainConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    writer = None
    if _TB:
        writer = SummaryWriter(log_dir=out_dir)

    # --- environment ---
    is_maniskill = maniskill_available() and cfg.env_id in {
        "PickCube-v1", "PushCube-v1", "StackCube-v1", "PullCube-v1",
    }
    if is_maniskill:
        env = make_env(cfg.env_id, obs_mode=cfg.obs_mode, num_envs=cfg.num_envs)
    else:
        env = make_env(cfg.env_id)

    obs_dim, action_dim = _obs_action_dims(env)
    print(f"[train_reppo] env={cfg.env_id}  obs={obs_dim}  act={action_dim}  device={cfg.device}")

    agent = PureRePPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=cfg.device,
        lr=cfg.lr,
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        num_atoms=cfg.num_atoms,
        vmin=cfg.vmin,
        vmax=cfg.vmax,
    )

    total_env_steps = 0
    update_idx = 0
    obs, cobs = None, None

    while total_env_steps < cfg.total_steps:
        traj, obs, cobs, info_list = agent.collect(
            env, obs, cobs, num_steps=cfg.steps_per_update
        )

        metrics = agent.update(traj)
        total_env_steps += cfg.steps_per_update * max(1, cfg.num_envs)
        update_idx += 1

        if update_idx % cfg.log_interval == 0:
            raw_r = traj["raw_rewards"].mean().item()
            shaped_r = traj["rewards"].mean().item()
            success = _info_success(info_list[-1])
            print(
                f"step={total_env_steps:>8d}  "
                f"raw_r={raw_r:.4f}  shaped_r={shaped_r:.4f}  "
                f"success={success:.3f}  "
                f"pi_loss={float(metrics.get('actor_loss', float('nan'))):.4f}"
            )
            if writer:
                writer.add_scalar("reward/raw", raw_r, total_env_steps)
                writer.add_scalar("reward/shaped", shaped_r, total_env_steps)
                writer.add_scalar("metrics/success", success, total_env_steps)
                for k, v in metrics.items():
                    writer.add_scalar(k, float(v), total_env_steps)

        if update_idx % cfg.save_interval == 0:
            ckpt = os.path.join(out_dir, "checkpoint.pth")
            agent.save(ckpt, step=total_env_steps)
            print(f"  -> saved {ckpt}")

    final_ckpt = os.path.join(out_dir, "final.pth")
    agent.save(final_ckpt, step=total_env_steps)
    print(f"Training complete. Final checkpoint: {final_ckpt}")
    env.close()
    if writer:
        writer.close()


if __name__ == "__main__":
    train(_parse())
