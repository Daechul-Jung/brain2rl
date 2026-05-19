"""
Pure RL REPPO training on the OpenArm MuJoCo pick-and-place task.

The environment: OpenArm bimanual robot (left arm only), brown cup, green goal ring.
Reward: reach → grasp → transport → success (curriculum-staged internally by the env).

Usage:
    # Headless training (fast)
    python3 research/pure_rl/experiments/train_openarm.py --total-steps 500000

    # With MuJoCo GUI (slower, for sanity-checking)
    python3 research/pure_rl/experiments/train_openarm.py --render --total-steps 50000

    # Print reward breakdown every N updates
    python3 research/pure_rl/experiments/train_openarm.py --log-interval 5

Checkpoints saved to output/openarm/<run_name>/.
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

_SCENE_XML = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "external", "openarm_mujoco", "v1", "openarm_bimanual_pick_place_scene.xml",
))


@dataclass
class OpenArmConfig:
    total_steps: int = 500_000
    steps_per_update: int = 128
    horizon: int = 300
    lr: float = 3e-4
    gamma: float = 0.99
    lmbda: float = 0.95
    num_atoms: int = 151
    vmin: float = -50.0
    vmax: float = 50.0
    action_scale: float = 1.0
    device: str = "cuda"
    seed: int = 0
    log_interval: int = 10
    save_interval: int = 50
    render: bool = False
    run_name: str = ""
    output_dir: str = "output/openarm"


def _parse() -> OpenArmConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--steps-per-update", type=int, default=128)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lmbda", type=float, default=0.95)
    p.add_argument("--action-scale", type=float, default=1.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--save-interval", type=int, default=50)
    p.add_argument("--render", action="store_true", help="Open MuJoCo GUI (passive viewer)")
    p.add_argument("--output-dir", default="output/openarm")
    p.add_argument("--run-name", default="")
    a = p.parse_args()
    run_name = a.run_name or f"reppo_{int(time.time())}"
    return OpenArmConfig(
        total_steps=a.total_steps,
        steps_per_update=a.steps_per_update,
        horizon=a.horizon,
        lr=a.lr,
        gamma=a.gamma,
        lmbda=a.lmbda,
        action_scale=a.action_scale,
        device=a.device,
        seed=a.seed,
        log_interval=a.log_interval,
        save_interval=a.save_interval,
        render=a.render,
        output_dir=a.output_dir,
        run_name=run_name,
    )


def _make_env(cfg: OpenArmConfig):
    from models.rl.envs.openarm_mj_env import OpenArmMjEnv
    if not os.path.exists(_SCENE_XML):
        raise FileNotFoundError(
            f"Scene XML not found: {_SCENE_XML}\n"
            "Make sure external/openarm_mujoco/ is initialized."
        )
    return OpenArmMjEnv(
        xml_path=_SCENE_XML,
        horizon=cfg.horizon,
        render=cfg.render,
        action_scale=cfg.action_scale,
        target_cup="cup",
    )


def _info_summary(infos: list) -> dict:
    """Average key metrics from info dicts across recent steps."""
    keys = ("dist_ee_cup", "cup_to_goal", "grasping", "progress", "total_reward")
    out = {}
    for k in keys:
        vals = [i.get(k, 0.0) for i in infos if isinstance(i, dict)]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def train(cfg: OpenArmConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if not os.path.exists(_SCENE_XML):
        raise FileNotFoundError(
            f"Scene XML not found:\n  {_SCENE_XML}\n"
            "Run: git submodule update --init external/openarm_mujoco"
        )

    out_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    writer = SummaryWriter(log_dir=out_dir) if _TB else None

    env = _make_env(cfg)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"

    print(f"[train_openarm] obs={obs_dim}  act={action_dim}  device={device}")
    print(f"  action_space: low={env.action_space.low[:3]}... high={env.action_space.high[:3]}...")
    print(f"  Scene: {_SCENE_XML}")

    agent = PureRePPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
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
    recent_infos: list = []

    while total_env_steps < cfg.total_steps:
        traj, obs, cobs, info_list = agent.collect(
            env, obs, cobs, num_steps=cfg.steps_per_update
        )
        metrics = agent.update(traj)
        total_env_steps += cfg.steps_per_update
        update_idx += 1
        recent_infos.extend(info_list)
        # keep only the last 5 updates worth of infos
        recent_infos = recent_infos[-5 * cfg.steps_per_update:]

        if update_idx % cfg.log_interval == 0:
            raw_r = traj["raw_rewards"].mean().item()
            shaped_r = traj["rewards"].mean().item()
            summary = _info_summary(recent_infos)
            stage = max((i.get("curr_stage", 0) for i in recent_infos if isinstance(i, dict)), default=0)
            print(
                f"step={total_env_steps:>8d}  "
                f"raw_r={raw_r:+.4f}  shaped_r={shaped_r:+.4f}  "
                f"ee→cup={summary.get('dist_ee_cup', float('nan')):.3f}  "
                f"cup→goal={summary.get('cup_to_goal', float('nan')):.3f}  "
                f"grasp={summary.get('grasping', 0.0):.2f}  "
                f"stage={stage}"
            )
            if writer:
                writer.add_scalar("reward/raw", raw_r, total_env_steps)
                writer.add_scalar("reward/shaped", shaped_r, total_env_steps)
                for k, v in summary.items():
                    writer.add_scalar(f"info/{k}", v, total_env_steps)
                writer.add_scalar("info/curr_stage", stage, total_env_steps)
                for k, v in metrics.items():
                    writer.add_scalar(k, v, total_env_steps)

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
