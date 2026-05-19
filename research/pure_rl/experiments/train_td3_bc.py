"""
TD3+BC offline training script for ManiSkill pick-and-place tasks.

Offline data must be collected first:
    python scripts/generate_offline_data.py --env PickCube-v1 --out data/offline/PickCube.npz

Usage:
    python research/pure_rl/experiments/train_td3_bc.py \
        --env PickCube-v1 \
        --data data/offline/PickCube.npz \
        --total-updates 200000

Checkpoints saved to output/td3bc/<run_name>/.
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

from research.pure_rl.agents.td3bc import TD3BCAgent, TD3BCConfig, OfflineReplayBuffer

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
class TrainTD3BCConfig:
    env_id: str = "PickCube-v1"
    data_path: str = ""
    total_updates: int = 200_000
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 2.5
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2
    hidden_dim: int = 256
    device: str = "cuda"
    seed: int = 0
    log_interval: int = 1000
    save_interval: int = 10_000
    run_name: str = ""
    output_dir: str = "output/td3bc"


def _parse() -> TrainTD3BCConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="PickCube-v1")
    p.add_argument("--data", required=True, help="Path to .npz offline dataset")
    p.add_argument("--total-updates", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--alpha", type=float, default=2.5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="output/td3bc")
    p.add_argument("--run-name", default="")
    a = p.parse_args()

    run_name = a.run_name or f"{a.env}_td3bc_{int(time.time())}"
    return TrainTD3BCConfig(
        env_id=a.env,
        data_path=a.data,
        total_updates=a.total_updates,
        batch_size=a.batch_size,
        actor_lr=a.actor_lr,
        critic_lr=a.critic_lr,
        gamma=a.gamma,
        tau=a.tau,
        alpha=a.alpha,
        hidden_dim=a.hidden_dim,
        device=a.device,
        seed=a.seed,
        output_dir=a.output_dir,
        run_name=run_name,
    )


def train(cfg: TrainTD3BCConfig) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"Offline dataset not found: {cfg.data_path}\n"
            f"Generate it first with: python scripts/generate_offline_data.py --env {cfg.env_id}"
        )

    out_dir = os.path.join(cfg.output_dir, cfg.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    writer = None
    if _TB:
        writer = SummaryWriter(log_dir=out_dir)

    # --- load offline dataset ---
    replay_buffer = OfflineReplayBuffer.from_file(cfg.data_path, device=cfg.device)
    obs_dim = replay_buffer.observations.shape[-1]
    action_dim = replay_buffer.actions.shape[-1]
    print(
        f"[train_td3bc] env={cfg.env_id}  obs={obs_dim}  act={action_dim}  "
        f"dataset_size={len(replay_buffer)}  device={cfg.device}"
    )

    td3_cfg = TD3BCConfig(
        batch_size=cfg.batch_size,
        actor_lr=cfg.actor_lr,
        critic_lr=cfg.critic_lr,
        gamma=cfg.gamma,
        tau=cfg.tau,
        alpha=cfg.alpha,
        hidden_dim=cfg.hidden_dim,
    )

    agent = TD3BCAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        device=cfg.device,
        config=td3_cfg,
    )

    for update_idx in range(1, cfg.total_updates + 1):
        metrics = agent.update(replay_buffer)

        if update_idx % cfg.log_interval == 0:
            print(
                f"update={update_idx:>8d}  "
                f"actor_loss={metrics.get('actor/loss', float('nan')):.4f}  "
                f"critic_loss={metrics.get('critic/loss', float('nan')):.4f}"
            )
            if writer:
                for k, v in metrics.items():
                    writer.add_scalar(k, v, update_idx)

        if update_idx % cfg.save_interval == 0:
            ckpt = os.path.join(out_dir, f"checkpoint_{update_idx}.pth")
            agent.save(ckpt)
            print(f"  -> saved {ckpt}")

    final_ckpt = os.path.join(out_dir, "final.pth")
    agent.save(final_ckpt)
    print(f"Training complete. Final checkpoint: {final_ckpt}")
    if writer:
        writer.close()


if __name__ == "__main__":
    train(_parse())
