"""
train_reppo.py — Entry point for EEG-conditioned REPPO training.

Usage:
    python research/experiments/train_reppo.py --env HalfCheetah-v4 --total-steps 500000
    python research/experiments/train_reppo.py --env Combined-v1 --eeg-csv data/processed/train.csv

Without EEG data (pure RL baseline, no brain conditioner):
    python research/experiments/train_reppo.py --env HalfCheetah-v4 --no-eeg
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from models.rl.agents.reppo import RePPOAgent
from models.rl.utils.reppo_network import EmpiricalNormalizer
from models.rl.utils.any_utils import compute_gve
from research.eeg.tokenizer import EEGTokenizer
from research.brain.transformer_delta import TransformerDelta
from research.agents.eeg_reppo import EEGRePPOAgent
from research.envs.registry import make_env
from research.experiments.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="HalfCheetah-v4")
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--eeg-csv", default=None, help="Path to processed EEG CSV")
    p.add_argument("--no-eeg", action="store_true", help="Disable brain conditioner (pure REPPO)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--name", default=None)
    p.add_argument("--render", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_eeg_data(csv_path: str, device: str):
    """Load EEG segments from processed CSV. Returns (segments, labels) or (None, None)."""
    try:
        import pandas as pd
        from models.classification.data_utilities import load_sensor_data, preprocess_multilabel
    except ImportError:
        print("[warn] Could not load EEG data utilities — running without EEG.")
        return None, None

    X_raw, y_str, _, _ = load_sensor_data(csv_path)
    X, y_enc, _, encoders = preprocess_multilabel(X_raw, y_str)
    labels = encoders["behavior"].transform(y_enc[:, 0]) if hasattr(encoders.get("behavior", None), "transform") else y_enc[:, 0].astype(int)
    segments = X.astype(np.float32)  # (N, C, T_eeg)
    return segments, labels


def build_reppo(obs_dim: int, action_dim: int, cfg: ExperimentConfig) -> RePPOAgent:
    obs_normalizer = EmpiricalNormalizer(obs_dim, device=cfg.device)
    return RePPOAgent(
        observation_dim=obs_dim,
        action_dim=action_dim,
        num_atoms=cfg.reppo_num_atoms,
        vmin=cfg.reppo_vmin,
        vmax=cfg.reppo_vmax,
        device=cfg.device,
        lr=cfg.reppo_lr,
        gamma=cfg.reppo_gamma,
        lmbda=cfg.reppo_lmbda,
        obs_normalizer=obs_normalizer,
        critic_obs_normalizer=obs_normalizer,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    env_kwargs = {}
    if args.render:
        env_kwargs["render_mode"] = "human"

    env = make_env(args.env, **env_kwargs)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    name = args.name or f"{args.env.split('-')[0].lower()}_reppo_{'eeg' if not args.no_eeg else 'base'}"
    cfg = ExperimentConfig(
        name=name,
        seed=args.seed,
        device=device,
        env_id=args.env,
        eeg_csv=args.eeg_csv,
        obs_dim=obs_dim,
        action_dim=action_dim,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        output_dir=args.output_dir,
    )
    os.makedirs(cfg.output_path(), exist_ok=True)
    cfg.save(cfg.output_path("config.json"))
    print(f"[train] Experiment: {cfg.name}  |  env: {cfg.env_id}  |  device: {device}")

    reppo = build_reppo(obs_dim, action_dim, cfg)

    # --- EEG + brain conditioner ---
    eeg_segments, eeg_labels = None, None
    use_eeg = (not args.no_eeg) and (args.eeg_csv is not None)

    if use_eeg:
        eeg_segments, eeg_labels = load_eeg_data(args.eeg_csv, device)
        use_eeg = eeg_segments is not None

    if use_eeg:
        tokenizer = EEGTokenizer(
            n_channels=eeg_segments.shape[1],
            token_dim=cfg.eeg_token_dim,
            hidden_dim=cfg.eeg_hidden_dim,
            n_layers=cfg.eeg_n_layers,
            dropout=cfg.eeg_dropout,
        )
        brain = TransformerDelta(
            token_dim=cfg.eeg_token_dim,
            action_dim=action_dim,
            d_model=cfg.brain_d_model,
            n_heads=cfg.brain_n_heads,
            n_layers=cfg.brain_n_layers,
            dropout=cfg.brain_dropout,
            max_seq_len=cfg.rollout_steps + 4,
            stochastic=cfg.brain_stochastic,
        )
        agent = EEGRePPOAgent(
            reppo=reppo,
            tokenizer=tokenizer,
            brain=brain,
            eeg_segments=eeg_segments,
            eeg_labels=eeg_labels,
            T_rl=cfg.rollout_steps,
            brain_lr=cfg.brain_lr,
        )
        print(f"[train] EEG conditioner: TransformerDelta  |  token_dim={cfg.eeg_token_dim}")
    else:
        agent = None  # pure REPPO
        print("[train] Running pure REPPO (no EEG conditioning)")

    # --- Training loop ---
    global_step = 0
    obs, cobs = None, None
    t0 = time.time()

    while global_step < cfg.total_steps:
        if use_eeg and agent is not None:
            trajectory, obs, cobs, infos, eeg_tokens = agent.collect(
                env, obs, cobs,
                task_label=cfg.task_label,
                num_steps=cfg.rollout_steps,
            )
            eeg_seg = agent.sample_eeg_segment(cfg.task_label)
            metrics = agent.update(trajectory, eeg_seg)
        else:
            trajectory, obs, cobs, infos = reppo.collect(
                env, obs, cobs, num_steps=cfg.rollout_steps
            )
            gve_list = compute_gve(
                rewards=trajectory["rewards"],
                dones=trajectory["dones"],
                truncations=trajectory["truncations"],
                next_values=trajectory["next_values"],
                gamma=reppo.gamma,
                lmbda=reppo.lmbda,
            )
            gve = torch.stack(gve_list, dim=0)
            flat = trajectory.reshape(-1)
            flat["gve"] = gve.reshape(-1, 1)
            reppo.old_actor.load_state_dict(reppo.actor.state_dict())
            metrics = {**reppo.update_critic(flat), **reppo.update_actor(flat)}

        global_step += cfg.rollout_steps

        if global_step % 1000 == 0:
            elapsed = time.time() - t0
            fps = global_step / elapsed
            raw_rew = trajectory["raw_rewards"].mean().item()
            print(
                f"[train] step={global_step:>8}  "
                f"raw_rew={raw_rew:+.3f}  "
                f"fps={fps:.0f}  "
                + ("brain_loss={:.4f}  alpha={:.3f}".format(
                    metrics.get("brain_loss", 0),
                    metrics.get("alpha_mean", 1),
                ) if use_eeg else "")
            )

        if global_step % cfg.save_every == 0:
            ckpt_path = cfg.output_path(f"checkpoint_{global_step}.pth")
            if use_eeg and agent is not None:
                agent.save(ckpt_path, step=global_step)
            else:
                reppo.save(ckpt_path, step=global_step)
            print(f"[train] Saved checkpoint → {ckpt_path}")

    env.close()
    print(f"[train] Done. Total steps: {global_step}")


if __name__ == "__main__":
    main()
