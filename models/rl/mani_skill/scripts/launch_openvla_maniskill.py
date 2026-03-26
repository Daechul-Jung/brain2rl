"""
launch_openvla_maniskill.py
===========================
Evaluate / fine-tune OpenVLA on the ManiSkill Combined-v1 environment
without any EEG input.  Three modes:

  eval   -- run N episodes with OpenVLA-7b + LLaMA planner, report success rate
  collect -- collect (obs, instruction, action) demonstrations using a scripted
             policy and save them as a .npz dataset
  finetune -- LoRA fine-tune OpenVLA on a collected demo dataset

Usage examples
--------------
  # Zero-shot evaluation (needs GPU with ~16 GB VRAM for OpenVLA-7b)
  HF_TOKEN=hf_xxx python models/rl/mani_skill/scripts/launch_openvla_maniskill.py \\
      --mode eval \\
      --prompt "push first, then pull, pick, stack" \\
      --episodes 5

  # Collect 50 demonstrations using scripted/random policy
  python models/rl/mani_skill/scripts/launch_openvla_maniskill.py \\
      --mode collect \\
      --n-demos 50 \\
      --demo-out output/demos/combined_v1.npz

  # LoRA fine-tune OpenVLA on collected demos
  HF_TOKEN=hf_xxx python models/rl/mani_skill/scripts/launch_openvla_maniskill.py \\
      --mode finetune \\
      --demo-path output/demos/combined_v1.npz \\
      --lora-rank 16 \\
      --epochs 5 \\
      --output output/openvla_finetuned/
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _PROJECT_ROOT)

import mani_skill.envs                                              # noqa: register envs
import gymnasium as gym
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper

from models.rl.mani_skill.tasks.multiple_tasks_env import CombinedTaskEnv  # noqa
from models.rl.mani_skill.thinkers.sequential_vla_agent import (
    SequentialVLAAgent, RunConfig, TASK_TO_TEXT)
from models.rl.mani_skill.thinkers.task_thinker import (
    TaskPlannerLLM, heuristic_parse)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_rgbd_env(robot_uids: str = 'panda', render: bool = False) -> gym.Env:
    env = gym.make(
        'Combined-v1',
        obs_mode='rgbd',
        control_mode='pd_joint_delta_pos',
        robot_uids=robot_uids,
        render_mode='human' if render else None,
    )
    env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
    return env


def make_state_env(robot_uids: str = 'panda', render: bool = False) -> gym.Env:
    """State-only env; no images. Useful for scripted-policy demo collection."""
    return gym.make(
        'Combined-v1',
        obs_mode='state',
        control_mode='pd_joint_delta_pos',
        robot_uids=robot_uids,
        render_mode='human' if render else None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 -- Evaluate OpenVLA zero-shot
# ─────────────────────────────────────────────────────────────────────────────

def run_eval(args: argparse.Namespace):
    from models.rl.mani_skill.thinkers.openvla_policy import (
        OpenVLAPolicy, OpenVLAActionScaling)

    HF_TOKEN   = os.getenv('HF_TOKEN', '')
    device     = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[eval] Loading LLM planner ...')
    planner = TaskPlannerLLM(
        model_id=args.llm_model_id, token=HF_TOKEN, device_map='auto')

    print('[eval] Loading OpenVLA ...')
    # Scale actions for panda / so100 (tune these empirically)
    scaling = OpenVLAActionScaling(pos_scale=0.02, rot_scale=0.05, grip_scale=1.0)
    vla = OpenVLAPolicy(
        model_id=args.vla_model_id,
        device=device,
        use_flash_attn=args.flash_attn,
        scaling=scaling,
        unnorm_key=args.unnorm_key or None,
    )

    env  = make_rgbd_env(robot_uids=args.robot, render=args.render)
    cfg  = RunConfig(max_step=args.max_steps, render=args.render, print_every=50)
    agent = SequentialVLAAgent(env, vla, planner, device, run_config=cfg)

    results = []
    for ep in range(args.episodes):
        print(f'\n--- Episode {ep + 1}/{args.episodes} ---')
        log = agent.run_episode(args.prompt)
        results.append(log)
        print(f'    steps={log["steps"]}  '
              f'terminated={log["terminated"]}  truncated={log["truncated"]}')

    successes = sum(r['terminated'] for r in results)
    print(f'\n[eval] Success rate: {successes}/{args.episodes} '
          f'({100 * successes / args.episodes:.1f}%)')
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 -- Collect demonstrations with a scripted/random policy
# ─────────────────────────────────────────────────────────────────────────────

def run_collect(args: argparse.Namespace):
    """
    Collect (obs, action, instruction, reward, done) tuples.

    Policy: random for now. Replace _scripted_action() with your
    motion-planning or teleoperation policy for higher quality data.
    """
    env      = make_state_env(robot_uids=args.robot, render=args.render)
    task_seq = heuristic_parse(args.prompt) or {'push': 1, 'pull': 2, 'pick': 3, 'stack': 4}
    ordered  = [t for t, _ in sorted(task_seq.items(), key=lambda kv: kv[1])]

    obs_list, act_list, instr_list, rew_list, done_list = [], [], [], [], []

    for demo_idx in range(args.n_demos):
        obs, info = env.reset(seed=demo_idx, options={'task_sequence': ordered})
        if isinstance(obs, dict):
            obs_vec = obs.get('agent', np.zeros(env.observation_space.shape))
        else:
            obs_vec = np.asarray(obs, dtype=np.float32)

        ep_steps = 0
        done = False
        while not done and ep_steps < args.max_steps:
            stage   = getattr(env.unwrapped, 'curr_stage', 'push')
            instr   = TASK_TO_TEXT.get(stage, stage)
            action  = env.action_space.sample()          # replace with better policy

            next_obs, reward, terminated, truncated, _ = env.step(action)

            obs_list.append(obs_vec.copy())
            act_list.append(action.copy())
            instr_list.append(instr)
            rew_list.append(float(reward[0] if hasattr(reward, '__len__') else reward))
            done_list.append(bool(terminated) or bool(truncated))

            if isinstance(next_obs, dict):
                obs_vec = next_obs.get('agent', obs_vec)
            else:
                obs_vec = np.asarray(next_obs, dtype=np.float32)

            done     = bool(terminated) or bool(truncated)
            ep_steps += 1

        if (demo_idx + 1) % 10 == 0:
            print(f'[collect] demo {demo_idx + 1}/{args.n_demos}  '
                  f'total transitions={len(obs_list)}')

    os.makedirs(os.path.dirname(args.demo_out) or '.', exist_ok=True)
    np.savez(
        args.demo_out,
        obs=np.array(obs_list, dtype=np.float32),
        actions=np.array(act_list, dtype=np.float32),
        instructions=np.array(instr_list),
        rewards=np.array(rew_list, dtype=np.float32),
        dones=np.array(done_list, dtype=bool),
    )
    print(f'[collect] Saved {len(obs_list)} transitions -> {args.demo_out}')
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 -- LoRA fine-tune OpenVLA on collected demos
# ─────────────────────────────────────────────────────────────────────────────

def run_finetune(args: argparse.Namespace):
    """
    Fine-tune OpenVLA with LoRA on (image, instruction, action) demos.

    Requires:
        pip install peft bitsandbytes

    Note: The demos collected by run_collect() only have state obs, not images.
    For image-action fine-tuning you need to re-collect with obs_mode='rgbd'
    (use make_rgbd_env) and store the RGB frames alongside the actions.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise RuntimeError('Install peft: pip install peft')

    HF_TOKEN = os.getenv('HF_TOKEN', '')
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Load demo dataset ----
    data     = np.load(args.demo_path, allow_pickle=True)
    actions  = torch.from_numpy(data['actions']).float()          # (N, act_dim)
    instrs   = list(data['instructions'])                         # (N,)
    N        = len(actions)
    print(f'[finetune] Dataset: {N} transitions')

    if 'images' not in data:
        print('[finetune] WARNING: dataset has no "images" key. '
              'Fine-tuning requires RGB observations. '
              'Re-collect demos with make_rgbd_env and save RGB frames as "images".')
        print('[finetune] Aborting.')
        return

    images = data['images']  # (N, H, W, 3) uint8

    # ---- Load OpenVLA and apply LoRA ----
    from transformers import AutoModelForVision2Seq, AutoProcessor
    print(f'[finetune] Loading {args.vla_model_id} ...')
    processor = AutoProcessor.from_pretrained(
        args.vla_model_id, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.vla_model_id, token=HF_TOKEN, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=['q_proj', 'v_proj'],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    from PIL import Image

    # ---- Training loop ----
    batch_size = args.batch_size
    for epoch in range(args.epochs):
        indices = np.random.permutation(N)
        total_loss = 0.0
        n_batches  = 0
        for start in range(0, N, batch_size):
            idx_batch = indices[start:start + batch_size]
            batch_imgs   = [Image.fromarray(images[i]) for i in idx_batch]
            batch_instrs = [instrs[i] for i in idx_batch]
            batch_acts   = actions[idx_batch].to(device)

            # format prompt
            prompts = [
                f'In: What action should the robot take to {instr}?\nOut:'
                for instr in batch_instrs
            ]

            inputs = processor(prompts, batch_imgs,
                               return_tensors='pt', padding=True).to(device)

            # Tokenize ground-truth actions as target tokens (simplified)
            act_str = [
                ' '.join(f'{v:.4f}' for v in a.cpu().numpy())
                for a in batch_acts
            ]
            labels = processor.tokenizer(
                act_str, return_tensors='pt', padding=True).input_ids.to(device)

            outputs = model(**inputs, labels=labels)
            loss    = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(f'[finetune] Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}')

    # ---- Save ----
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    print(f'[finetune] Saved fine-tuned model -> {args.output}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='OpenVLA on ManiSkill Combined-v1 (eval / collect / finetune)')
    p.add_argument('--mode',  choices=['eval', 'collect', 'finetune'], default='eval')
    p.add_argument('--robot', default='panda', choices=['panda', 'so100'])
    p.add_argument('--render', action='store_true')

    # Shared
    p.add_argument('--prompt',   default='push first, then pull, pick and stack')
    p.add_argument('--max-steps', type=int, default=10_000)

    # eval
    p.add_argument('--episodes',   type=int, default=3)
    p.add_argument('--vla-model-id', default='openvla/openvla-7b')
    p.add_argument('--llm-model-id',
                   default='meta-llama/Llama-3.1-8B-Instruct')
    p.add_argument('--flash-attn', action='store_true')
    p.add_argument('--unnorm-key', default=None,
                   help='Unnorm key for OpenVLA (default: bridge_orig)')

    # collect
    p.add_argument('--n-demos',  type=int, default=50)
    p.add_argument('--demo-out', default='output/demos/combined_v1.npz')

    # finetune
    p.add_argument('--demo-path',   default='output/demos/combined_v1.npz')
    p.add_argument('--lora-rank',   type=int, default=16)
    p.add_argument('--epochs',      type=int, default=5)
    p.add_argument('--batch-size',  type=int, default=4)
    p.add_argument('--output',      default='output/openvla_finetuned/')

    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.mode == 'eval':
        run_eval(args)
    elif args.mode == 'collect':
        run_collect(args)
    elif args.mode == 'finetune':
        run_finetune(args)
