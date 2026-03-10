# Brain2RL – CLAUDE.md

This file documents the project architecture, conventions, and key commands for Claude Code.

---

## Project Overview

**Brain2RL** converts EEG brain signals representing motor intentions (pick, push, pull, stack) into robot control signals. There are two parallel tracks:

| Track | EEG → | Used for |
|-------|-------|----------|
| **VLA track** | Action language (text) via LLM | OpenVLA / custom VLA robot control |
| **RL track** | Action delta (continuous vector) | REPPO reinforcement learning |

### Data
- **EEG data**: `data/alpha_wave_arm_data/alpha-training-task/` — raw `.txt` EEG recordings from 8 subjects across 4 tasks
- **ROS bag data**: `data/alpha_wave_arm_data/ROS-message-data/` — robot trajectories for VLA training

---

## Directory Structure

```
brain2rl/
├── core/                          # Pipeline runners (entry points)
│   ├── classification_pipeline.py # EEG → action classification (CNN)
│   ├── tokenizer_vla_pipeline.py  # EEG → LLM soft-prompt tokens
│   ├── tokenizer_rl_pipeline.py   # EEG → action delta + REPPO training
│   ├── tokenization_pipeline.py   # Legacy tokenizer (classifier tokens → RL)
│   ├── rl_training_pipeline.py    # General RL training utilities
│   └── main_pipeline.py           # End-to-end orchestrator
│
├── models/
│   ├── classification/            # CNN action classifier
│   │   ├── action_classifier.py   # ActionClassifier (CNN trunk + heads)
│   │   ├── data_utilities.py      # load_sensor_data, preprocess_multilabel
│   │   └── segment_utils.py       # contiguous_segments, SegmentDataset
│   │
│   ├── tokenization/              # Transformer tokenizer (legacy)
│   │   └── brain_tokenizer_transformer.py
│   │
│   └── rl/
│       ├── agents/
│       │   ├── reppo.py           # RePPOAgent (main RL algorithm)
│       │   ├── ppo.py             # PPOAgent
│       │   └── sac.py             # SACAgent
│       ├── utils/
│       │   ├── reppo_network.py   # Actor / Critic networks
│       │   ├── train.py           # train_reppo(), train_agent()
│       │   └── any_utils.py       # compute_gve, EmpiricalNormalizer, etc.
│       ├── envs/
│       │   └── openarm_mj_env.py  # OpenArm MuJoCo environment
│       ├── vla/                   # Custom VLA (Octo-style, PyTorch)
│       │   └── model/components/  # action_heads, vit_encoder, tokenizers, diffusion
│       └── mani_skill/
│           ├── tasks/multiple_tasks_env.py  # Combined-v1 (push/pull/pick/stack)
│           ├── thinkers/
│           │   ├── openvla_policy.py        # OpenVLA-7b wrapper
│           │   ├── task_thinker.py          # LLaMA task planner
│           │   └── sequential_vla_agent.py  # Multi-task execution agent
│           └── scripts/
│               ├── launch.py               # OpenVLA + LLM launch
│               └── launch_reppo.py         # REPPO training on ManiSkill
│
├── data/
│   └── alpha_wave_arm_data/
│       ├── alpha-training-task/   # Raw EEG .txt files (s{1-8}_d{1-2}_training.txt)
│       └── ROS-message-data/      # ROS bag files
│
├── notebooks/
│   └── eeg_data_analysis.ipynb   # EEG data exploration and visualization
│
├── ros2_ws/                       # ROS2 Humble workspace (OpenArm)
├── external/openarm_mujoco/       # OpenArm MuJoCo XML files
└── output/                        # Training artifacts (gitignored)
```

---

## Pipeline Flows

### 1. EEG Classification (prerequisite for both tracks)
```
data/alpha_wave_arm_data/alpha-training-task/  →  processed train.csv
    ↓
core/classification_pipeline.py
    ActionClassifier (CNN) trains on EEG segments
    Output: output/classifier/best_classifier.pth
            output/classifier/segment_tokens_K16.npz
```

### 2. VLA Track (language prediction)
```
output/classifier/segment_tokens_K16.npz
    ↓
core/tokenizer_vla_pipeline.py   (EEGVLATokenizer)
    Trains CNN + projection → LLM embedding space
    Output: output/tokenizer_vla/vla_tokenizer.pth
            output/tokenizer_vla/cnn_tokens.npy
    ↓
inference_with_llm(X_eeg, llama_model, hf_tokenizer)
    → action language strings ("pick the cube", "push to target", ...)
    ↓
models/rl/mani_skill/thinkers/openvla_policy.py  (OpenVLA-7b)
    → 7D robot actions
```

### 3. RL Track (action delta)
```
data/alpha_wave_arm_data/alpha-training-task/  →  processed train.csv
    ↓
core/tokenizer_rl_pipeline.py   (EEGRLTokenizer + EEGActionHead)
    Trains end-to-end with REPPO on ManiSkill Combined-v1
    Output: output/rl_tokenizer/rl_tokenizer.pth

    final_action = actor_base_action + EEGActionHead(obs, eeg_tokens)
```

---

## Key Commands

### Step 1 – Train EEG Classifier
```bash
cd /home/jdcjdb/brain2rl
python core/classification_pipeline.py \
    --train-csv data/processed/train.csv \
    --output-dir output \
    --train-mode segments \
    --pool-k 16 \
    --export-segment-tokens true
```

### Step 2a – Train VLA Tokenizer
```bash
python core/tokenizer_vla_pipeline.py \
    --train-csv data/processed/train.csv \
    --pool-k 16 \
    --llm-dim 4096 \
    --epochs 50 \
    --output-dir output/tokenizer_vla
```

### Step 2b – Train RL Tokenizer (requires ManiSkill)
```bash
python core/tokenizer_rl_pipeline.py \
    --eeg-csv data/processed/train.csv \
    --env Combined-v1 \
    --total-steps 200000 \
    --pool-k 16 \
    --output output/rl_tokenizer/rl_tokenizer.pth
```

### Step 3 – Train REPPO on ManiSkill (without EEG, for baseline)
```bash
# Headless (parallel envs, fast)
python models/rl/mani_skill/scripts/launch_reppo.py \
    --num-envs 4 \
    --total-steps 500000 \
    --output output/reppo_maniskill/checkpoint.pth

# With GUI (single env, slow)
python models/rl/mani_skill/scripts/launch_reppo.py \
    --render \
    --total-steps 50000
```

### Step 4 – Launch OpenVLA + LLM Planner on ManiSkill
```bash
HF_TOKEN=hf_xxx python models/rl/mani_skill/scripts/launch.py \
    --prompt "push first, then pull and pick and stack"
```

### Train on MuJoCo OpenArm (plain REPPO)
```bash
python models/rl/launch/try_mj.py --algo reppo --render
```

### ROS2 Workspace
```bash
cd ros2_ws
colcon build --packages-select openarm_description openarm_ros
source install/setup.bash
ros2 launch openarm_ros openarm_gazebo.launch.py
```

---

## Architecture Notes

### ActionClassifier (CNN trunk)
- Input: `(B, C, T)` — batch, EEG channels, time steps
- Architecture: 3× Conv1D → BatchNorm → ReLU → MaxPool → Dropout → Linear projection
- Output tokens: `(B, K, 128)` — K pooled 128-dim tokens per segment
- Heads: `behavior_logits (B, n_beh)`, `gesture_logits (B, n_ges)`

### EEGVLATokenizer (`core/tokenizer_vla_pipeline.py`)
- Reuses CNN trunk from ActionClassifier
- Adds `to_llm` projection: `128 → 2×128 → llm_dim` (default 4096)
- Training: cross-entropy on action label classification
- Inference: `get_llm_tokens(X)` → `(N, K, llm_dim)` for LLM prefix injection

### EEGRLTokenizer + EEGActionHead (`core/tokenizer_rl_pipeline.py`)
- `EEGRLTokenizer`: CNN trunk → K tokens (same as VLA tokenizer but separate weights)
- `EEGActionHead`: cross-attention(obs_query, eeg_keys/values) → `action_delta`
- Final action: `tanh(base_action + scale * action_delta)`, scale default 0.3
- Trained jointly with REPPO from RL loss

### RePPOAgent (`models/rl/agents/reppo.py`)
- Categorical Q-learning with 151 atoms, Vmin=-2500, Vmax=2500
- Actor: entropy + KL regularization (temperature α, Lagrange λ)
- Uses `EmpiricalNormalizer` for observation normalization
- Key methods: `collect()`, `update_actor()`, `update_critic()`, `save()`, `load()`

### Combined-v1 ManiSkill Environment
- Tasks in sequence: push → pull → pick → stack (configurable)
- Stage transitions on success; +5.0 bonus reward per stage
- Supported robots: `panda`, `so100`
- Obs modes: `state` (fast), `rgbd` (images), `pointcloud`

---

## Conventions

- **Python 3.10+** (uses `int | None` union syntax)
- All models use `torch.device` for device placement; check `torch.cuda.is_available()`
- Observation normalization via `EmpiricalNormalizer` (running mean/std)
- EEG segments are stored as `(N, C, T)` tensors; channel dimension is **second**
- Action space is always `[-1, 1]` (tanh-squashed); rescaled to env range in launch scripts
- Save paths default to `output/<component>/` — this directory is gitignored
- Never commit large data files or model checkpoints

---

## Known Issues / TODOs

- `multiple_tasks_env.py`: `_set_stage()` has a bug on line 77 (`==` should be `=`)
- `core/rl_training_pipeline.py`: environment initialization is a placeholder (`self.env = ...`)
- VLA model (`models/rl/vla/`): `DiscreteActionHead` and data loaders are incomplete
- ROS2 Gazebo: OpenArm spawns without robot body (joint offset / base link issue in URDF)
- `core/tokenizer_rl_pipeline.py`: EEG delta regularization loss is minimal; full joint backprop through REPPO loss is a future improvement

---

## EEG Data Format

Raw files: `data/alpha_wave_arm_data/alpha-training-task/s{N}_d{M}_training.txt`
- Tab/space separated
- Columns: timestamp + EEG channels (number varies by recording)
- Label column: action label (behavior class)
- See `notebooks/eeg_data_analysis.ipynb` for exploration

Pre-processing via `models/classification/data_utilities.py`:
- `load_sensor_data(csv_path, group_by='sequence_id')` → `(X_raw, y_str, groups, df)`
- `preprocess_multilabel(X_raw, y_str)` → `(X, y_enc, scaler, encoders)`
