# Architecture

## Directory Structure

```
brain2rl/
│
├── research/                          # Research environment (plug-and-play)
│   ├── eeg/
│   │   ├── __init__.py
│   │   └── tokenizer.py               # EEGTokenizer: (B,C,T_eeg) → (B,T_rl,token_dim)
│   │
│   ├── brain/                         # Pluggable brain conditioner (ML idea slot)
│   │   ├── __init__.py
│   │   ├── base.py                    # BrainConditioner ABC
│   │   ├── transformer_delta.py       # CURRENT: causal transformer → delta_action + alpha
│   │   └── diffusion_conditioned.py   # FUTURE: diffusion conditioned on EEG token
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── eeg_reppo.py               # EEGRePPOAgent (REPPO + BrainConditioner)
│   │   └── eeg_td3.py                 # EEGTd3Agent (TD3+BC + BrainConditioner, offline)
│   │
│   ├── envs/
│   │   ├── __init__.py
│   │   └── registry.py                # make_env(env_id, **kwargs) factory
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── config.py                  # ExperimentConfig dataclass
│   │   ├── train_reppo.py             # Online RL training entry point
│   │   └── evaluate.py                # Evaluation / rollout entry point
│   │
│   └── ideas/                         # Idea specifications (write here first)
│       ├── README.md                  # How to write an idea + interface contract
│       ├── 001_transformer_delta.md   # Current idea: causal transformer delta
│       └── 002_diffusion_conditioned.md  # Future idea: diffusion conditioned action
│
├── models/                            # Existing model implementations
│   ├── classification/                # ActionClassifier CNN (EEG → action class)
│   ├── tokenization/                  # EEGRLTokenizer, IntentTransformerDelta
│   └── rl/
│       ├── agents/                    # RePPOAgent, PPOAgent, SACAgent, TD3+BC
│       ├── utils/                     # Networks, train utilities, buffers
│       ├── envs/                      # OpenArm MuJoCo environment
│       ├── mani_skill/                # ManiSkill tasks + launch scripts
│       └── vla/                       # Custom VLA (Octo-style)
│
├── core/                              # Pipeline runners (end-to-end entry points)
│   ├── classification_pipeline.py
│   ├── tokenizer_vla_pipeline.py
│   ├── tokenizer_rl_pipeline.py
│   └── main_pipeline.py
│
├── tests/                             # Layered test suite
│   ├── test_01_tokenizer.py           # Layer 1: EEGTokenizer shapes and forward
│   ├── test_02_brain.py               # Layer 2: BrainConditioner interface + TransformerDelta
│   ├── test_03_agent.py               # Layer 3: EEGRePPOAgent integration
│   ├── test_04_envs.py                # Layer 4: Environment creation and step
│   └── test_05_integration.py         # Layer 5: Full mini-rollout (CPU, dummy data)
│
├── data/                              # Raw EEG data (gitignored)
├── output/                            # Training artifacts (gitignored)
├── scripts/
│   ├── execute.py                     # Harness phase runner
│   └── download_d4rl.py               # D4RL dataset downloader
├── docs/
│   ├── ADR.md
│   ├── PRD.md
│   └── ARCHITECTURE.md
└── .agents/                           # Claude Code config (gitignored)
```

---

## Patterns

### Plugin Pattern (BrainConditioner)

The `BrainConditioner` abstract class in `research/brain/base.py` is the single extension point for new ML ideas:

```python
class BrainConditioner(nn.Module, ABC):
    @abstractmethod
    def forward(self, token_seq: Tensor, t: int) -> dict:
        # token_seq: (B, t+1, token_dim)  — EEG tokens up to and including step t
        # Returns: {'delta_action': (B, action_dim), 'alpha': (B, 1)}
        ...

    @property
    @abstractmethod
    def token_dim(self) -> int: ...

    @property
    @abstractmethod
    def action_dim(self) -> int: ...
```

To add a new idea:
1. Write a spec in `research/ideas/NNN_idea_name.md`
2. Implement `YourConditioner(BrainConditioner)` in `research/brain/your_conditioner.py`
3. Pass it to `EEGRePPOAgent(brain=YourConditioner(...))`
4. Run `pytest tests/test_02_brain.py` to verify the interface

### Factory Pattern (Environment Registry)

```python
env = make_env("HalfCheetah-v4")           # MuJoCo Gymnasium
env = make_env("Combined-v1", robot="panda") # ManiSkill
env = make_env("OpenArm-v0", render=True)   # Custom OpenArm MuJoCo
```

---

## Data Flow

```
┌────────────────────────────────────────────────────────────┐
│  Training loop (research/experiments/train_reppo.py)       │
│                                                            │
│  EEG segment (B, C, T_eeg)                                 │
│      │                                                     │
│      ▼                                                     │
│  EEGTokenizer (Conv1D)                                     │
│      │  (B, T_rl, token_dim)   ← one token per RL step    │
│      ▼                                                     │
│  token_seq[:t+1]  ──────────────────────────┐             │
│                                              ▼             │
│  Observation (B, obs_dim)         BrainConditioner         │
│      │                            (TransformerDelta)       │
│      ▼                                      │              │
│  REPPO Actor                      delta_action, alpha      │
│      │  base_mean, base_log_std             │              │
│      └──────────────────────────────────────┘             │
│                        │                                   │
│                        ▼                                   │
│           final_mean = base_mean + alpha * delta           │
│           action ~ TanhNormal(final_mean, exp(log_std))    │
│                        │                                   │
│                        ▼                                   │
│                 Environment step                           │
│                        │                                   │
│                        ▼                                   │
│              RL Loss (REPPO objective)                     │
│                        │                                   │
│          ┌─────────────┴──────────────┐                   │
│          ▼                            ▼                    │
│      REPPO update              BrainConditioner update     │
│                                EEGTokenizer update         │
│          (all via same backward pass)                      │
└────────────────────────────────────────────────────────────┘
```

---

## State Management

- **Observation normalization**: `EmpiricalNormalizer` (running mean/std, updated online)
- **EEG token buffer**: token sequence accumulated per episode in a `(T_rl, token_dim)` tensor; reset on episode start
- **Trajectory buffer**: TensorDict with shape `(T, N)` where T = rollout steps, N = parallel envs; also stores raw EEG segments for re-tokenization during update
- **Checkpoints**: saved to `output/{experiment_name}/checkpoint_{step}.pth`; format: `{actor, critic, tokenizer, brain, step, config}`
