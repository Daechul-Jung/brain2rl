# Idea 001: REPPO Baseline on ManiSkill Pick-and-Place

## Status
[x] Proposed  [x] In progress  [ ] Implemented  [ ] Archived

## Motivation
Establish a pure RL baseline on `PickCube-v1` / `PushCube-v1` / `StackCube-v1` / `PullCube-v1`
before adding EEG conditioning. A working baseline tells us:
- What reward scale the task uses
- How many steps are needed to reach a reasonable success rate
- Whether the observation space is informative enough

## Change
No change from the existing RePPOAgent. Run as-is with ManiSkill pick-and-place.

## Expected behaviour
- `PickCube-v1`: success rate > 50% within 500k steps
- Reward starts near 0 (sparse) or increases slowly (dense); check `raw_rewards` curve
- Episode length decreases as policy improves

## Risks
- ManiSkill 3.0 returns obs as `torch.Tensor (N, obs_dim)` — need to handle tensor inputs in agent
- GPU memory: with `num_envs=4` the simulation uses GPU; may need `num_envs=1` on 12 GB VRAM

## Transfer to EEG track
Once baseline works, run the same env with `EEGRePPOAgent` and a `PickCube` EEG segment.
The EEG conditioner should push reward higher than the pure baseline.

## Implementation
Uses `RePPOAgent` from `models/rl/agents/reppo.py` directly.
Training script: `research/pure_rl/experiments/train_reppo.py`
