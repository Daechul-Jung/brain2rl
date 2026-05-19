# Architecture Decision Records

## Philosophy

Every component that encodes an idea — the RL algorithm, the brain conditioner, the environment — must be swappable without touching anything else. Correctness before performance. One file, one responsibility.

---

### ADR-001: PyTorch as the sole ML framework

**Decision**: Use PyTorch exclusively for all model components (tokenizer, brain conditioner, RL agents).

**Reason**: The existing codebase is fully PyTorch. REPPO, ActionClassifier, and VLA components are all PyTorch. Mixing frameworks would fragment the gradient tape across the tokenizer → brain → RL pipeline, breaking the end-to-end backprop requirement.

**Trade-off**: JAX's functional transforms (vmap, jit) would be faster for some RL loops, but the integration cost outweighs the benefit at this stage.

---

### ADR-002: Conv1D EEG tokenizer (not a pretrained foundation model)

**Decision**: Encode EEG using a Conv1D trunk (3× Conv1D → BN → ReLU → MaxPool → Linear) trained end-to-end with the RL loss.

**Reason**: EEG signals are low-dimensional time series (~8–64 channels, ~250 Hz). A lightweight Conv1D trunk can be trained jointly with the RL agent so the token representations are shaped specifically to maximize reward — something a frozen foundation model cannot do. This also keeps VRAM usage well within the 4070's 12 GB budget.

**Trade-off**: No transfer learning from large EEG datasets. Acceptable because our dataset is subject-specific and task-specific.

---

### ADR-003: One token per RL timestep

**Decision**: The tokenizer produces exactly `T_rl` tokens from an EEG segment, one token per RL step, via `adaptive_avg_pool1d(output_size=T_rl)` on the Conv1D feature map.

**Reason**: This gives the brain conditioner a natural "clock" — at step `t` it reads `token[t]`, which encodes brain intent at that moment. It also decouples EEG sampling rate from RL control frequency without any manual resampling.

**Trade-off**: The temporal alignment is approximate (pooling smooths across raw EEG samples). Exact alignment would require knowing the EEG-to-RL timestep mapping from the recording setup.

---

### ADR-004: BrainConditioner abstract interface for pluggable ideas

**Decision**: All EEG-conditioning ML components must implement `research/brain/base.py::BrainConditioner`, which exposes `forward(token_seq, t) → dict` and the `token_dim` / `action_dim` properties.

**Reason**: The user's workflow is to propose new ideas and implement them. Without a fixed interface, every new idea would require changes in the agent and training loop. With this interface, only `research/brain/` and `research/ideas/` need to change.

**Trade-off**: The interface constrains output format (must return a dict with at least `delta_action`). New ideas that produce something fundamentally different (e.g., a full trajectory) would require an interface extension.

---

### ADR-005: Causal transformer as the first brain conditioner (TransformerDelta)

**Decision**: The initial brain conditioner is a causal transformer encoder over EEG tokens up to the current step `t`, outputting `delta_action` (action perturbation) and `alpha` (gating scalar).

**Reason**: A transformer can attend to the history of EEG tokens, not just the current one — this captures evolving brain intent over the episode. The delta-and-gate design lets the RL base policy (REPPO) dominate when EEG is noisy (`alpha → 0`).

**Trade-off**: Requires storing the full token sequence during rollout, adding `O(T_rl × token_dim)` memory per episode. Acceptable for sequences up to ~512 steps.

---

### ADR-006: REPPO for on-policy RL, TD3+BC for offline RL

**Decision**: Use REPPO (Relative Entropy Pairwise Policy Optimization) for online RL and TD3+BC for offline RL with D4RL datasets.

**Reason**: REPPO has shown strong results on continuous control tasks and fits naturally with EEG conditioning (its categorical Q-learning handles sparse and shaped rewards robustly). TD3+BC is a simple, well-understood offline baseline that can warm-start the EEG-conditioned agent from robot demonstrations.

**Trade-off**: REPPO does not use a replay buffer, so every rollout is on-policy — sample efficiency is lower than SAC. This is acceptable given the relatively low wall-clock cost of MuJoCo/ManiSkill simulation.

---

### ADR-007: MuJoCo + ManiSkill as simulation environments; ROS 2 Gazebo deferred

**Decision**: Develop and validate on MuJoCo (Gymnasium) and ManiSkill2 (Combined-v1). Port to ROS 2 Gazebo only after simulation success.

**Reason**: MuJoCo and ManiSkill are faster to iterate on, have well-tested reset/reward logic, and support parallelized environments (ManiSkill). Gazebo integration requires URDF validation and real-time ROS 2 control, which adds significant overhead before the core algorithm is validated.

**Trade-off**: MuJoCo physics do not perfectly match the real OpenArm robot. Sim-to-real transfer will require domain randomization or fine-tuning — out of scope for now.

---

### ADR-008: End-to-end backprop from RL loss through tokenizer

**Decision**: During the policy update step, raw EEG segments are re-tokenized inside the computation graph so that RL loss gradients flow through the brain conditioner **and** the Conv1D tokenizer.

**Reason**: If the tokenizer is frozen, it produces tokens that are optimal for classification, not for maximizing reward. End-to-end training shapes the tokens to be maximally useful to the conditioner, which in turn shapes the actions to maximize reward.

**Trade-off**: Storing raw EEG segments in the trajectory buffer increases memory usage. For long rollouts with many parallel environments, this may require gradient checkpointing.
