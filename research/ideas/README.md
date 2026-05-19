# Ideas Folder

This folder holds specifications for EEG brain conditioning ideas.
Each idea is described **before** implementation so the design intent is clear.

---

## How to propose a new idea

1. Create a file `NNN_short_name.md` (e.g., `003_vq_codebook.md`).
2. Fill in the template below.
3. Share it and discuss with collaborators.
4. Once approved, implement in `research/brain/your_idea.py` as a `BrainConditioner` subclass.
5. Run `pytest tests/test_02_brain.py` to verify the interface contract.

---

## Idea template

```markdown
# Idea NNN: {Short Name}

## Status
[ ] Proposed  [ ] In progress  [ ] Implemented  [ ] Archived

## Motivation
Why do you think this idea will work better than the current approach?
What limitation of the previous idea does this address?

## Architecture

### Inputs
- token_seq : (B, t+1, token_dim) — EEG tokens up to current step t
- t         : current RL timestep

### Processing
Describe the neural network architecture step by step.

### Outputs (must satisfy BrainConditioner interface)
- delta_action : (B, action_dim) — action perturbation
- alpha        : (B, 1) — gate/confidence in [0, 1]
- (optional) other keys for logging/debugging

## Integration with REPPO
How does this idea change the action computation?
  final_mean = base_mean + alpha * delta_action  ← default
  or describe a different combination.

## Key hyperparameters
| Name | Default | Description |
|------|---------|-------------|
|      |         |             |

## Expected behaviour
What observable behaviour do you expect when this works correctly?
How would you know it's actually using the EEG signal?

## Risks / open questions
What could go wrong? What would you check first?

## Implementation file
`research/brain/your_idea.py`
```

---

## Active ideas

| # | File | Status | Description |
|---|------|--------|-------------|
| 001 | [001_transformer_delta.md](001_transformer_delta.md) | In progress | Causal transformer → action delta + alpha gate |
| 002 | [002_diffusion_conditioned.md](002_diffusion_conditioned.md) | Proposed | Diffusion model conditioned on EEG token → full action |
