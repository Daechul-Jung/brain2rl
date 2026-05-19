# Pure RL Ideas

This folder holds specifications for pure RL ideas — algorithm changes, reward shaping experiments,
network architecture modifications — that do **not** involve EEG data.

Validated ideas here are candidates for transfer to the EEG RL track.

---

## How to propose a new idea

1. Create `NNN_short_name.md` (e.g., `002_shaped_reward.md`).
2. Fill in the template below.
3. Implement in `research/pure_rl/agents/your_idea.py` (or modify an existing agent).
4. Run `pytest tests/test_06_pure_rl.py tests/test_07_rewards.py` to verify.
5. If successful, decide whether to port it to the EEG track.

---

## Template

```markdown
# Idea NNN: {Short Name}

## Status
[ ] Proposed  [ ] In progress  [ ] Implemented  [ ] Archived

## Motivation
What problem does this solve? What do you expect to improve?

## Change
What exactly changes from the current baseline?
- Algorithm: REPPO / TD3+BC / new
- Reward: shaped / sparse / ...
- Network: larger / different architecture / ...

## Expected behaviour
How do you know it's working?
- Reward curve should show X
- Specific metric to check: ...

## Risks
What could go wrong?

## Transfer to EEG track
How would this idea be applied with EEG conditioning if it works?

## Implementation
`research/pure_rl/agents/your_idea.py` or note existing agent used.
```

---

## Active ideas

| # | File | Status | Description |
|---|------|--------|-------------|
| 001 | [001_reppo_baseline.md](001_reppo_baseline.md) | In progress | REPPO on ManiSkill pick-and-place (baseline) |
