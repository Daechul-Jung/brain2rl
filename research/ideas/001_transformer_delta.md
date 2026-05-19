# Idea 001: Transformer Delta

## Status
[x] Proposed  [x] In progress  [ ] Implemented  [ ] Archived

## Motivation
REPPO produces a good base action distribution from robot observations alone, but it has no
knowledge of the human's intent. EEG signals represent motor intention before a movement
happens, so conditioning on EEG tokens should shift the action distribution toward what the
human operator wants to do.

A causal transformer over the token history captures *evolving intent* — e.g., the agent
can tell from EEG that a pick intent is becoming stronger over the first 10 steps.

The alpha gate suppresses the delta when EEG is noisy or ambiguous, letting the REPPO
base policy dominate.

## Architecture

### Inputs
- `token_seq` : (B, t+1, token_dim) — EEG tokens from step 0 to t
- `t`         : current RL timestep (used for positional encoding slicing)

### Processing

```
token_seq
  │ Linear(token_dim → d_model) + sinusoidal PE
  ▼
TransformerEncoder(n_layers, n_heads, causal mask)
  │  (B, t+1, d_model)
  ▼
Learned [ACT] query cross-attends to encoded tokens
  │  MultiheadAttention(query=[ACT], key/value=encoded_tokens)
  │  (B, 1, d_model) → squeeze → (B, d_model)
  ▼  act_feat
  ├──► delta_head: Linear → ReLU → Linear → (B, action_dim)
  └──► alpha_head: Linear → ReLU → Linear → Sigmoid → (B, 1) in [0, 1]
```

### Outputs
- `delta_action` : (B, action_dim) — raw (pre-tanh) additive perturbation
- `alpha`        : (B, 1) — gate; 0 = ignore EEG, 1 = full delta applied

## Integration with REPPO

```
base_action = REPPO_actor(obs).sample()             # (B, action_dim)
delta, alpha = TransformerDelta(token_seq, t)
final_action = tanh(base_action + alpha * delta)    # (B, action_dim) in (-1, 1)
```

The final action is sampled and stepped in the environment.
During update, RL loss backprops through: REPPO actor + TransformerDelta + EEGTokenizer.

## Key hyperparameters

| Name | Default | Description |
|------|---------|-------------|
| `d_model` | 256 | Transformer hidden width |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 2 | Encoder layers |
| `dropout` | 0.1 | Dropout rate |
| `max_seq_len` | 512 | Max episode length (PE buffer size) |
| `stochastic` | False | If True, also output log_std for delta |

## Expected behaviour

- Early training: alpha ≈ 0.5, delta ≈ random → marginal effect.
- After convergence: alpha is higher when EEG clearly encodes an action, lower when noisy.
- With pick EEG: base_action + delta biased toward pick motion; without EEG: reverts to REPPO.
- Ablation: setting alpha=0 at inference should produce the same reward as vanilla REPPO.

## Risks / open questions

- **Reward credit assignment**: EEG tokens are fixed per episode; the transformer sees the same
  tokens at every step. Does it learn temporal structure or just a fixed-episode bias?
  → Check: does the causal mask actually matter (compare to non-causal version)?

- **Token quality**: Conv1D trunk trained from scratch may produce uninformative tokens early.
  Warm-starting from the ActionClassifier checkpoint (call `EEGTokenizer.load_pretrained_trunk`)
  may help.

- **Alpha collapse**: alpha → 0 is an easy escape from the EEG loss. Monitor alpha_mean during
  training; if it stays near 0, increase the brain loss coefficient.

## Implementation file

`research/brain/transformer_delta.py` — class `TransformerDelta`
