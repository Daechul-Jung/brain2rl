# Idea 002: Diffusion-Conditioned Action

## Status
[x] Proposed  [ ] In progress  [ ] Implemented  [ ] Archived

## Motivation
Idea 001 (TransformerDelta) produces an *additive perturbation* to the REPPO base action.
This means the EEG signal can only adjust the base action, not replace it.

A diffusion model conditioned on EEG tokens can generate a *complete action distribution*
from scratch, bypassing the REPPO base entirely. This is more expressive: the model learns
the full action manifold for each brain intent rather than just a correction.

Diffusion also naturally handles multimodal action distributions (e.g., two plausible ways
to pick an object) — which TanhNormal in REPPO cannot.

## Architecture

### Inputs
- `token_seq` : (B, t+1, token_dim) — EEG tokens up to step t
- `t`         : current RL timestep
- (internal) noisy action `x_k` at diffusion step k, timestep embedding `k`

### Processing

```
token_seq
  │ TransformerEncoder (same as Idea 001, or shared weights)
  │ → eeg_context (B, d_model)   via [ACT] cross-attention
  ▼
Diffusion denoising network:
  Input: cat(x_k, eeg_context, timestep_emb)
  Architecture: UNet-1D or MLP with FiLM conditioning on (eeg_context, k)
  Output: predicted noise ε_θ(x_k, eeg_context, k)

Inference (DDIM, ~10 steps):
  x_K = N(0, I)
  for k = K, K-1, ..., 0:
      x_{k-1} = DDIM_step(x_k, ε_θ(x_k, eeg_context, k))
  final_action = tanh(x_0)
```

### Outputs
- `delta_action` : set to `final_action - base_action` so the interface is satisfied.
- `alpha`        : fixed at 1.0 (diffusion replaces base action, gate is not meaningful here).

## Integration with REPPO

Two options:

**Option A (compatible)**: Express diffusion output as a delta and use the standard combination.
```
final_action = tanh(diffusion_output(eeg_context))   # (B, action_dim)
delta = final_action - base_action.detach()
alpha = ones(B, 1)
```

**Option B (replace)**: Use diffusion output directly as the action; disable REPPO actor.
- Requires a modified agent that skips the REPPO actor forward pass.
- The REPPO critic still evaluates `Q(obs, diffusion_action)`.

Recommend starting with Option A to maintain backward compatibility.

## Key hyperparameters

| Name | Default | Description |
|------|---------|-------------|
| `K` | 100 | Total diffusion steps (training) |
| `K_infer` | 10 | DDIM denoising steps at inference |
| `d_model` | 256 | Denoising network hidden width |
| `n_layers` | 4 | UNet / MLP depth |
| `beta_schedule` | cosine | Noise schedule |
| `conditioning` | FiLM | How eeg_context enters the denoising network |

## Expected behaviour

- Should produce multimodal action distributions when EEG encodes ambiguous intent.
- Inference slower than Idea 001 due to K_infer denoising steps; use DDIM to keep it fast.
- Should outperform Idea 001 on tasks where the action distribution is not unimodal.

## Risks / open questions

- **Training stability**: Diffusion + RL is notoriously tricky. Consider pre-training the
  diffusion model on robot demonstrations (D4RL) before fine-tuning with RL.

- **Inference latency**: Even with DDIM (10 steps), diffusion adds latency compared to a
  single forward pass. Profile this against the RL control frequency.

- **Credit assignment**: The diffusion output is a stochastic sample; log_prob is non-trivial
  to compute. May need score-function estimator or REINFORCE for the brain optimizer gradient.

- **VRAM**: Running a diffusion network alongside REPPO may exceed the 4070's 12 GB budget
  at large batch sizes.

## Implementation file

`research/brain/diffusion_conditioned.py` — class `DiffusionConditioned`
(Placeholder exists; implement when ready.)

## References

- DDPM: Ho et al. 2020 — https://arxiv.org/abs/2006.11239
- DDIM: Song et al. 2020 — https://arxiv.org/abs/2010.02502
- Diffusion Policy: Chi et al. 2023 — https://arxiv.org/abs/2303.04137
- DPPO (Diffusion + PPO): Ren et al. 2024 — https://arxiv.org/abs/2409.00588
