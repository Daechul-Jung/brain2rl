import torch

import torch

def _ensure_tensor(x, device=None, dtype=torch.float32):
    if torch.is_tensor(x):
        if device is not None and x.device != device:
            x = x.to(device)
        if dtype is not None and x.dtype != dtype and x.dtype.is_floating_point:
            x = x.to(dtype)
        return x
    return torch.as_tensor(x, device=device, dtype=dtype)

def _ensure_2d(x):
    """-> [B, D]"""
    if x.dim() == 0:  # [] -> [1,1]
        return x.view(1, 1)
    if x.dim() == 1:  # [D] -> [1,D]
        return x.unsqueeze(0)
    if x.dim() == 2:  # [B,D]
        return x
    return x.flatten(start_dim=1)  # [B, ...]

def _ensure_B1(x):
    """-> [B, 1] for scalar-like tensors (values, rewards, dones, truncations)."""
    if x.dim() == 0:  # [] -> [1,1]
        return x.view(1, 1)
    if x.dim() == 1:  # [B] -> [B,1]
        return x.unsqueeze(-1)
    return x  # [B,1] already

def _ensure_logp(x, as_B1=False):
    """
    Accepts [B], [B,1], or [B, A] (unsummed).
    Returns [B] by default, or [B,1] if as_B1=True.
    """
    if x.dim() == 0:
        return x.view(1, 1) if as_B1 else x.view(1)
    if x.dim() == 1:  # [B]
        return x.unsqueeze(-1) if as_B1 else x
    if x.dim() == 2:
        if x.shape[-1] == 1:
            return x if as_B1 else x.squeeze(-1)  # [B,1] -> [B]
        # [B, A] -> sum over action dim to get event logp
        x = x.sum(-1)  # [B]
        return x.unsqueeze(-1) if as_B1 else x
    # higher dims: flatten tail first
    x = x.flatten(start_dim=1)  # [B, -1]
    x = x.sum(-1)               # [B]
    return x.unsqueeze(-1) if as_B1 else x

def wrap_batch_dim(
    observation,
    critic_observation,
    action,
    log_prob,
    rewards,
    raw_reward,
    next_features,
    next_value_batch,
    dones_batch,
    truncated_batch,
    device=None,
    logprob_as_B1=False,
):
    """
    Normalize shapes for TensorDict storage & updates.

    Returns:
      observation_b        [B, obs_dim]
      critic_observation_b [B, obs_dim]
      action_b             [B, act_dim]
      log_prob_b           [B] or [B,1] (controlled by logprob_as_B1)
      rewards_b            [B,1]
      raw_reward_b         [B,1]
      next_features_b      [B, feat_dim]
      next_value_b         [B,1]
      dones_b              [B,1]
      truncations_b        [B,1]
    """
    # Cast to tensors on the right device first
    observation        = _ensure_tensor(observation,        device)
    critic_observation = _ensure_tensor(critic_observation, device)
    action             = _ensure_tensor(action,             device)
    log_prob           = _ensure_tensor(log_prob,           device)
    rewards            = _ensure_tensor(rewards,            device)
    raw_reward         = _ensure_tensor(raw_reward,         device)
    next_features      = _ensure_tensor(next_features,      device)
    next_value_batch   = _ensure_tensor(next_value_batch,   device)
    dones_batch        = _ensure_tensor(dones_batch,        device, dtype=torch.float32)
    truncated_batch    = _ensure_tensor(truncated_batch,    device, dtype=torch.float32)

    # Enforce shapes
    observation_b        = _ensure_2d(observation)
    critic_observation_b = _ensure_2d(critic_observation)
    action_b             = _ensure_2d(action)
    log_prob_b           = _ensure_logp(log_prob, as_B1=logprob_as_B1)
    rewards_b            = _ensure_B1(rewards)
    raw_reward_b         = _ensure_B1(raw_reward)
    next_features_b      = _ensure_2d(next_features)
    next_value_b         = _ensure_B1(next_value_batch)
    dones_b              = _ensure_B1(dones_batch)
    truncations_b        = _ensure_B1(truncated_batch)

    return (
        observation_b,
        critic_observation_b,
        action_b,
        log_prob_b,
        rewards_b,
        raw_reward_b,
        next_features_b,
        next_value_b,
        dones_b,
        truncations_b,
    )


def _ensure_batch(x, device, N: int):
    t = torch.as_tensor(x, device=device)
    if t.ndim == 1:
        t = t.unsqueeze(0)            # (1, dim)
    if t.shape[0] != N:
        if t.ndim == 3 and t.shape[0] == 1 and t.shape[1] == N:
            t = t.squeeze(0)          # (N, dim)
        elif t.ndim == 2 and N == 1 and t.shape[0] > 1:
            t = t[:1]                 # keep first
        elif N == 1 and t.ndim > 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        elif t.shape[0] != N:
            raise RuntimeError(f"Expected leading batch {N}, got {t.shape}")
    return t


def _env_shape(env):
    num_envs = getattr(env, "num_envs", 1)
    max_steps = getattr(env, "max_episode_steps", 5000)
    asymmetric = getattr(env, "asymmetric_obs", False )
    return num_envs, max_steps, asymmetric

def _to_tensor(input, device, dtype=torch.float32):
    """
    Converting input to tensor with the given device
    """
    if torch.is_tensor(input):
        return input.to(device = device, dtype=dtype if input.dtype.is_floating_point else input.dtype)
    
    return torch.as_tensor(input, device=device, dtype=dtype)

def _split_step_return(returns):
    if len(returns) == 5:
        next_observation, reward, done, truncated, info = returns 
    else:
        next_observation, reward, done, info = returns 
        truncated = torch.zeros_like(done, dtype=torch.bool) if torch.is_tensor(done) else False
    return next_observation, reward, done, truncated, info

def compute_gve(rewards, dones, truncations, next_values, gamma:float, lmbda:float):
    """
    Compute Generalized Value Estimator for REPPO
    """
    gves = []
    last_gve = torch.zeros_like(next_values[-1])
    trunc = truncations.clone()
    trunc[-1] = 1.0

    for t in reversed(range(rewards.shape[0])):
        lmbda_sum = lmbda * last_gve + (1 - lmbda) * next_values[t]
        delta = gamma * torch.where(trunc[t].bool(), next_values[t], (1.0 - dones[t]) * lmbda_sum)
        last_gve = rewards[t] + delta
        gves.insert(0, last_gve)
        
    return gves