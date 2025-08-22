import torch

def wrap_batch_dim(observation, critic_observation, action, ):
    return 




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