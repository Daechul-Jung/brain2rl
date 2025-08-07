import numpy as np
import torch

class PPOBuffer:
    """
    Buffer for storing trajectories experienced by PPO agent.
    Stores obs, actions, rewards, dones, log_probs, values.
    Computes advantages and returns for PPO update.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size
        self.device = device

    def store(self, obs, act, rew, val, logp, done):
        assert self.ptr < self.max_size     # Buffer has to have room!
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.done_buf[self.ptr] = done
        self.ptr += 1

    def finish_path(self, last_val=0):
        """Call this at the end of a trajectory/episode!"""
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # GAE-Lambda advantage calculation
        adv = np.zeros_like(rews[:-1])
        lastgaelam = 0
        for t in reversed(range(len(rews)-1)):
            delta = rews[t] + self.gamma * vals[t+1] - vals[t]
            adv[t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam

        self.adv_buf[path_slice] = adv
        self.ret_buf[path_slice] = adv + self.val_buf[path_slice]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size    # Buffer must be full before you get
        # Advantage normalization
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf) + 1e-8
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=torch.as_tensor(self.obs_buf, device=self.device),
                    act=torch.as_tensor(self.act_buf, device=self.device),
                    ret=torch.as_tensor(self.ret_buf, device=self.device),
                    adv=torch.as_tensor(self.adv_buf, device=self.device),
                    logp=torch.as_tensor(self.logp_buf, device=self.device))
        return data

    def clear(self):
        self.ptr = 0
        self.path_start_idx = 0