"""
Relative Entropy Proximal Pairwise Policy Optimization (My own Idea)
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import torch.optim as optim
import copy
from torch.amp import GradScaler
import gymnasium as gym
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.reppo_network import *
from models.rl.utils.any_utils import (compute_gve, _ensure_batch, _env_shape, _split_step_return, _to_tensor, wrap_batch_dim)



class REP3OAgent:
    """
    Relative Entropy Proximal Pairwis Optimization Agent which clip between 
    This agent does not use GradScaler and Normalizer. Maybe later can use normalerizer for better performance
    """
    def __init__(self, observation_dim, action_dim, num_atoms = 151,  ### num_atoms (51 ~ 151)
                 vmin= -2000, vmax=4000, device='cuda',
                 lr = 3e-4, gamma = 0.99, kl_start = 0.05, entropy_start = 0.05,
                 lmbda = 0.95, obs_normalizer= None , critic_obs_normalizer = None):

        self.vmin = vmin
        self.vmax = vmax
        self.gamma = gamma

        self.kl_start = kl_start
        self.entropy_start = entropy_start
        self.lmbda = lmbda

        self.entropy_target = 0.3 * action_dim   # was 0.5 * action_dim (too high)
        self.kl_bound = 0.3                      # was 0.1 (too tight)
        self.kl_target = 0.01  #### previous 0.25 and try 0.5, 0.75 

        self.num_atoms = num_atoms
        self.observation_dim = observation_dim
        self.action_dim = action_dim 
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        
        self.aux_loss_mult = 1.0
        self.max_grad_norm = 0.6
        self.actor_kl_clip_mode = "clipped"
        # Actor(Policy) includes entropy and kl-regularization which are subject to optimization
        self.actor = Actor(observation_dim=observation_dim,
                            action_dim=action_dim, 
                            hidden_dim=512,
                           kl_start=kl_start, entropy_start=entropy_start, 
                           device=self.device)
        
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        
        self.critic = Critic(observation_dim=observation_dim, 
                             action_dim=action_dim,
                             hidden_dim=512, 
                             num_atoms=num_atoms, 
                             vmin=vmin, vmax=vmax, 
                             device=self.device)
        
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr = lr)#, betas=(0.9,0.999), eps=1e-5)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr = lr)#, betas=(0.9, 0.999), eps=1e-5)
        self.observation_normalizer = obs_normalizer
        self.critic_observation_normalizer = critic_obs_normalizer


    def _actor_forward(self, observation):
        pi, mean, temp, lagrangian = self.actor(observation)
        return pi, mean, temp, lagrangian

    def _old_actor_forward(self, observation):
        old_pi, old_mean, old_temp, old_lagrangian = self.old_actor(observation)
        return old_pi, old_mean, old_temp, old_lagrangian
    
    def _critic_forward(self, observation, action):
        value, logit, next_pred_features, features = self.critic(observation, action) 
        return 
    
    def update_actor(self, batch):
        """
        Update actor with the method used in PPO but still use KL and Lagrangian parameters.
        """
        eps = 1e-6
        self.actor.train()
        observation = batch['observation']
        critic_observation = batch['critic_observation']

        pi, mean, temp, lagran = self._actor_forward(observation)

        action = pi.sample().clamp(-1 + eps, 1 - eps)
        log_prob = pi.log_prob(action).sum(-1)

        entropy = -log_prob

        q_scalar, _, _, _ = self._critic_forward(critic_observation, action)

        with torch.no_grad():
            old_pi, old_mean, old_temp, old_lagran = self._old_actor_forward(observation)
            old_action = old_pi.sample().clamp(-1 + eps, 1 - eps)
            old_log_prob = old_pi.log_prob(old_action).sum(-1)

        return 
    
    def update_critic(self, batch):
        self.critic.train()

        critic_observation = batch['critic_observation']
        

        return 
    
    def collect(self, env, observation, critic_observation, num_steps):
        N, _, asymmetric = _env_shape(env)
        eps = 1e-6
        trajectory = []
        info_list = []
        ## initial reset

        if observation is None:
            reset_return = env.reset()
            observation = reset_return[0] if isinstance(reset_return, tuple) else reset_return

        if critic_observation is None:
            critic_observation = observation

        observation = _to_tensor(observation, self.device)
        critic_observation = _to_tensor(critic_observation, self.device)

        for _ in range(num_steps):
            norm_obs = self.observation_normalizer(observation)
            norm_critic_obs = self.critic_observation_normalizer(critic_observation)
            
            with torch.no_grad():
                old_pi, old_mean, old_temp, old_lagrangian = self._old_actor_forward(norm_obs)
                old_action = old_pi.sample().clamp(-1 + eps, 1 - eps)
                old_log_prob = old_pi.log_prob(old_action).sum(-1)
                old_action = old_action.detach().cpu().numpy().astype(np.float32)

                if isinstance(env.action_space, gym.spaces.Box):
                    low, high = env.action_space.low, env.action_space.high
                    if not (np.allclose(low, -1.0) and np.allclose(high, 1.0)):
                        old_action = low + 0.5 * (old_action + 1.0) * (high - low)

            returns = env.step(old_action)
            next_observation, rewards, dones, truncated, infos = _split_step_return(returns)

            next_critic_observation = next_observation

            next_norm_obs = self.observation_normalizer(next_observation)
            next_critic_norm_obs = self.critic_observation_normalizer(next_critic_observation)

            old_log_prob_torch = _to_tensor(old_log_prob, self.device)

            with torch.no_grad():
                old_next_pi, old_next_mean, old_next_temp, old_next_lagr = self._old_actor_forward(next_norm_obs)
                old_next_action = old_next_pi.sample().clamp(-1 + eps, 1 - eps)
                old_next_log_prob = old_next_pi.log_prob(old_next_action).sum(-1)

                next_value, next_qlogit, _, next_features = self._critic_forward(next_critic_norm_obs, old_next_action)

                shaped_rewards = rewards - self.gamma * old_next_log_prob * old_next_temp
                observation_batch, critic_observation_batch, action_batch, log_prob_batch, rewards_batch, raw_reward_batch, next_features_batch, next_value_batch, dones_batch, truncated_batch = wrap_batch_dim(
                norm_obs, norm_critic_obs, old_action, old_log_prob_torch, shaped_rewards, rewards, next_features, next_value, dones, truncated, self.device
            )

            td = TensorDict(
                {
                    "observation":        observation_batch,             # [N, obs_dim]
                    "critic_observation": critic_observation_batch,            # [N, obs_dim]
                    "actions":            action_batch,             # [N, act_dim]
                    "log_prob":           log_prob_batch,            # [N] (or use logp_b.unsqueeze(-1) if you prefer [N,1])
                    "rewards":            rewards_batch,             # [N,1]
                    "raw_rewards":        raw_reward_batch,         # [N,1]
                    "next_embedding":     next_features_batch,       # [N, F]
                    "next_values":        next_value_batch,        # [N,1]
                    "dones":              dones_batch,            # [N,1]
                    "truncations":        truncated_batch,           # [N,1]
                },
                batch_size=(N,),
            )

            trajectory.append(td)
            info_list.append(infos)
            
            observation = _to_tensor(next_observation, self.device)
            critic_observation = _to_tensor(next_critic_observation, self.device)
            
        transition  = torch.stack(trajectory, dim = 0) ## Shape: (T, N)
            
        return transition, norm_obs, norm_critic_obs, info_list
    
    def get_action(self, observation, training=False):
        # to tensor on the SAME device/dtype as the model
        obs_t = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)  # [1, obs_dim]

        self.actor.eval()
        with torch.no_grad():
            pi, mean, temp, lag = self._actor_forward(obs_t)   # actor is on self.device
            action_t = pi.rsample() if training else pi.sample()  # [1, act_dim] in (-1,1) due to Tanh
        self.actor.train()

        # remove batch dim, move to CPU, ensure float32 numpy
        return action_t.squeeze(0).cpu().numpy().astype(np.float32), ...

