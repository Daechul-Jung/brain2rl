import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.optim as optim
import copy
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.fcnn import *

"""
This algorithm is Relative Entropy Pairwise Policy Optimization (RePPO)
This is on-policy and actor-critic alogrithm and train robust surrogate value function and effectively use pairwise policy gradient.
Build Maximum entropy framework and combines with a principled KL regularization, 
Categorical Q-Learning, Appropriately normalized neural network architecture and auxilary task
Show how a joint entropy and policy deviation tuning objective can address the twin problems of sufficient exploration and controlled policy update 
According to the paper, it does not utilize the replay buffer
"""

def _env_shape(env):
    num_envs = getattr(env, "num_envs", 1)
    max_steps = getattr(env, "max_episode_steps", 5000)
    asymmetric = getattr(env, "asymmetric_obs", False )
    return num_envs, max_steps, asymmetric

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

@dataclass(slots=True)
class TrainState:
    device: torch.device
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actor: Actor
    old_actor: Actor
    critic: Critic
    # normalizer: EmpiricalNormalization
    # critic_normalizer: EmpiricalNormalization
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    # scaler: GradScaler



class RePPOAgent:
    def __init__(self, observation_dim, action_dim, num_atoms = 101, 
                 vmin=-250, vmax=250, device='cuda',
                 lr = 3e-4, gamma = 0.99, kl_start = 0.1, entropy_start = 0.1,
                 lmbda = 0.95):
        super().__init__()
        
        self.critic = Critic(observation_dim, action_dim, num_atoms, 
                             vmin=vmin, vmax=vmax, device=device)
        self.vmin = vmin
        self.vmax = vmax
        self.num_atoms = num_atoms
        self.observation_dim = observation_dim
        self.action_dim = action_dim 
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.aux_loss_mult = 1.0
        self.max_grad_norm = 1.0
        # Actor(Policy) includes entropy and kl-regularization which are subject to optimization
        self.actor = Actor(observation_dim=observation_dim,
                            action_dim=action_dim, 
                           kl_start=kl_start, entropy_start=entropy_start, 
                           device=device)
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(observation_dim=observation_dim, 
                             action_dim=action_dim, 
                             num_atoms=num_atoms, 
                             vmin=vmin, vmax=vmax, 
                             device=device)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr = lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr = lr)
        
    def _actor_forward(self, observation):
        """
        Forward pass for the actor network
        
        Returns:
            pi: Action probabilities with transformed distributions
            mean: Mean of the action distribution
            log_temp: Log of entropy temperature
            log_lagrange: Log of KL regularization parameter
        """
        pi, mean, log_temp, log_lagrange = self.actor(observation)
        return pi, mean, log_temp, log_lagrange
    
    def _old_actor_forward(self, observation):
        """
        Based on the observation with old actor network, getting old pi distribution
        """

        old_pi, _, _, _ = self.old_actor(observation)

        return old_pi
    
    
    def _critic_forward(self, observation, action):
        """
        Critic returns value, log, next_pred, and featuers. Critic network has three modules which are feature module, prediction module, critic module. 
        Firstly, concatenate observation and action and put those in feature FCNN network. Then, using features resulted from feature network, get next pred
        Embedded next pred is current state's prediction of the next embedding prediction and via feature network, we can store next state's target embedding 
        """
        value, logits, next_pred, features = self.critic(observation, action)

        return value, logits, next_pred, features 

    def update_actor(self, batch: TensorDict):
        """
        Update actor network 
        """
        self.critic.train()

        observation_critic = batch['critic_observation'] # Shape: (Batch, observation_critic)
        actions = batch['actions'] # shape: (Batch, Action)
        targets = batch['gve'].squeeze(-1) # Shape: (Batch, )
        truncated = batch['truncation'].squeeze(-1) # Shape: (Batch, )
        target_next_feature = batch['next_embedding'] # Shape: (Batch, )

        trunc_mask = (1.0 - truncated).to(observation_critic.dtype)

        with torch.no_grad():
            q_target_dist = hl_gauss(targets, self.vmin, self.vmax, self.num_atoms)

        q_scalar, q_logit, next_pred, _features = self._critic_forward(observation_critic, actions)
        log_probs = F.log_softmax(q_logit, dim = -1)
        ce = -(q_target_dist * log_probs).sum(-1)

        emb_loss = F.mse_loss(next_pred, target_next_feature, reduction=None)
        loss = (trunc_mask * ( ce +  self.aux_loss_mult * emb_loss)).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)    
        return 
    
    def update_critic(self):
        return 
    
    @torch.no_grad()
    def collect(self, env, observation, critic_observation, num_steps =10000):
        """
        On-policy rollout and return (transition tensordict, final_obs, final_critic_obs, infos)
        """
        transition = []
        info_list = []
        
        for _ in range(num_steps):
            ...
        
        return 
    
    def evaluate(self,):
        return 
    
    def load(self):
        return 
    
    def save(self, ):
        return 