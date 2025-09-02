import os
import sys
import time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple
from collections import deque, namedtuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.buffer import PPOBuffer
from models.rl.utils.NeuralNetwork import NeuralNetwork

class SACAgent:    
    def __init__(self, observation_dim: int, action_dim: int, device: str = "cuda"):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if torch.cuda.is_available() and device == "cuda":
            device = "cuda"
        else:
            device  = 'cpu'
        self.device = torch.device(device)
        
        # Hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.2  # Entropy regularization
        self.buffer_size = 100000
        self.batch_size = 256
        
        # Networks
        self.actor = NeuralNetwork(observation_dim, action_dim * 2).to(self.device)
        self.critic1 = NeuralNetwork(observation_dim + action_dim, 1).to(self.device)
        self.critic2 = NeuralNetwork(observation_dim + action_dim, 1).to(self.device)
        self.target_critic1 = NeuralNetwork(observation_dim + action_dim, 1).to(self.device)
        self.target_critic2 = NeuralNetwork(observation_dim + action_dim, 1).to(self.device)
        
        # Copy weights to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr)
        
        # Replay buffer
        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        print(f"SAC Agent initialized - Obs: {observation_dim}, Action: {action_dim}")
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get action from policy"""
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            actor_output = self.actor(obs_tensor)
            mean = actor_output[:, :self.action_dim]
            log_std = actor_output[:, self.action_dim:]
            log_std = torch.clamp(log_std, -20, 2)
            
            if training:
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()  # Reparameterization trick
                action = torch.tanh(action)
            else:
                action = torch.tanh(mean)
        
        action_info = {
            'mean': mean.squeeze().cpu().numpy(),
            'std': torch.exp(log_std).squeeze().cpu().numpy()
        }
        
        return action.squeeze().cpu().numpy(), action_info
    
    def store_transition(self, obs, action, reward, next_obs, done, action_info):
        """Store transition in replay buffer"""
        self.replay_buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })
    
    def update(self):
        """Update SAC networks"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        batch = np.random.choice(self.replay_buffer, self.batch_size, replace=False)
        
        obs = torch.FloatTensor([t['obs'] for t in batch]).to(self.device)
        actions = torch.FloatTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_obs = torch.FloatTensor([t['next_obs'] for t in batch]).to(self.device)
        dones = torch.BoolTensor([t['done'] for t in batch]).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_action_output = self.actor(next_obs)
            next_mean = next_action_output[:, :self.action_dim]
            next_log_std = torch.clamp(next_action_output[:, self.action_dim:], -20, 2)
            next_std = torch.exp(next_log_std)
            
            next_dist = torch.distributions.Normal(next_mean, next_std)
            next_actions = next_dist.rsample()
            next_actions = torch.tanh(next_actions)
            next_log_probs = next_dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            
            next_q1 = self.target_critic1(torch.cat([next_obs, next_actions], dim=1))
            next_q2 = self.target_critic2(torch.cat([next_obs, next_actions], dim=1))
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            
            target_q = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1).float()) * next_q
        
        current_q1 = self.critic1(torch.cat([obs, actions], dim=1))
        current_q2 = self.critic2(torch.cat([obs, actions], dim=1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        actor_output = self.actor(obs)
        actor_mean = actor_output[:, :self.action_dim]
        actor_log_std = torch.clamp(actor_output[:, self.action_dim:], -20, 2)
        actor_std = torch.exp(actor_log_std)
        
        actor_dist = torch.distributions.Normal(actor_mean, actor_std)
        actor_actions = actor_dist.rsample()
        actor_actions = torch.tanh(actor_actions)
        actor_log_probs = actor_dist.log_prob(actor_actions).sum(dim=-1, keepdim=True)
        
        q1_new = self.critic1(torch.cat([obs, actor_actions], dim=1))
        q2_new = self.critic2(torch.cat([obs, actor_actions], dim=1))
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * actor_log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic1_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'q_value': current_q1.mean().item()
        }
    
    def _soft_update(self, target, source):
        """Soft update target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save agent"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict()
        }, filepath)
        print(f"SAC Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])
        print(f"SAC Agent loaded from {filepath}")