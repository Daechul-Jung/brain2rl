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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.buffer import PPOBuffer
from models.rl.policy.NeuralNetwork import NeuralNetwork

class PPOAgent:
    """PPO Agent for OpenArm or humanoid control"""
    
    def __init__(self, observation_dim: int, action_dim: int, device: str = "cuda", buffer = PPOBuffer ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        if torch.cuda.is_available() and device == "cuda":
            device = "cuda"
        else:
            device  = 'cpu'
        self.device = torch.device(device)
        
        # Hyperparameters
        self.lr = 3e-4
        self.gamma = 0.99  ## Discount factor
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.step_per_rollout = 4096
        self.value_coef = 0.5 ## Value function coefficient 
        self.entropy_coef = 0.01 ## Entropy coefficient 
        self.minibatch_size = 256
        ###### Policy Gradient Hyperparameters ######
        # Networks
        self.policy_net = NeuralNetwork(observation_dim, action_dim * 2).to(self.device)  # mean + std
        self.value_net = NeuralNetwork(observation_dim, 1).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        ##############################################
        
        self.step_count = 0
        
        # Memory
        ## For storing trajectories and actions 
        self.memory = []
        self.buffer = PPOBuffer(obs_dim=observation_dim, act_dim=action_dim, size=self.step_per_rollout, gamma=self.gamma, lam=0.95, device=self.device.type)
        
        print(f"PPO Agent initialized - Obs: {observation_dim}, Action: {action_dim}")
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get action from policy without buffer
        During getting action, we do not take gradient descent
        """
        if isinstance(observation, (list, tuple)):
            obs_np = np.asarray(observation, dtype=np.float32)
        elif isinstance(observation, np.ndarray):
            obs_np = observation.astype(np.float32, copy=False)
        else:
            obs_np = np.array(observation, dtype=np.float32)

        if obs_np.ndim == 1:
            obs_np = obs_np[None, :]  # add batch dim
        ### Make observations as torch
        obs_tensor = torch.from_numpy(obs_np).to(self.device)
        
        
        ### Even if training mode, policy net does not take gradient descent for getting action. Do update in update method
        with torch.no_grad():
            ## Get Policy output based on observation
            policy_output = self.policy_net(obs_tensor) 
            ### Since policy net would return 2 * action dim, 
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(torch.clamp(log_std, -20, 2))
            dist = torch.distributions.Normal(mean, std)
            if training: # When training
                ### Sample action from distribution based on mean and std            
                action = dist.sample()
                ### Take log probability of action and take sum over action's last dimension
                log_prob = dist.log_prob(action).sum(dim=-1)
            else: # Inference
                action = mean
                ### Since log prob is not used in inference, we set it to zero
                log_prob = torch.zeros(action.shape[0])
            
            
            ### For numerical issue, we bound action to [-1,1]
            action = torch.tanh(action)  # Bound action to [-1, 1]


        action_info = {
            'log_prob': log_prob.item(),
            'mean': mean.squeeze().cpu().numpy(),
            'std': std.squeeze().cpu().numpy()
        }
        
        return action.squeeze().cpu().numpy(), action_info
    
    def get_action_with_buffer(self, observation, training):
        """
        Get action from policy with buffer
        Args:
            observation (_type_): _description_
            training (_type_): _description_

        Returns:
            _type_: _description_
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        obs_tensor = torch.from_numpy(obs).to(self.device)

        ### Even if training mode, policy net does not take gradient descent for getting action. Do update in update method
        with torch.no_grad():
            ## Get Policy output based on observation
            policy_output = self.policy_net(obs_tensor) 
            ### Since policy net would return 2 * action dim, 
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(torch.clamp(log_std, -20, 2))
            dist = torch.distributions.Normal(mean, std)
            if training: # When training
                ### Sample action from distribution based on mean and std            
                action = dist.sample()
                ### Take log probability of action and take sum over action's last dimension
                log_prob = dist.log_prob(action).sum(dim=-1)
            else: # Inference
                action = mean
                ### Since log prob is not used in inference, we set it to zero
                log_prob = torch.zeros(action.shape[0])
            
            
            ### For numerical issue, we bound action to [-1,1]
            env_action = torch.tanh(action)  # Bound action to [-1, 1]
            value = self.value_net(obs_tensor).squeeze(-1)
        
        return (env_action.squeeze(0).cpu().numpy(), float(log_prob.squeeze(0).item()), float(value.squeeze(0).item()), action.squeeze(0).cpu().numpy())
    

    def store_transition(self, obs, action, reward, next_obs, done, action_info):
        """Store transition in memory"""
        self.memory.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'log_prob': action_info['log_prob']
        })
        
    
    def update(self):
        """
        Update policy and value networks after collecting trajectories in one episode
        """
        
        # Convert memory to tensors
        ## Trajectories of agent stored in memory
        observations = torch.FloatTensor([m['obs'] for m in self.memory]).to(self.device) 
        actions = torch.FloatTensor([m['action'] for m in self.memory]).to(self.device)
        rewards = torch.FloatTensor([m['reward'] for m in self.memory]).to(self.device)
        dones = torch.BoolTensor([m['done'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        
        with torch.no_grad():
            ### While calculating returns and advantages, we do not take gradient descent
            # Calculate returns and advantages
            returns = self._calculate_returns(rewards, dones)
            # Calculate values of observations via value network 
            values = self.value_net(observations).squeeze()
            ## Since advantages are used to normalize returns, we calculate advantages 
            ## Advantages are returns - values which represent how much better the action is compared to the value of the state
            advantages = returns - values
            ## Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update for K epochs
        total_policy_loss = 0
        total_value_loss = 0
        
        for _ in range(self.K_epochs):
            # Getting Policy via network
            policy_output = self.policy_net(observations)
            
            ## Divide policy output into mean and log std
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(torch.clamp(log_std, -20, 2))
            
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            ## should explain more mathematically
            entropy = dist.entropy().sum(dim=-1)
            
            ### Importance Sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs) ### ratio = new_log_probs / old_log_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            ### Policy update based on loss which is scalar
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
            
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()
            
            total_policy_loss += policy_loss.item()
            
            # Value update
            new_values = self.value_net(observations).squeeze()
            value_loss = F.mse_loss(new_values, returns)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
            self.value_optimizer.step()
            
            total_value_loss += value_loss.item()
        
        # Clear memory
        self.memory.clear()
        
        return {
            'policy_loss': total_policy_loss / self.K_epochs,
            'value_loss': total_value_loss / self.K_epochs,
            'mean_return': returns.mean().item()
        }
    def updateWithBuffer(self):
        data = self.buffer.get()
        
        obs = data['obs']
        act = data['act']
        ret = data['ret']
        adv = data['adv']
        log_prob = data['logp']

        n = obs.size(0)
        idx = torch.arange(n, device=self.device)

        policy_loss_acc = 0.0
        value_loss_acc = 0.0

        for _ in range(self.K_epochs):
            perm = idx[torch.randperm(n)]
            for start in range(0, n ,self.minibatch_size):
                mini_batch = perm[start: start + self.minibatch_size]
                policy_out = self.policy_net(obs[mini_batch])
                mean = policy_out[:, :self.action_dim]
                std = torch.exp(torch.clamp(policy_out[:, self.action_dim:], -20, 2))
                dist = torch.distributions.Normal(mean, std)

                new_log_prob = dist.log_prob(act[mini_batch]).sum(dim = -1)
                entropy = dist.entropy().sum(dim = -1)

                ratio = torch.exp(new_log_prob - log_prob[mini_batch])
                surr1 = ratio * adv[mini_batch]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv[mini_batch]
                policy_loss = -(torch.min(surr1, surr2).mean()) - self.entropy_coef * entropy.mean()
                
                # policy loss update
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.policy_optimizer.step()

                # value loss update
                value_pred = self.value_net(obs[mini_batch]).squeeze(-1)
                value_loss = F.mse_loss(value_pred, ret[mini_batch])

                self.value_optimizer.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 0.5)
                self.value_optimizer.step()

                policy_loss_acc += policy_loss.item() * mini_batch.numel()
                value_loss_acc  += value_loss.item() * mini_batch.numel()

        return {
            'policy_loss': policy_loss_acc / n / self.K_epochs,
            'value_loss':  value_loss_acc  / n / self.K_epochs,
        }

    def _calculate_returns(self, rewards, dones):
        """
        Calculate discounted returns with discount factor gamma
        Calculate returns in reverse order
        """
        
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        return returns
    
    def save(self, filepath: str):
        """
        Save agent's parameters and optimizers
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load agent's best parameters and optimizers with checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        print(f"Agent loaded from {filepath}")