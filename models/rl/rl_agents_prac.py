"""
Reinforcement Learning Agent
Implements various RL algorithms
"""

import os
import sys
import time
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Any, Optional, Tuple, Union
import pickle
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import RL libraries
# try:
#     import stable_baselines3 as sb3
#     from stable_baselines3 import PPO, SAC, DDPG
#     from stable_baselines3.common.vec_env import DummyVecEnv
#     from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
#     SB3_AVAILABLE = True
# except ImportError:
#     print("WARNING: stable-baselines3 not available. Using custom implementation.")
#     SB3_AVAILABLE = False
from rl.buffer import PPOBuffer
class NeuralNetwork(nn.Module):
    """
    Neural network for OpenArm or humanoid v-4
    Later I should fix this part more for utilizing information and complex tasks
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        """_summary_

        Args:
            input_dim (int): input dimension of the network which would be observation space dimension
            output_dim (int): output dimension of the network which would be action space dimension
            hidden_dims (List[int], optional): Hidden dimension of neural network. Defaults to [256, 256, 128].
        """
        super(NeuralNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

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

class SACAgent:
    """SAC Agent for OpenArm or Humanoid control"""
    
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


def train_agent(env, agent, num_episodes, max_steps=1000, render=False):
    episode_rewards = []
    for ep in range(num_episodes):
        # For every episode, reset the environment 
        state, info = env.reset()
        episode_reward = 0.0
        for t in range(max_steps):
            if render:
                env.render()

            action, action_info = agent.get_action(state, training=True)

            # map [-1,1] -> env range
            if hasattr(env.action_space, "low") and isinstance(env.action_space, gym.spaces.Box):
                low, high = env.action_space.low, env.action_space.high
                action = low + 0.5 * (action + 1.0) * (high - low)

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done, action_info)

            state = next_state
            episode_reward += reward
            if done:
                break
        # After every episode, update the agent 
        agent.update()
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{num_episodes}, Reward: {episode_reward:.2f}")
    return episode_rewards

def train_agent_with_buffer(env, agent:PPOAgent, num_episodes:int, max_steps = 1000, render=False):
    episode_rewards = []
    episode_done = 0

    state, info = env.reset()
    episode_ret = 0.0
    episode_len = 0

    while episode_done < num_episodes:
        for t in range(agent.step_per_rollout):
            if render:
                env.render()

            action, log_prob, value, action_before_tahn = agent.get_action_with_buffer(state, training=True)

            if hasattr(env.action_space, "low") and isinstance(env.action_space, gym.spaces.Box):
                low, high = env.action_space.low, env.action_space.high
                env_action = low + 0.5 * (action + 1.0) * (high - low)

            else:
                env_action = action

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)

            agent.buffer.store(
                obs = state, 
                act = action_before_tahn,
                rew = reward,
                val = value, 
                logp = log_prob,
                done = done
            )

            episode_ret += reward
            episode_len += 1

            state = next_state

            if done or episode_len >= max_steps:
                agent.buffer.finish_path(last_val=0)
                episode_rewards.append(episode_ret)
                episode_done += 1
                state, info = env.reset()
                episode_ret = 0
                episode_len = 0

                if episode_done > num_episodes:
                    break 
        if episode_len > 0:
            with torch.no_grad():
                s = np.asarray(state, dtype=np.float32)
                if s.ndim == 1:
                    s = s[None, :]
                v_boot = agent.value_net(torch.from_numpy(s).to(agent.device)).squeeze(-1).item()
            agent.buffer.finish_path(last_val=v_boot)

            # Update PPO on the collected rollout
            info = agent.updateWithBuffer()
            if info:
                print(f"Update: policy_loss={info['policy_loss']:.4f}, value_loss={info['value_loss']:.4f}")
            # Clear buffer for next iteration
            agent.buffer.clear()


        return episode_rewards
