"""
KUKA Reinforcement Learning Agent
Implements various RL algorithms
"""

import os
import sys
import time
import numpy as np
import gym
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

class NeuralNetwork(nn.Module):
    """
    Neural network for KUKA arm control
    Later I should fix this part more
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
    """PPO Agent for KUKA arm control"""
    
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
        self.gamma = 0.99  ## Discount factor
        self.eps_clip = 0.2
        self.K_epochs = 4
        self.value_coef = 0.5 ## Value function coefficient 
        self.entropy_coef = 0.01 ## Entropy coefficient 
        
        ###### Policy Gradient Hyperparameters ######
        # Networks
        self.policy_net = NeuralNetwork(observation_dim, action_dim * 2).to(self.device)  # mean + std
        self.value_net = NeuralNetwork(observation_dim, 1).to(self.device)
        
        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.lr)
        ##############################################
        
        # Memory
        ## For storing trajectories and actions 
        self.memory = []
        
        print(f"PPO Agent initialized - Obs: {observation_dim}, Action: {action_dim}")
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Get action from policy
        During getting action, we do not take gradient descent
        """
        
        ### Make observations as torch
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        ### Even if training mode, policy net does not take gradient descent for getting action. Do update in update method
        with torch.no_grad():
            ## Get Policy output based on observation
            policy_output = self.policy_net(obs_tensor) 
            ### Since policy net would return 2 * action dim, 
            mean = policy_output[:, :self.action_dim]
            log_std = policy_output[:, self.action_dim:]
            std = torch.exp(torch.clamp(log_std, -20, 2))
            
            if training: # When training
                
                ### Sample action from distribution based on mean and std
                dist = torch.distributions.Normal(mean, std)
            
                action = dist.sample()
                ### Take log probability of action and take sum over action's last dimension
                log_prob = dist.log_prob(action).sum(dim=-1)
            else: # Inference
                action = mean
                ### Since log prob is not used in inference, we set it to zero
                log_prob = torch.zeros(1)
            
            
            ### For numerical issue, we bound action to [-1,1]
            action = torch.tanh(action)  # Bound action to [-1, 1]
        
        action_info = {
            'log_prob': log_prob.item(),
            'mean': mean.squeeze().cpu().numpy(),
            'std': std.squeeze().cpu().numpy()
        }
        
        return action.squeeze().cpu().numpy(), action_info
    
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
        """Update policy and value networks"""
        if len(self.memory) == 0:
            return {}
        
        # Convert memory to tensors
        observations = torch.FloatTensor([m['obs'] for m in self.memory]).to(self.device)
        actions = torch.FloatTensor([m['action'] for m in self.memory]).to(self.device)
        rewards = torch.FloatTensor([m['reward'] for m in self.memory]).to(self.device)
        dones = torch.BoolTensor([m['done'] for m in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([m['log_prob'] for m in self.memory]).to(self.device)
        
        # Calculate returns and advantages
        returns = self._calculate_returns(rewards, dones)
        # Calculate values of observations via value network 
        values = self.value_net(observations).squeeze()
        ## Since advantages are used to normalize returns, we calculate advantages 
        ## Advantages are returns - values
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
            
            ### importance Sampling ratio
            ratio = torch.exp(new_log_probs - old_log_probs) ### ratio = new_log_probs / old_log_probs
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
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
        """Save agent"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict()
        }, filepath)
        print(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        print(f"Agent loaded from {filepath}")

class SACAgent:
    """SAC Agent for KUKA arm control"""
    
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

class KUKARLAgent:
    """Unified RL Agent interface for KUKA arm control"""
    
    def __init__(self, algorithm: str = "ppo", observation_dim: int = 30, action_dim: int = 7, 
                 device: str = "cuda", **kwargs):
        """
        Initialize KUKA RL Agent
        
        Args:
            algorithm: RL algorithm ('ppo', 'sac', 'thinking')
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            device: Computing device ('cuda')
        """
        if device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.algorithm = algorithm.lower()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'best_reward': -float('inf'),
            'recent_rewards': deque(maxlen=100)
        }
        
        # Initialize agent based on algorithm
        if self.algorithm == "ppo":
            self.agent = PPOAgent(observation_dim, action_dim, device)
        elif self.algorithm == "sac":
            self.agent = SACAgent(observation_dim, action_dim, device)
        elif self.algorithm == "thinking":
            self.agent
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        print(f"KUKA RL Agent initialized - Algorithm: {algorithm}, Device: {device}")
    
    def get_action(self, observation: np.ndarray, training: bool = True) -> Tuple[np.ndarray, Dict]:
        """Get action from agent"""
        return self.agent.get_action(observation, training)
    
    def store_transition(self, obs, action, reward, next_obs, done, action_info):
        """Store transition"""
        self.agent.store_transition(obs, action, reward, next_obs, done, action_info)
    
    def update(self) -> Dict:
        """Update agent"""
        return self.agent.update()
    
    def train_episode(self, env) -> Dict:
        """Train agent for one episode"""
        obs, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            # Get action
            action, action_info = self.get_action(obs, training=True)
            
            # Execute action
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            
            # Store transition
            self.store_transition(obs, action, reward, next_obs, done, action_info)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
        
        # Update agent
        update_info = self.update()
        
        # Update statistics
        self.training_stats['episodes'] += 1
        self.training_stats['total_steps'] += episode_steps
        self.training_stats['recent_rewards'].append(episode_reward)
        if episode_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = episode_reward
        
        episode_info = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'success': step_info.get('episode_success', False),
            'task_completed': step_info.get('task_completed', False),
            **update_info
        }
        
        return episode_info
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict:
        """Evaluate agent"""
        eval_rewards = []
        eval_successes = []
        
        for _ in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.get_action(obs, training=False)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            eval_rewards.append(episode_reward)
            eval_successes.append(info.get('episode_success', False))
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'success_rate': np.mean(eval_successes),
            'episodes': num_episodes
        }
    
    def save(self, filepath: str):
        """Save agent and training statistics"""
        # Save agent
        agent_path = filepath.replace('.pkl', '_agent.pth')
        self.agent.save(agent_path)
        
        # Save training statistics
        with open(filepath, 'wb') as f:
            pickle.dump(self.training_stats, f)
        
        print(f"Full agent saved: {filepath}")
    
    def load(self, filepath: str):
        """Load agent and training statistics"""
        # Load agent
        agent_path = filepath.replace('.pkl', '_agent.pth')
        if os.path.exists(agent_path):
            self.agent.load(agent_path)
        
        # Load training statistics
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.training_stats = pickle.load(f)
        
        print(f"Full agent loaded: {filepath}")

def train_agent(
    env: gym.Env,
    agent: Union[ PPOAgent, SACAgent],
    num_episodes: int,
    max_steps: int = 1000,
    render: bool = False
) -> List[float]:
    """Train an RL agent"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()

            if isinstance(agent, PPOAgent):
                action, log_prob = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done, log_prob)
            
            elif isinstance(agent, SACAgent):
                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                loss = agent.update()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    return episode_rewards

# def main():
#     """Test KUKA RL Agent"""
#     env = gym.make('Humanoid-v4')

#     env = env(task_type="reach")
    
#     try:
#         obs, _ = env.reset()
#         obs_dim = len(obs)
#         action_dim = 7
        
#         print(f"Environment: obs_dim={obs_dim}, action_dim={action_dim}")
        
#         # Test different algorithms
#         algorithms = ["ppo", "sac", "thinking"]
        
#         for algo in algorithms:
#             print(f"\n=== Testing {algo.upper()} Agent ===")
            
#             agent = KUKARLAgent(
#                 algorithm=algo,
#                 observation_dim=obs_dim,
#                 action_dim=action_dim
#             )
            
#             # Train for a few episodes
#             for episode in range(3):
#                 episode_info = agent.train_episode(env)
#                 print(f"Episode {episode + 1}: {episode_info}")
            
#             # Evaluate
#             eval_info = agent.evaluate(env, num_episodes=2)
#             print(f"Evaluation: {eval_info}")
            
#             # Save/load test
#             save_path = f"/tmp/kuka_{algo}_test.pkl"
#             agent.save(save_path)
#             agent.load(save_path)
            
#     except Exception as e:
#         print(f"Error: {e}")
#     finally:
#         env.close()

# if __name__ == "__main__":
#     main() 