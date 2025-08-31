"""
RL Training Pipeline
===================

This module handles the reinforcement learning training of KUKA robot arm
using tokenized brain signal data for trajectory optimization.

Author: Daechul Jung
Version: 1.0.0
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import gymnasium as gym
from collections import deque
import random
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.rl.agents.ppo import PPOAgent
from models.rl.agents.sac import SACAgent
from models.rl.agents.reppo import *


class TokenGuidedRL:
    """
    Token-guided reinforcement learning agent that incorporates brain signal tokens
    into action selection and trajectory optimization
    """
    
    def __init__(self, state_dim: int, action_dim: int, token_dim: int, 
                 embedding_dim: int = 128, hidden_dim: int = 256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.token_dim = token_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Token processing network
        self.token_processor = nn.Sequential(
            nn.Linear(token_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        
        # State processing network
        self.state_processor = nn.Sequential(
            nn.Linear(state_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
        
        # Attention mechanism for token-state fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Policy and value networks
        self.policy_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q/K/V processing for attention-based action selection
        self.qkv_processor = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),  # Process Q, K, V together
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU()
        )
    
    def process_tokens_with_attention(self, tokens: torch.Tensor, 
                                    queries: torch.Tensor, 
                                    keys: torch.Tensor, 
                                    values: torch.Tensor) -> torch.Tensor:
        """
        Process tokens using Q/K/V attention mechanism
        
        Args:
            tokens: Token sequences (batch_size, seq_len, token_dim)
            queries: Query matrices (batch_size, seq_len, n_heads, head_dim)
            keys: Key matrices (batch_size, seq_len, n_heads, head_dim)
            values: Value matrices (batch_size, seq_len, n_heads, head_dim)
            
        Returns:
            Processed token features
        """
        batch_size, seq_len, _ = tokens.shape
        
        # Reshape Q/K/V to (batch_size, seq_len, -1) for processing
        q_flat = queries.view(batch_size, seq_len, -1)
        k_flat = keys.view(batch_size, seq_len, -1)
        v_flat = values.view(batch_size, seq_len, -1)
        
        # Concatenate Q/K/V
        qkv_combined = torch.cat([q_flat, k_flat, v_flat], dim=-1)
        
        # Process through QKV processor
        qkv_features = self.qkv_processor(qkv_combined)
        
        # Process tokens
        token_features = self.token_processor(tokens)
        
        # Combine token and QKV features
        combined_features = token_features + qkv_features
        
        return combined_features
    
    def forward(self, state: torch.Tensor, tokens: torch.Tensor,
                queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the token-guided RL network
        
        Args:
            state: Current state (batch_size, state_dim)
            tokens: Token sequences (batch_size, seq_len, token_dim)
            queries: Query matrices
            keys: Key matrices
            values: Value matrices
            
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        # Process state
        state_features = self.state_processor(state)  # (batch_size, embedding_dim)
        
        # Process tokens with attention
        token_features = self.process_tokens_with_attention(tokens, queries, keys, values)
        
        # Apply attention between state and token features
        # Expand state features to match token sequence length
        state_expanded = state_features.unsqueeze(1).expand(-1, token_features.size(1), -1)
        
        # Use attention to focus on relevant tokens
        attended_tokens, attention_weights = self.attention(
            state_expanded, token_features, token_features
        )
        
        # Pool attended tokens (mean pooling)
        pooled_tokens = attended_tokens.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Combine state and token features
        combined_features = torch.cat([state_features, pooled_tokens], dim=-1)
        
        # Generate action and value
        action_logits = self.policy_network(combined_features)
        value_estimate = self.value_network(combined_features)
        
        return action_logits, value_estimate


class TokenReplayBuffer:
    """
    Replay buffer that stores states, actions, rewards, and token information
    """
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float, 
            next_state: np.ndarray, done: bool, tokens: np.ndarray,
            queries: np.ndarray, keys: np.ndarray, values: np.ndarray):
        """Add experience to buffer"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'tokens': tokens,
            'queries': queries,
            'keys': keys,
            'values': values
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Sample batch from buffer"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # Convert to numpy arrays
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        tokens = np.array([exp['tokens'] for exp in batch])
        queries = np.array([exp['queries'] for exp in batch])
        keys = np.array([exp['keys'] for exp in batch])
        values = np.array([exp['values'] for exp in batch])
        
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'tokens': tokens,
            'queries': queries,
            'keys': keys,
            'values': values
        }
    
    def __len__(self):
        return len(self.buffer)


class RLTrainingPipeline:
    """
    Pipeline for training RL agents with token guidance
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize the RL training pipeline
        
        Args:
            model_config: Configuration for the RL model
        """
        self.model_config = model_config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize logger
        self.logger = logging.getLogger('Brain2RL.RLTraining')
        
        # Environment and agent
        self.env = None
        self.agent = None
        self.token_guided_agent = None
        
        # Training components
        self.replay_buffer = None
        self.optimizer = None
        
        # Training state
        self.training_history = []
        self.current_episode = 0
        self.current_step = 0
        
        self.logger.info("RL training pipeline initialized")
    
    def initialize_environment(self, env_config: Optional[Dict[str, Any]] = None):
        """Initialize the KUKA robot environment"""
        if env_config is None:
            env_config = {
                'task': 'reach',
                'render_mode': 'rgb_array',
                'control_frequency': 10,
                'max_episode_steps': 1000
            }
        
        self.env = ...
        self.logger.info(f"Environment initialized with config: {env_config}")
    
    def initialize_agent(self, token_data: Dict[str, np.ndarray]):
        """
        Initialize the token-guided RL agent
        
        Args:
            token_data: Dictionary containing tokens and Q/K/V matrices
        """
        if self.env is None:
            self.initialize_environment()
        
        # Get environment dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        # Get token dimensions
        token_dim = token_data['tokens'].shape[-1]
        
        # Initialize token-guided agent
        self.token_guided_agent = TokenGuidedRL(
            state_dim=state_dim,
            action_dim=action_dim,
            token_dim=token_dim,
            embedding_dim=self.model_config.get('embedding_dim', 128),
            hidden_dim=self.model_config.get('hidden_dim', 256)
        ).to(self.device)
        
        # Initialize traditional RL agent for comparison
        if self.model_config.get('algorithm', 'ppo') == 'ppo':
            self.agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.model_config.get('learning_rate', 0.0003),
                gamma=self.model_config.get('gamma', 0.99),
                clip_range=self.model_config.get('clip_range', 0.2)
            )
        else:
            self.agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                learning_rate=self.model_config.get('learning_rate', 0.0003),
                gamma=self.model_config.get('gamma', 0.99)
            )
        
        # Initialize replay buffer
        self.replay_buffer = TokenReplayBuffer(
            max_size=self.model_config.get('buffer_size', 100000)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.token_guided_agent.parameters(),
            lr=self.model_config.get('learning_rate', 0.0003)
        )
        
        self.logger.info(f"Agent initialized - State dim: {state_dim}, Action dim: {action_dim}, Token dim: {token_dim}")
    
    def get_token_sequence(self, token_data: Dict[str, np.ndarray], 
                          episode_step: int, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get token sequence for current episode step
        
        Args:
            token_data: Dictionary containing tokens and Q/K/V matrices
            episode_step: Current step in episode
            sequence_length: Length of token sequence to use
            
        Returns:
            Tuple of (tokens, queries, keys, values)
        """
        # Simple approach: cycle through available tokens
        total_tokens = len(token_data['tokens'])
        
        start_idx = (episode_step * sequence_length) % total_tokens
        end_idx = min(start_idx + sequence_length, total_tokens)
        
        # Handle wrap-around
        if end_idx - start_idx < sequence_length:
            remaining = sequence_length - (end_idx - start_idx)
            tokens = np.concatenate([
                token_data['tokens'][start_idx:end_idx],
                token_data['tokens'][:remaining]
            ])
            queries = np.concatenate([
                token_data['queries'][start_idx:end_idx],
                token_data['queries'][:remaining]
            ])
            keys = np.concatenate([
                token_data['keys'][start_idx:end_idx],
                token_data['keys'][:remaining]
            ])
            values = np.concatenate([
                token_data['values'][start_idx:end_idx],
                token_data['values'][:remaining]
            ])
        else:
            tokens = token_data['tokens'][start_idx:end_idx]
            queries = token_data['queries'][start_idx:end_idx]
            keys = token_data['keys'][start_idx:end_idx]
            values = token_data['values'][start_idx:end_idx]
        
        return tokens, queries, keys, values
    
    def train_with_tokens(self, token_data: Dict[str, np.ndarray], 
                         num_episodes: int = 1000) -> Dict[str, Any]:
        """
        Train the RL agent using token guidance
        
        Args:
            token_data: Dictionary containing tokens and Q/K/V matrices
            num_episodes: Number of training episodes
            
        Returns:
            Dictionary with training results
        """
        if self.token_guided_agent is None:
            self.initialize_agent(token_data)
        
        # Training metrics
        episode_rewards = []
        episode_lengths = []
        token_guided_rewards = []
        baseline_rewards = []
        
        sequence_length = self.model_config.get('token_sequence_length', 10)
        
        for episode in tqdm(range(num_episodes), desc="Training Episodes"):
            # Reset environment
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Train both token-guided and baseline agents
            baseline_state = state.copy()
            baseline_reward = 0
            
            while not done:
                # Get token sequence for current step
                tokens, queries, keys, values = self.get_token_sequence(
                    token_data, episode_length, sequence_length
                )
                
                # Token-guided action selection
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    tokens_tensor = torch.FloatTensor(tokens).unsqueeze(0).to(self.device)
                    queries_tensor = torch.FloatTensor(queries).unsqueeze(0).to(self.device)
                    keys_tensor = torch.FloatTensor(keys).unsqueeze(0).to(self.device)
                    values_tensor = torch.FloatTensor(values).unsqueeze(0).to(self.device)
                    
                    action_logits, value_estimate = self.token_guided_agent.forward(
                        state_tensor, tokens_tensor, queries_tensor, keys_tensor, values_tensor
                    )
                    
                    # Add noise for exploration
                    noise = torch.randn_like(action_logits) * 0.1
                    action = (action_logits + noise).squeeze(0).cpu().numpy()
                    action = np.clip(action, -1, 1)
                
                # Baseline action selection
                baseline_action = self.agent.select_action(baseline_state)
                
                # Execute actions
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Store experience in replay buffer
                self.replay_buffer.add(
                    state, action, reward, next_state, done,
                    tokens, queries, keys, values
                )
                
                # Update baseline agent
                baseline_next_state, baseline_reward_step, _, _, _ = self.env.step(baseline_action)
                self.agent.store_transition(baseline_state, baseline_action, baseline_reward_step, 
                                          baseline_next_state, done)
                
                # Update states and rewards
                state = next_state
                baseline_state = baseline_next_state
                episode_reward += reward
                baseline_reward += baseline_reward_step
                episode_length += 1
                self.current_step += 1
                
                # Train token-guided agent
                if len(self.replay_buffer) > self.model_config.get('batch_size', 64):
                    self.train_token_guided_agent()
                
                # Train baseline agent
                if self.current_step % self.model_config.get('update_frequency', 100) == 0:
                    self.agent.update()
            
            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            token_guided_rewards.append(episode_reward)
            baseline_rewards.append(baseline_reward)
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_baseline = np.mean(baseline_rewards[-100:])
                self.logger.info(f"Episode {episode}: Token-guided reward: {avg_reward:.2f}, "
                               f"Baseline reward: {avg_baseline:.2f}")
        
        # Store training history
        self.training_history = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'token_guided_rewards': token_guided_rewards,
            'baseline_rewards': baseline_rewards
        }
        
        # Training results
        results = {
            'trained_model': self.token_guided_agent,
            'baseline_model': self.agent,
            'training_history': self.training_history,
            'final_performance': {
                'token_guided_avg': np.mean(token_guided_rewards[-100:]),
                'baseline_avg': np.mean(baseline_rewards[-100:]),
                'improvement': np.mean(token_guided_rewards[-100:]) - np.mean(baseline_rewards[-100:])
            }
        }
        
        self.logger.info(f"Training completed. Token-guided agent achieved {results['final_performance']['improvement']:.2f} improvement over baseline")
        
        return results
    
    def train_token_guided_agent(self):
        """Train the token-guided agent using replay buffer"""
        if len(self.replay_buffer) < self.model_config.get('batch_size', 64):
            return
        
        # Sample batch
        batch = self.replay_buffer.sample(self.model_config.get('batch_size', 64))
        
        # Convert to tensors
        states = torch.FloatTensor(batch['states']).to(self.device)
        actions = torch.FloatTensor(batch['actions']).to(self.device)
        rewards = torch.FloatTensor(batch['rewards']).to(self.device)
        next_states = torch.FloatTensor(batch['next_states']).to(self.device)
        dones = torch.BoolTensor(batch['dones']).to(self.device)
        tokens = torch.FloatTensor(batch['tokens']).to(self.device)
        queries = torch.FloatTensor(batch['queries']).to(self.device)
        keys = torch.FloatTensor(batch['keys']).to(self.device)
        values = torch.FloatTensor(batch['values']).to(self.device)
        
        # Forward pass
        action_logits, value_estimates = self.token_guided_agent.forward(
            states, tokens, queries, keys, values
        )
        
        # Compute targets
        with torch.no_grad():
            next_action_logits, next_value_estimates = self.token_guided_agent.forward(
                next_states, tokens, queries, keys, values
            )
            targets = rewards + self.model_config.get('gamma', 0.99) * next_value_estimates.squeeze() * (~dones)
        
        # Compute losses
        value_loss = F.mse_loss(value_estimates.squeeze(), targets)
        
        # Policy loss (simplified actor-critic)
        advantages = targets - value_estimates.squeeze()
        action_log_probs = F.log_softmax(action_logits, dim=-1)
        action_probs = torch.sum(action_log_probs * actions, dim=-1)
        policy_loss = -torch.mean(action_probs * advantages.detach())
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
    
    def evaluate_agent(self, token_data: Dict[str, np.ndarray], 
                      num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate the trained agent
        
        Args:
            token_data: Dictionary containing tokens and Q/K/V matrices
            num_episodes: Number of evaluation episodes
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.token_guided_agent is None:
            raise ValueError("Agent must be trained before evaluation")
        
        self.token_guided_agent.eval()
        
        rewards = []
        episode_lengths = []
        success_rate = 0
        
        for episode in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                # Get token sequence
                tokens, queries, keys, values = self.get_token_sequence(
                    token_data, episode_length, 10
                )
                
                # Select action
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    tokens_tensor = torch.FloatTensor(tokens).unsqueeze(0).to(self.device)
                    queries_tensor = torch.FloatTensor(queries).unsqueeze(0).to(self.device)
                    keys_tensor = torch.FloatTensor(keys).unsqueeze(0).to(self.device)
                    values_tensor = torch.FloatTensor(values).unsqueeze(0).to(self.device)
                    
                    action_logits, _ = self.token_guided_agent.forward(
                        state_tensor, tokens_tensor, queries_tensor, keys_tensor, values_tensor
                    )
                    
                    action = action_logits.squeeze(0).cpu().numpy()
                    action = np.clip(action, -1, 1)
                
                # Execute action
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                
                # Check for success (task-specific)
                if info.get('success', False):
                    success_rate += 1
                    break
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate metrics
        metrics = {
            'average_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'average_length': np.mean(episode_lengths),
            'success_rate': success_rate / num_episodes,
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }
        
        self.logger.info(f"Evaluation completed: Avg reward: {metrics['average_reward']:.2f}, "
                        f"Success rate: {metrics['success_rate']:.2f}")
        
        return metrics
    
    def save_models(self, save_path: str):
        """Save trained models"""
        save_data = {
            'token_guided_agent': self.token_guided_agent.state_dict(),
            'baseline_agent': self.agent.get_state_dict() if hasattr(self.agent, 'get_state_dict') else None,
            'model_config': self.model_config,
            'training_history': self.training_history
        }
        
        torch.save(save_data, save_path)
        self.logger.info(f"Models saved to {save_path}")
    
    def load_models(self, load_path: str):
        """Load trained models"""
        save_data = torch.load(load_path, map_location=self.device)
        
        self.model_config = save_data['model_config']
        self.training_history = save_data['training_history']
        
        # Initialize agents with saved config
        # This would need proper initialization with correct dimensions
        self.logger.info(f"Models loaded from {load_path}")
    
    def plot_training_results(self, save_path: Optional[str] = None):
        """Plot training results"""
        if not self.training_history:
            self.logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.training_history['token_guided_rewards'], label='Token-guided')
        axes[0, 0].plot(self.training_history['baseline_rewards'], label='Baseline')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Moving average rewards
        window = 100
        if len(self.training_history['token_guided_rewards']) > window:
            token_ma = np.convolve(self.training_history['token_guided_rewards'], 
                                 np.ones(window)/window, mode='valid')
            baseline_ma = np.convolve(self.training_history['baseline_rewards'], 
                                    np.ones(window)/window, mode='valid')
            axes[0, 1].plot(token_ma, label='Token-guided (MA)')
            axes[0, 1].plot(baseline_ma, label='Baseline (MA)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Moving Average Reward')
            axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Episode lengths
        axes[1, 0].plot(self.training_history['episode_lengths'])
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Episode Length')
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].grid(True)
        
        # Performance comparison
        improvement = np.array(self.training_history['token_guided_rewards']) - np.array(self.training_history['baseline_rewards'])
        axes[1, 1].plot(improvement)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Improvement over Baseline')
        axes[1, 1].set_title('Token-guided vs Baseline Performance')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training plots saved to {save_path}")
        
        plt.show()


def main():
    """Main function for standalone RL training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RL Training Pipeline')
    parser.add_argument('--token-data', type=str, required=True, help='Path to tokenized data')
    parser.add_argument('--mode', choices=['train', 'evaluate'], default='train')
    parser.add_argument('--model-path', type=str, help='Path to save/load model')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo', help='RL algorithm')
    parser.add_argument('--learning-rate', type=float, default=0.0003, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create model config
    model_config = {
        'algorithm': args.algorithm,
        'learning_rate': args.learning_rate,
        'batch_size': 64,
        'gamma': 0.99,
        'clip_range': 0.2,
        'embedding_dim': 128,
        'hidden_dim': 256,
        'token_sequence_length': 10,
        'update_frequency': 100,
        'buffer_size': 100000
    }
    
    # Initialize pipeline
    pipeline = RLTrainingPipeline(model_config)
    
    # Load token data
    token_data = np.load(args.token_data)
    if not isinstance(token_data, np.lib.npyio.NpzFile):
        raise ValueError("Token data must be in .npz format")
    
    token_dict = {
        'tokens': token_data['tokens'],
        'queries': token_data['queries'],
        'keys': token_data['keys'],
        'values': token_data['values']
    }
    
    if args.mode == 'train':
        # Train agent
        results = pipeline.train_with_tokens(token_dict, num_episodes=args.episodes)
        
        # Save model
        if args.model_path:
            pipeline.save_models(args.model_path)
        
        # Plot results
        pipeline.plot_training_results('training_results.png')
        
        print(f"Training completed. Final improvement: {results['final_performance']['improvement']:.2f}")
        
    elif args.mode == 'evaluate':
        # Load model
        if args.model_path:
            pipeline.load_models(args.model_path)
        
        # Evaluate agent
        metrics = pipeline.evaluate_agent(token_dict, num_episodes=50)
        
        print(f"Evaluation completed:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")


if __name__ == '__main__':
    main() 