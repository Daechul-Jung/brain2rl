import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.data_utils import load_eeg_data, load_fmri_data, create_dataloader
from models.tokenization.brain_tokenizer import BrainTokenizer

class BrainIntegratedPolicy(ActorCriticPolicy):
    """
    Policy network that integrates brain signals with standard RL inputs
    """
    def __init__(self, observation_space, action_space, lr_schedule, 
                 brain_token_dim=512, brain_token_seq_len=64, **kwargs):
        super(BrainIntegratedPolicy, self).__init__(
            observation_space, action_space, lr_schedule, **kwargs
        )
        
        # Store brain token dimensions
        self.brain_token_dim = brain_token_dim
        self.brain_token_seq_len = brain_token_seq_len
        
        # Additional network for processing brain tokens
        self.brain_token_processor = nn.Sequential(
            nn.Linear(brain_token_dim * brain_token_seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Modify the policy network to include brain token information
        # Replace the final layer of the policy network
        self.action_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.policy_latent_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.shape[0])
        )
        
        # Modify the value network to include brain token information
        self.value_net = nn.Sequential(
            nn.Linear(self.mlp_extractor.value_latent_dim + 128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, obs, brain_tokens=None, deterministic=False):
        """
        Forward pass of the network
        Args:
            obs: Environment observation
            brain_tokens: Tokenized brain signals (optional)
            deterministic: Whether to sample from the action distribution
        Returns:
            actions: Actions to take
            values: Value function estimates
            log_probs: Log probabilities of the actions
        """
        # Process the observation through the feature extractor
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        
        # Process brain tokens if provided
        if brain_tokens is not None:
            # Flatten brain tokens
            batch_size = brain_tokens.shape[0]
            flattened_tokens = brain_tokens.reshape(batch_size, -1)
            token_features = self.brain_token_processor(flattened_tokens)
            
            # Combine with policy and value features
            latent_pi = torch.cat([latent_pi, token_features], dim=1)
            latent_vf = torch.cat([latent_vf, token_features], dim=1)
        
        # Get the action distribution
        action_logits = self.action_net(latent_pi)
        
        # Retrieve the value function
        values = self.value_net(latent_vf)
        
        # Sample from the action distribution
        distribution = self.get_distribution(action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)
        
        return actions, values, log_probs

class BrainGuidedAgent:
    """
    Reinforcement learning agent guided by brain signals
    Uses TRPO (Trust Region Policy Optimization) for smooth policy updates
    """
    def __init__(self, env_name, tokenizer_path, tokenizer_type='eeg', 
                 brain_token_dim=512, brain_token_seq_len=64):
        self.env_name = env_name
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self.brain_token_dim = brain_token_dim
        self.brain_token_seq_len = brain_token_seq_len
        
        # Create the environment
        self.env = gym.make(env_name)
        
        # Load the tokenizer
        if tokenizer_type == 'eeg':
            X, _ = load_eeg_data(os.path.join('data', 'eeg'))
            self.tokenizer = BrainTokenizer(input_channels=X.shape[1], input_length=X.shape[2])
        else:  # fmri
            X, _ = load_fmri_data(os.path.join('data', 'fmri'))
            self.tokenizer = BrainTokenizer(input_channels=X.shape[1], input_length=X.shape[2])
        
        # Load the tokenizer weights if path is provided
        if os.path.exists(tokenizer_path):
            self.tokenizer.load_state_dict(torch.load(tokenizer_path))
            print(f"Loaded tokenizer from {tokenizer_path}")
        else:
            print(f"Tokenizer path {tokenizer_path} not found. Using untrained tokenizer.")
        
        self.tokenizer.eval()  # Set to evaluation mode
        
        # Create the policy with brain token integration
        # For now, we'll use PPO from stable-baselines3 as it's similar to TRPO but more stable
        self.model = PPO(
            policy=BrainIntegratedPolicy,
            env=self.env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs={
                'brain_token_dim': brain_token_dim,
                'brain_token_seq_len': brain_token_seq_len
            },
            verbose=1
        )
    
    def train(self, total_timesteps=100000, log_dir='runs/brain_rl'):
        """Train the agent"""
        # Create callback for saving checkpoints
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join('models', 'rl'),
            name_prefix='brain_rl_model'
        )
        
        # Train the agent
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback
        )
    
    def evaluate(self, num_episodes=10):
        """Evaluate the agent"""
        obs = self.env.reset()
        
        total_rewards = 0
        for episode in range(num_episodes):
            episode_reward = 0
            done = False
            
            while not done:
                # Generate random brain signals for demonstration
                # In a real BCI system, these would come from a device
                fake_brain_signal = np.random.randn(1, 32, 512).astype(np.float32)
                brain_tensor = torch.tensor(fake_brain_signal)
                
                # Tokenize the brain signal
                with torch.no_grad():
                    brain_tokens = self.tokenizer(brain_tensor)
                
                # Get action from policy
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Step the environment
                obs, reward, done, _ = self.env.step(action)
                
                episode_reward += reward
                
                if done:
                    obs = self.env.reset()
                    break
            
            total_rewards += episode_reward
            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f}")
        
        avg_reward = total_rewards / num_episodes
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        
        return avg_reward

def train_rl(args):
    """Train the RL agent with brain signals"""
    # Determine the environment based on the settings
    env_name = "Pendulum-v0"  # A simple continuous control environment
    
    # Determine the tokenizer path
    tokenizer_path = os.path.join('models', 'tokenization', f'best_{args.data}_tokenizer.pth')
    
    # Create and train the agent
    agent = BrainGuidedAgent(
        env_name=env_name,
        tokenizer_path=tokenizer_path,
        tokenizer_type=args.data
    )
    
    # Train the agent
    agent.train(total_timesteps=args.epochs * 1000)  # Convert epochs to total timesteps
    
    print(f"Finished training the brain-guided RL agent with {args.data} signals")

def evaluate_rl(args):
    """Evaluate the RL agent with brain signals"""
    # Determine the environment based on the settings
    env_name = "Pendulum-v0"  # A simple continuous control environment
    
    # Determine the tokenizer path
    tokenizer_path = os.path.join('models', 'tokenization', f'best_{args.data}_tokenizer.pth')
    
    # Create the agent
    agent = BrainGuidedAgent(
        env_name=env_name,
        tokenizer_path=tokenizer_path,
        tokenizer_type=args.data
    )
    
    # Evaluate the agent
    avg_reward = agent.evaluate(num_episodes=10)
    
    print(f"Finished evaluating the brain-guided RL agent with {args.data} signals")
    print(f"Average reward: {avg_reward:.2f}") 