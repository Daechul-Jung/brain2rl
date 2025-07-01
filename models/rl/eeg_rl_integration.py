import os
import sys
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import custom modules
from models.pipelines.eeg_to_rl_pipeline import EEGTokenizationPipeline
from models.rl.brain_rl import BrainGuidedAgent

class EEGTokenBuffer:
    """
    Buffer to store and retrieve tokenized EEG data for RL
    """
    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.tokens = []
        self.actions = []
        self.current_idx = 0
    
    def add(self, tokens, actions):
        """Add tokens and actions to the buffer"""
        # Reshape if necessary to match expected dimensions
        if len(tokens.shape) == 3:  # (batch, seq_len, token_dim)
            self.tokens.extend(tokens)
            self.actions.extend(actions)
        else:
            self.tokens.append(tokens)
            self.actions.append(actions)
        
        # Trim if exceeding max size
        if len(self.tokens) > self.max_size:
            self.tokens = self.tokens[-self.max_size:]
            self.actions = self.actions[-self.max_size:]
    
    def get_random_batch(self, batch_size=32):
        """Get a random batch of tokens and actions"""
        if len(self.tokens) == 0:
            return None, None
        
        indices = np.random.randint(0, len(self.tokens), size=min(batch_size, len(self.tokens)))
        return np.array([self.tokens[i] for i in indices]), np.array([self.actions[i] for i in indices])
    
    def get_next(self):
        """Get the next token and action in sequence (for deterministic playback)"""
        if len(self.tokens) == 0:
            return None, None
        
        token = self.tokens[self.current_idx]
        action = self.actions[self.current_idx]
        
        self.current_idx = (self.current_idx + 1) % len(self.tokens)
        
        return token, action
    
    def __len__(self):
        return len(self.tokens)


class EEGTokenCallback(BaseCallback):
    """
    Callback for using EEG tokens in reinforcement learning
    """
    def __init__(self, token_buffer, token_usage_probability=0.5, verbose=0):
        super(EEGTokenCallback, self).__init__(verbose)
        self.token_buffer = token_buffer
        self.token_usage_probability = token_usage_probability
        self.episode_count = 0
        self.token_usage_history = []
    
    def _on_step(self):
        # Randomly decide whether to use a token from the buffer
        if np.random.random() < self.token_usage_probability and len(self.token_buffer) > 0:
            # Get a token from the buffer
            token, action = self.token_buffer.get_next()
            
            # Convert to tensor if not already
            if not isinstance(token, torch.Tensor):
                token = torch.tensor(token, dtype=torch.float32).unsqueeze(0)
            
            # Pass the token to the model's policy
            # Note: This requires that the model has been modified to accept brain tokens
            self.model.policy.brain_tokens = token
            
            # Track token usage
            self.token_usage_history.append(1)
        else:
            # No token used this step
            self.token_usage_history.append(0)
        
        return True
    
    def _on_rollout_end(self):
        self.episode_count += 1
        
        # Log token usage statistics
        if self.verbose > 0 and self.episode_count % 10 == 0:
            recent_usage = np.mean(self.token_usage_history[-100:]) if self.token_usage_history else 0
            print(f"Episode {self.episode_count}: Token usage rate: {recent_usage:.2f}")
        
        return True


class EEGRLIntegration:
    """
    Class to integrate EEG tokenization with RL
    """
    def __init__(self, data_dir, tokenizer_path, classifier_path=None, 
                 env_name="Pendulum-v0", window_size=500, overlap=0.5):
        self.data_dir = data_dir
        self.tokenizer_path = tokenizer_path
        self.classifier_path = classifier_path
        self.env_name = env_name
        self.window_size = window_size
        self.overlap = overlap
        
        # Initialize the environment
        self.env = gym.make(env_name)
        
        # Initialize the token buffer
        self.token_buffer = EEGTokenBuffer()
        
        # Initialize the EEG pipeline
        self.pipeline = EEGTokenizationPipeline(
            data_dir=data_dir,
            tokenizer_path=tokenizer_path,
            classifier_path=classifier_path,
            window_size=window_size,
            overlap=overlap
        )
        
        # Initialize the RL agent
        self.agent = None
    
    def prepare_token_buffer(self, subject_ids=None, series_ids=None, max_batches=None):
        """Prepare the token buffer with EEG data"""
        # Load and preprocess data
        self.pipeline.load_data(subject_ids, series_ids)
        
        # Run the pipeline to get tokens and actions
        tokens, actions, _ = self.pipeline.run_pipeline(batch_size=32, max_batches=max_batches)
        
        # Add to buffer
        self.token_buffer.add(tokens, actions)
        
        print(f"Added {len(tokens)} token sequences to buffer")
        return tokens, actions
    
    def initialize_agent(self, load_path=None):
        """Initialize the RL agent"""
        # Create the agent
        self.agent = BrainGuidedAgent(
            env_name=self.env_name,
            tokenizer_path=self.tokenizer_path,
            tokenizer_type='eeg'
        )
        
        # Load a pre-trained model if provided
        if load_path and os.path.exists(load_path):
            self.agent.model = PPO.load(load_path, env=self.env)
            print(f"Loaded pre-trained model from {load_path}")
        
        return self.agent
    
    def train_with_eeg_guidance(self, total_timesteps=100000, token_usage_prob=0.5, 
                               save_path='brain_rl_model'):
        """Train the RL agent with EEG token guidance"""
        if self.agent is None:
            self.initialize_agent()
        
        # Create the callback for token integration
        callback = EEGTokenCallback(
            token_buffer=self.token_buffer,
            token_usage_probability=token_usage_prob,
            verbose=1
        )
        
        # Train the agent
        print(f"Training agent for {total_timesteps} timesteps with token usage probability {token_usage_prob}")
        self.agent.model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
        
        # Save the trained model
        save_dir = os.path.join('models', 'rl')
        os.makedirs(save_dir, exist_ok=True)
        full_save_path = os.path.join(save_dir, save_path)
        self.agent.model.save(full_save_path)
        print(f"Saved trained model to {full_save_path}")
        
        # Plot token usage
        plt.figure(figsize=(10, 5))
        plt.plot(callback.token_usage_history)
        plt.title('EEG Token Usage During Training')
        plt.xlabel('Steps')
        plt.ylabel('Token Used (1=Yes, 0=No)')
        plt.ylim([-0.1, 1.1])
        plt.savefig('token_usage_history.png')
        plt.close()
        
        return callback.token_usage_history
    
    def evaluate_with_eeg_tokens(self, num_episodes=10, deterministic=True):
        """Evaluate the agent with EEG tokens"""
        if self.agent is None:
            print("Agent not initialized. Please initialize first.")
            return
        
        total_rewards = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Get a token from the buffer
                token, action_probs = self.token_buffer.get_next()
                
                if token is None:
                    # If buffer is empty, use random token
                    print("Token buffer empty, using random token")
                    token = np.random.randn(1, self.pipeline.tokenizer.output_length, 512).astype(np.float32)
                
                # Convert to tensor
                if not isinstance(token, torch.Tensor):
                    token = torch.tensor(token, dtype=torch.float32).unsqueeze(0)
                
                # Get action from the model
                # The model will use the brain token internally
                action, _ = self.agent.model.predict(obs, deterministic=deterministic)
                
                # Step the environment
                obs, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            total_rewards.append(episode_reward)
            print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}")
        
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")
        
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(total_rewards)
        plt.title('Episode Rewards with EEG Token Guidance')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('eeg_guided_rewards.png')
        plt.close()
        
        return avg_reward, total_rewards


def main():
    # Example usage
    data_dir = os.path.join('data', 'train')
    tokenizer_path = os.path.join('models', 'tokenization', 'best_eeg_tokenizer.pth')
    classifier_path = 'best_eeg_action_classifier.pth'
    
    # Initialize the integration
    integration = EEGRLIntegration(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        classifier_path=classifier_path
    )
    
    # Prepare token buffer with a subset of the data
    tokens, actions = integration.prepare_token_buffer(
        subject_ids=[1, 2],  # Use first 2 subjects
        max_batches=10  # Limit the number of batches for demonstration
    )
    
    # Initialize the agent
    agent = integration.initialize_agent()
    
    # Train the agent with EEG guidance
    # Using fewer timesteps for demonstration
    token_usage = integration.train_with_eeg_guidance(
        total_timesteps=10000,  # Reduced for demonstration
        token_usage_prob=0.7,
        save_path='eeg_guided_rl_model'
    )
    
    # Evaluate the agent
    avg_reward, rewards = integration.evaluate_with_eeg_tokens(num_episodes=5)
    
    print(f"Evaluation complete! Average reward: {avg_reward:.2f}")
    print("Token usage and rewards plotted to files.")


if __name__ == "__main__":
    main() 