import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Union
import torch
from pathlib import Path

class TrainingVisualizer:
    def __init__(self, save_dir: str = 'training_plots'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        sns.set_style('whitegrid')
    
    def plot_rewards(self, rewards: List[float], title: str = 'Training Rewards', window_size: int = 10):
        """Plot training rewards with moving average"""
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.plot(rewards, alpha=0.3, label='Raw Rewards')
        
        # Plot moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(rewards)), moving_avg, label=f'{window_size}-Episode Moving Average')
        
        plt.title(title)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.savefig(self.save_dir / 'training_rewards.png')
        plt.close()
    
    def plot_losses(self, losses: Dict[str, List[float]], title: str = 'Training Losses'):
        """Plot different types of losses"""
        plt.figure(figsize=(12, 6))
        
        for loss_name, loss_values in losses.items():
            plt.plot(loss_values, label=loss_name)
        
        plt.title(title)
        plt.xlabel('Update Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir / 'training_losses.png')
        plt.close()
    
    def plot_action_distribution(self, actions: List[int], num_actions: int, title: str = 'Action Distribution'):
        """Plot the distribution of actions taken during training"""
        plt.figure(figsize=(10, 6))
        
        action_counts = np.bincount(actions, minlength=num_actions)
        plt.bar(range(num_actions), action_counts)
        
        plt.title(title)
        plt.xlabel('Action')
        plt.ylabel('Count')
        plt.xticks(range(num_actions))
        plt.savefig(self.save_dir / 'action_distribution.png')
        plt.close()
    
    def plot_state_values(self, states: np.ndarray, values: np.ndarray, title: str = 'State Values'):
        """Plot state values for visualization"""
        plt.figure(figsize=(12, 6))
        
        # If states are high-dimensional, use PCA to reduce to 2D
        if states.shape[1] > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            states_2d = pca.fit_transform(states)
        else:
            states_2d = states
        
        plt.scatter(states_2d[:, 0], states_2d[:, 1], c=values, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title(title)
        plt.xlabel('State Dimension 1')
        plt.ylabel('State Dimension 2')
        plt.savefig(self.save_dir / 'state_values.png')
        plt.close()
    
    def plot_learning_curves(self, metrics: Dict[str, List[float]], title: str = 'Learning Curves'):
        """Plot multiple metrics over time"""
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.title(title)
        plt.xlabel('Training Step')
        plt.ylabel('Value')
        plt.legend()
        plt.savefig(self.save_dir / 'learning_curves.png')
        plt.close()

# Example usage
if __name__ == '__main__':
    # Create visualizer
    visualizer = TrainingVisualizer()
    
    # Example data
    rewards = [np.random.normal(0, 1) for _ in range(100)]
    losses = {
        'actor_loss': [np.random.normal(0, 1) for _ in range(100)],
        'critic_loss': [np.random.normal(0, 1) for _ in range(100)]
    }
    actions = np.random.randint(0, 5, 1000)
    
    # Generate plots
    visualizer.plot_rewards(rewards)
    visualizer.plot_losses(losses)
    visualizer.plot_action_distribution(actions, num_actions=5) 