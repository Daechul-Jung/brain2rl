import numpy as np
from visualization import TrainingVisualizer
import matplotlib.pyplot as plt

def generate_test_data():
    # Generate some realistic-looking training data
    episodes = 100
    rewards = []
    losses = {
        'actor_loss': [],
        'critic_loss': []
    }
    actions = []
    
    # Generate rewards with a learning trend
    base_reward = -10
    for i in range(episodes):
        # Add some noise and a learning trend
        reward = base_reward + i * 0.2 + np.random.normal(0, 2)
        rewards.append(reward)
        
        # Generate losses that decrease over time
        actor_loss = 2.0 * np.exp(-i/20) + np.random.normal(0, 0.1)
        critic_loss = 1.5 * np.exp(-i/25) + np.random.normal(0, 0.1)
        losses['actor_loss'].append(actor_loss)
        losses['critic_loss'].append(critic_loss)
        
        # Generate actions with some bias
        action = np.random.choice(5, p=[0.3, 0.2, 0.2, 0.2, 0.1])
        actions.append(action)
    
    return rewards, losses, actions

def main():
    # Create visualizer
    visualizer = TrainingVisualizer(save_dir='training_plots')
    
    # Generate test data
    rewards, losses, actions = generate_test_data()
    
    # Create plots
    print("Generating training rewards plot...")
    visualizer.plot_rewards(rewards, title='Training Progress')
    
    print("Generating loss curves...")
    visualizer.plot_losses(losses, title='Training Losses')
    
    print("Generating action distribution...")
    visualizer.plot_action_distribution(actions, num_actions=5, title='Action Distribution')
    
    # Generate some state values for visualization
    states = np.random.randn(100, 10)  # 100 states with 10 dimensions
    values = np.random.randn(100)      # Random values for demonstration
    
    print("Generating state value visualization...")
    visualizer.plot_state_values(states, values, title='State Value Distribution')
    
    print("\nAll plots have been saved to the 'training_plots' directory.")
    print("You can find the following files:")
    print("- training_rewards.png")
    print("- training_losses.png")
    print("- action_distribution.png")
    print("- state_values.png")

if __name__ == '__main__':
    main() 