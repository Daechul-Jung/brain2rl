#!/usr/bin/env python3
"""
Example: Integration with Reinforcement Learning

This script demonstrates how to use the generated tokens from the classification
and tokenization pipeline in reinforcement learning algorithms.
"""

import numpy as np
import torch
import os
from pathlib import Path

def load_generated_tokens(token_file: str = 'output/generated_tokens.npz'):
    """
    Load the generated tokens from the pipeline
    
    Args:
        token_file: Path to the generated tokens file
        
    Returns:
        Dictionary containing tokens and metadata
    """
    if not os.path.exists(token_file):
        raise FileNotFoundError(f"Token file not found: {token_file}")
    
    # Load tokens
    token_data = np.load(token_file)
    
    print(f"Loaded tokens with shape: {token_data['tokens'].shape}")
    print(f"Number of sequences: {token_data['tokens'].shape[0]}")
    print(f"Sequence length: {token_data['tokens'].shape[1]}")
    print(f"Token dimension: {token_data['tokens'].shape[2]}")
    print(f"Labels: {np.unique(token_data['labels'])}")
    
    return token_data

def create_rl_state_from_tokens(tokens: np.ndarray, sequence_idx: int = 0):
    """
    Convert tokens to a state representation for RL
    
    Args:
        tokens: Token array from the pipeline
        sequence_idx: Index of the sequence to use
        
    Returns:
        State vector for RL algorithm
    """
    # Get the specific sequence
    sequence = tokens[sequence_idx]  # Shape: (seq_len, token_dim)
    
    # Flatten the sequence to create a state vector
    state = sequence.flatten()
    
    print(f"Created RL state from sequence {sequence_idx}")
    print(f"State shape: {state.shape}")
    
    return state

def create_trajectory_dataset(tokens: np.ndarray, labels: np.ndarray):
    """
    Create a trajectory dataset for RL training
    
    Args:
        tokens: Token array from the pipeline
        labels: Action labels from the pipeline
        
    Returns:
        Dictionary containing trajectories and actions
    """
    trajectories = []
    actions = []
    
    for i in range(len(tokens)):
        # Each sequence is a trajectory
        trajectory = tokens[i]  # Shape: (seq_len, token_dim)
        action = labels[i]      # The action label for this trajectory
        
        trajectories.append(trajectory)
        actions.append(action)
    
    # Convert to numpy arrays
    trajectories = np.array(trajectories)
    actions = np.array(actions)
    
    print(f"Created trajectory dataset:")
    print(f"  - Number of trajectories: {len(trajectories)}")
    print(f"  - Trajectory shape: {trajectories[0].shape}")
    print(f"  - Action distribution: {np.bincount(actions)}")
    
    return {
        'trajectories': trajectories,
        'actions': actions,
        'n_trajectories': len(trajectories),
        'trajectory_length': trajectories[0].shape[0],
        'token_dim': trajectories[0].shape[1]
    }

def create_rl_environment_state(tokens: np.ndarray, window_size: int = 3):
    """
    Create sliding window states for RL environment
    
    Args:
        tokens: Token array from the pipeline
        window_size: Size of the sliding window
        
    Returns:
        Array of windowed states
    """
    all_states = []
    
    for sequence in tokens:
        # Create sliding windows
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            # Flatten the window
            state = window.flatten()
            all_states.append(state)
    
    states = np.array(all_states)
    
    print(f"Created RL environment states:")
    print(f"  - Number of states: {len(states)}")
    print(f"  - State dimension: {states.shape[1]}")
    
    return states

def integrate_with_ppo_algorithm(trajectory_data: dict):
    """
    Example integration with PPO (Proximal Policy Optimization)
    
    Args:
        trajectory_data: Dictionary containing trajectories and actions
    """
    print("\n=== PPO Integration Example ===")
    
    # Extract data
    trajectories = trajectory_data['trajectories']
    actions = trajectory_data['actions']
    
    # Create state-action pairs
    states = []
    for traj in trajectories:
        # Use the mean of the trajectory as state representation
        state = np.mean(traj, axis=0)
        states.append(state)
    
    states = np.array(states)
    
    print(f"PPO State-Action pairs:")
    print(f"  - States shape: {states.shape}")
    print(f"  - Actions shape: {actions.shape}")
    
    # Example: Create a simple policy network
    state_dim = states.shape[1]
    action_dim = len(np.unique(actions))
    
    print(f"  - State dimension: {state_dim}")
    print(f"  - Action dimension: {action_dim}")
    
    # Here you would typically:
    # 1. Initialize PPO agent
    # 2. Set state and action spaces
    # 3. Use trajectories for pre-training or demonstration learning
    # 4. Train the agent
    
    return {
        'states': states,
        'actions': actions,
        'state_dim': state_dim,
        'action_dim': action_dim
    }

def integrate_with_sac_algorithm(trajectory_data: dict):
    """
    Example integration with SAC (Soft Actor-Critic)
    
    Args:
        trajectory_data: Dictionary containing trajectories and actions
    """
    print("\n=== SAC Integration Example ===")
    
    # Extract data
    trajectories = trajectory_data['trajectories']
    actions = trajectory_data['actions']
    
    # Create continuous state representation
    states = []
    for traj in trajectories:
        # Use the full trajectory as state
        state = traj.flatten()
        states.append(state)
    
    states = np.array(states)
    
    print(f"SAC State-Action pairs:")
    print(f"  - States shape: {states.shape}")
    print(f"  - Actions shape: {actions.shape}")
    
    # For SAC, we might want continuous actions
    # Convert discrete actions to continuous embeddings
    unique_actions = np.unique(actions)
    action_embeddings = []
    for action in actions:
        # Create one-hot encoding
        one_hot = np.zeros(len(unique_actions))
        # Find the index of the action in unique_actions
        action_idx = np.where(unique_actions == action)[0][0]
        one_hot[action_idx] = 1.0
        action_embeddings.append(one_hot)
    
    action_embeddings = np.array(action_embeddings)
    
    print(f"  - Action embeddings shape: {action_embeddings.shape}")
    
    return {
        'states': states,
        'action_embeddings': action_embeddings,
        'state_dim': states.shape[1],
        'action_dim': action_embeddings.shape[1]
    }

def main():
    """Main function demonstrating RL integration"""
    print("Reinforcement Learning Integration Example")
    print("=" * 50)
    
    try:
        # Load generated tokens
        print("1. Loading generated tokens...")
        token_data = load_generated_tokens()
        
        # Extract tokens and labels
        tokens = token_data['tokens']
        labels = token_data['labels'].flatten()  # Flatten from (n, 1) to (n,)
        
        # Create RL state
        print("\n2. Creating RL state representation...")
        state = create_rl_state_from_tokens(tokens, sequence_idx=0)
        
        # Create trajectory dataset
        print("\n3. Creating trajectory dataset...")
        trajectory_data = create_trajectory_dataset(tokens, labels)
        
        # Create RL environment states
        print("\n4. Creating RL environment states...")
        env_states = create_rl_environment_state(tokens, window_size=2)
        
        # Example integrations
        print("\n5. Example RL algorithm integrations...")
        
        # PPO integration
        ppo_data = integrate_with_ppo_algorithm(trajectory_data)
        
        # SAC integration
        sac_data = integrate_with_sac_algorithm(trajectory_data)
        
        print("\n" + "=" * 50)
        print("INTEGRATION COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print(f"\nSummary of available data for RL:")
        print(f"  - Raw tokens: {tokens.shape}")
        print(f"  - Trajectories: {trajectory_data['n_trajectories']}")
        print(f"  - PPO states: {ppo_data['state_dim']} dimensions")
        print(f"  - SAC states: {sac_data['state_dim']} dimensions")
        
        print(f"\nNext steps:")
        print(f"  1. Use these states in your RL algorithm")
        print(f"  2. Implement reward functions based on action labels")
        print(f"  3. Train your RL agent with the token-based states")
        print(f"  4. Use the trained agent for real-time control")
        
        return {
            'token_data': token_data,
            'trajectory_data': trajectory_data,
            'ppo_data': ppo_data,
            'sac_data': sac_data,
            'env_states': env_states
        }
        
    except Exception as e:
        print(f"Error during RL integration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()
