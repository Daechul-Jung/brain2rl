import argparse
import gymnasium as gym
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'rl'))
from rl_agents_prac import PPOAgent, SACAgent, train_agent

def main():
    parser = argparse.ArgumentParser(description="Run PPO/SAC agent on OpenAI Gym continuous environment")
    parser.add_argument('--env', type=str, default='Humanoid-v4', help='Gym environment name (e.g., HalfCheetah-v4, Ant-v4, Pendulum-v1)')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac'], help='RL algorithm to use (ppo or sac)')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--render', action='store_true', help='Render environment')
    args = parser.parse_args()

    print(f"\n=== RL Demo on Gym ===")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")

    # Create environment
    env = gym.make(args.env)
    obs_space = env.observation_space
    act_space = env.action_space

    # Only support continuous action spaces
    if not isinstance(act_space, gym.spaces.Box):
        raise ValueError("This script only supports continuous action spaces (gym.spaces.Box)")

    state_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    # Create agent
    if args.algo == 'ppo':
        agent = PPOAgent(state_dim, action_dim)
    else:
        agent = SACAgent(state_dim, action_dim)

    # Train agent
    print(f"\n--- Training for {args.episodes} episodes ---")
    rewards = train_agent(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, render=args.render)
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    env.close()

if __name__ == "__main__":
    main() 