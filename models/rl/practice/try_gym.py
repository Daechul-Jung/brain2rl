import argparse
import gymnasium as gym
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'rl'))
from models.rl.practice.agents.ppo import PPOAgent 
from models.rl.practice.agents.sac import SACAgent
from models.rl.utils.train import train_agent, train_agent_with_buffer



def render_agent(agent, env_name: str, episodes: int = 1):
    env = gym.make(env_name, render_mode="human")
    for _ in range(episodes):
        state, info = env.reset()
        done = False
        while not done:
            action, _ = agent.get_action(state, training=False)
            if isinstance(env.action_space, gym.spaces.Box):
                low, high = env.action_space.low, env.action_space.high
                action = low + 0.5 * (action + 1.0) * (high - low)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Run PPO/SAC agent on OpenAI Gym continuous environment")
    parser.add_argument('--env', type=str, default='Humanoid-v5', help='Gym environment name')
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac'], help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--buffer', type=str, default='no', help='Training without buffer')
    args = parser.parse_args()

    print(f"\n=== RL Demo on Gym ===")
    print(f"Environment: {args.env}")
    print(f"Algorithm: {args.algo}")

    # Create environment
    env = gym.make(args.env)
    obs_space = env.observation_space
    act_space = env.action_space

    if not isinstance(act_space, gym.spaces.Box):
        raise ValueError("This script only supports continuous action spaces (gym.spaces.Box)")

    state_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]

    agent = PPOAgent(state_dim, action_dim) if args.algo == 'ppo' else SACAgent(state_dim, action_dim)

    print(f"\n--- Training for {args.episodes} episodes ---")
    if args.buffer == 'no':
        rewards = train_agent(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, render=args.render)
    else :
        rewards = train_agent_with_buffer(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, render = args.render)
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    agent.save('ppo_humanoid.pth')
    env.close()

    env = gym.make(args.env)  # just to get dims (or store them)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    agent2 = PPOAgent(obs_dim, act_dim, device="cuda")
    agent2.load("ppo_humanoid.pth")
    render_agent(agent2, args.env, episodes=args.episodes)



if __name__ == "__main__":
    main() 