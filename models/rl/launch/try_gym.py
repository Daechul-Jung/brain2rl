import argparse
import gymnasium as gym
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'rl'))
from models.rl.agents.ppo import PPOAgent 
from models.rl.agents.ppoWdiff import DiffusionPPOAgent
from models.rl.agents.reppo import *
from models.rl.agents.sac import SACAgent
from models.rl.utils.train import train_agent, train_reppo
from models.rl.launch.compare_algo import *


def compare_result(list_of_rewards):
    for algo, rewards in list_of_rewards:
        print(f"{algo}: \nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")


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
    parser.add_argument('--env', type=str, default='HumanoidStandup-v5', help='Gym environment name')   #####  HumanoidStandup-v5, Reacher-v5, Humanoid-v5
    parser.add_argument('--algo', type=str, default='ppo', choices=['ppo', 'sac', 'reppo', 'comic'], help='RL algorithm to use')
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
    print(f'observation dimension: {obs_space.shape} and action dimension: {act_space.shape}')

    if not isinstance(act_space, gym.spaces.Box):
        raise ValueError("This script only supports continuous action spaces (gym.spaces.Box)")

    state_dim = obs_space.shape[0]
    action_dim = act_space.shape[0]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.algo == "reppo":
        obs_normalizer = EmpiricalNormalizer(shape=state_dim, device=device)
        critic_obs_normalizer = EmpiricalNormalizer(shape=state_dim, device=device)
        agent = RePPOAgent(observation_dim=state_dim, action_dim=action_dim, obs_normalizer=obs_normalizer, critic_obs_normalizer=critic_obs_normalizer)
        rewards = train_reppo(env, agent, total_steps=10000)

    elif args.algo == "ppo" :
        agent = PPOAgent(state_dim, action_dim, device)
        rewards = train_agent(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, render = args.render)

    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    # agent2 = PPOAgent(state_dim, action_dim)
    # ppo_reward = train_agent(env, agent2, num_episodes=782, max_steps=10000)
    
    agent.save(f'{args.algo}_humanoid.pth')
    env.close()
    # compare_result(list([('reppo', rewards), ('ppo', ppo_reward)]))

    render_agent(agent, args.env, episodes=args.episodes)

if __name__ == "__main__":
    main() 