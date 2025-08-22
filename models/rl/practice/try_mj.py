import argparse, os, sys
import gymnasium as gym
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'rl'))
from models.rl.practice.agents.reppo import *
from models.rl.practice.agents.ppo import *
from models.rl.practice.agents.sac import *
from models.rl.utils.train import train_reppo, train_agent_with_buffer, train_agent
from models.rl.envs.openarm_mj_env import OpenArmMjEnv

def make_env(sim, mjcf, steps, render):
    if sim == "mujoco":
        # from brain2rl.models.rl.envs.openarm_mj_env import OpenArmMjEnv
        xml = mjcf or os.path.expanduser("~/brain2rl/external/openarm_mujoco/v1/scene.xml")
        return OpenArmMjEnv(xml_path=xml, horizon=steps, render=render)
    else:
        # your existing Gym env fallback
        import gymnasium as gym
        return gym.make("Humanoid-v5")
    

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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sim", choices=["mujoco","gym"], default="mujoco")
    p.add_argument("--mjcf", type=str, default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--steps", type=int, default=300)
    # ... your existing RL args (algo, lr, etc.)
    args = p.parse_args()

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

    if args.algo == "reppo":
        agent = RePPOAgent(observation_dim=state_dim, action_dim=action_dim)
        rewards = train_reppo(env, agent, total_steps=10000)

    elif args.algo == "ppo":
        agent = PPOAgent(observation_dim=state_dim, action_dim=action_dim)
        rewards = train_agent_with_buffer(env, agent, num_episodes=args.episodes, max_steps=args.max_steps, render = args.render)

        
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    agent.save(f'{args.algo}_openarm_mujoco.pth')
    env.close()


    render_agent(agent, args.env, episodes=args.episodes)

