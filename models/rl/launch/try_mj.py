import argparse, os, sys
import gymnasium as gym

sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'rl'))

from models.rl.agents.reppo import *
from models.rl.agents.ppo import *
from models.rl.agents.sac import *
from models.rl.utils.train import train_reppo, train_agent_with_buffer, train_agent
from models.rl.envs.openarm_mj_env import OpenArmMjEnv

def make_env(sim, mjcf, steps, render):
    
    if sim == "mujoco":
        # Use the new scene file with cameras and multiple cups
        xml = mjcf or os.path.join(os.path.dirname(__file__), '..', 'envs', 'openarm_scene_with_cameras.xml')

        return OpenArmMjEnv(
            xml_path=xml,
            camera='left_wrist_cam',  # Use the wrist-mounted camera
            camera_size=(256, 256),
            horizon=steps, 
            render=render,
            vision_rewards_weight=0.4,  # 40% vision, 60% physics
            physics_rewards_weight=0.6,
            target_cup='cup1'  # Target the brown cup
        )
    
    else:
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
    p.add_argument("--algo", type=str, default='reppo')
    p.add_argument("--sim", choices=["mujoco","gym"], default="mujoco")
    p.add_argument("--mjcf", type=str, default=None)
    p.add_argument("--render", action="store_true")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--env", type=str, default="Humanoid-v5")
    # ... your existing RL args (algo, lr, etc.)
    args = p.parse_args()

    print(f"\n=== RL Demo on OpenArm with Computer Vision ===")
    print(f"Environment: {args.sim}")
    print(f"Algorithm: {args.algo}")

    # Create environment
    import mujoco
    env = make_env(args.sim, args.mjcf, args.steps, args.render)
    obs_space = env.observation_space
    act_space = env.action_space
    print(f'Observation space: {obs_space.shape} and action space: {act_space.shape}')
    print(f'Vision features: 6 additional features for cup detection')

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

    elif args.algo == "ppo":
        agent = PPOAgent(observation_dim=state_dim, action_dim=action_dim)
        rewards = train_agent(env, agent, num_episodes=10000, max_steps=10000)

        
    print(f"\nMean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    agent.save(f'{args.algo}_openarm_mujoco.pth')
    env.close()

    # Note: render_agent is commented out as it's designed for gym environments
    # render_agent(agent, args.env, episodes=args.episodes)

