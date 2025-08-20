from models.rl.practice.agents.ppo import *
import gymnasium as gym
from models.rl.practice.agents.reppo import *
from typing import Optional

######################## Training REPPO ###################################
def _episode_stats_from_rollout(transition: TensorDict, prefer_raw: bool = True):
    """
    Aggregate episode returns/lengths from a time-major rollout.
    If 'raw_rewards' is present it uses that; otherwise falls back to shaped 'rewards'.
    """
    key = "raw_rewards" if prefer_raw and "raw_rewards" in transition.keys(True) else "rewards"
    rewards = transition[key].squeeze(-1)           # (T, N)
    dones   = transition["dones"].squeeze(-1).bool()
    truncs  = transition["truncations"].squeeze(-1).bool()
    finished = dones | truncs

    T, N = rewards.shape
    running_ret = torch.zeros(N, device=rewards.device)
    running_len = torch.zeros(N, device=rewards.device)
    ep_returns, ep_lengths = [], []

    for t in range(T):
        running_ret += rewards[t]
        running_len += 1
        fin = finished[t]
        if fin.any():
            ep_returns += running_ret[fin].tolist()
            ep_lengths += running_len[fin].tolist()
            running_ret[fin] = 0.0
            running_len[fin] = 0.0

    return ep_returns, ep_lengths


def train_reppo(env: gym.Env, agent:RePPOAgent, total_steps = 10000, num_step= 1000, num_epoch = 1, 
                num_mini_batch = 4, logger: Optional[callable] = None, num_eval = 5, evaluate_func = None,render=False):
    
    device = agent.device
    N_envs = getattr(env, 'num_envs', 1)
    
    batch_size = (N_envs * num_step) // num_mini_batch
    
    total_updates = total_steps // (N_envs * num_step) + 1
    eval_interval = max(1, total_updates // (max(1, num_eval)))
    
    reset_return = env.reset()
    observation  = reset_return[0] if isinstance(reset_return, tuple) else reset_return
    
    critic_observation = None 
    
    global_update = 0 
    all_episode_returns = []
    while global_update < total_updates:
        transition, observation, critic_observation, infos = agent.collect(env, observation, critic_observation, num_step)
        ep_returns, ep_lengths = _episode_stats_from_rollout(transition, prefer_raw=True)
        all_episode_returns.extend(ep_returns)
        gves = compute_gve(
            rewards=transition['rewards'],
            dones= transition['dones'],
            truncations=transition['truncations'],
            next_values=transition['next_values'],
            gamma=agent.gamma,
            lmbda=agent.lmbda
        )
        data = TensorDict(
            {
                'observation': transition['observation'],
                'critic_observation': transition['critic_observation'],
                'actions': transition['actions'],
                'rewards': transition['rewards'],
                'next_embedding': transition['next_embedding'],
                'next_values': transition['next_values'],
                'dones': transition['dones'],
                'truncations': transition['truncations'],
                'gve': torch.stack(gves, dim= 0)
            },
            batch_size=(num_step, N_envs),
            device=device
        ).float().flatten(0, 1).detach() ### Shape would be (T*N , ...)
        
        ## SGD
        for _ in range(num_epoch):
            index = torch.randperm(num_step * N_envs, device=device)
            data_shuffle = data[index].contiguous()
            for mini_batch in range(num_mini_batch):
                batch_data = data_shuffle[mini_batch * batch_size: (mini_batch + 1) * batch_size]
                critic_logs = agent.update_critic(batch_data)
                actor_logs = agent.update_actor(batch_data)
                logs = {**{f"critic/{k}": v for k, v in critic_logs.items()},
                        **{f"actor/{k}":  v for k, v in actor_logs.items()}}
                
        ## syncronize old actor
        with torch.no_grad():
            for p,q in zip(agent.old_actor.parameters(), agent.old_actor.parameters()):
                q.data.copy_(data)
                
        if evaluate_func and (global_update % eval_interval == 0):
            eval_metrics = evaluate_func(agent) or {}
            if logger:
                logger(
                    step=global_update * N_envs * num_step,
                    **{f"eval/{k}": (float(v) if torch.is_tensor(v) else v) for k, v in eval_metrics.items()},
                    **{k: (float(val) if torch.is_tensor(val) else val) for k, val in logs.items()},
                )
        elif logger:
            logger(
                step=global_update * N_envs * num_step,
                **{k: (float(val) if torch.is_tensor(val) else val) for k, val in logs.items()},
            )

        global_update += 1

    return all_episode_returns

###############################################################################################


def train_agent(env, agent, num_episodes, max_steps=1000, render=False):
    episode_rewards = []
    for ep in range(num_episodes):
        # For every episode, reset the environment 
        state, info = env.reset()
        episode_reward = 0.0
        for t in range(max_steps):
            if render:
                env.render()

            action, action_info = agent.get_action(state, training=True)

            # map [-1,1] -> env range
            if hasattr(env.action_space, "low") and isinstance(env.action_space, gym.spaces.Box):
                low, high = env.action_space.low, env.action_space.high
                action = low + 0.5 * (action + 1.0) * (high - low)

            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done, action_info)

            state = next_state
            episode_reward += reward
            if done:
                break
        # After every episode, update the agent 
        agent.update()
        episode_rewards.append(episode_reward)
        print(f"Episode {ep+1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
    return episode_rewards

def train_agent_with_buffer(env, agent:PPOAgent, num_episodes:int, max_steps = 1000, render=False):
    episode_rewards = []
    episode_done = 0

    state, info = env.reset()
    episode_ret = 0.0
    episode_len = 0

    while episode_done < num_episodes:
        for t in range(agent.step_per_rollout):
            if render:
                env.render()

            action, log_prob, value, action_before_tahn = agent.get_action_with_buffer(state, training=True)

            if hasattr(env.action_space, "low") and isinstance(env.action_space, gym.spaces.Box):
                low, high = env.action_space.low, env.action_space.high
                env_action = low + 0.5 * (action + 1.0) * (high - low)

            else:
                env_action = action

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = bool(terminated or truncated)

            agent.buffer.store(
                obs = state, 
                act = action_before_tahn,
                rew = reward,
                val = value, 
                logp = log_prob,
                done = done
            )

            episode_ret += reward
            episode_len += 1

            state = next_state

            if done or episode_len >= max_steps:
                agent.buffer.finish_path(last_val=0)
                episode_rewards.append(episode_ret)
                episode_done += 1
                state, info = env.reset()
                episode_ret = 0
                episode_len = 0

                if episode_done > num_episodes:
                    break 
        if episode_len > 0:
            with torch.no_grad():
                s = np.asarray(state, dtype=np.float32)
                if s.ndim == 1:
                    s = s[None, :]
                v_boot = agent.value_net(torch.from_numpy(s).to(agent.device)).squeeze(-1).item()
            agent.buffer.finish_path(last_val=v_boot)

            # Update PPO on the collected rollout
            info = agent.updateWithBuffer()
            if info:
                print(f"Update: policy_loss={info['policy_loss']:.4f}, value_loss={info['value_loss']:.4f}")
            # Clear buffer for next iteration
            agent.buffer.clear()
        return episode_rewards
