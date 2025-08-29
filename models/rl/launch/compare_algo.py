import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'rl'))
from models.rl.agents.ppo import PPOAgent 
from models.rl.agents.ppoWdiff import DiffusionPPOAgent
from models.rl.agents.reppo import *
from models.rl.agents.sac import SACAgent
from models.rl.utils.train import train_agent, train_agent_with_buffer, train_reppo

def compare_and_visualize_ppo_vs_reppo(
    env_name: str = "Humanoid-v5",
    ppo_episodes: int = 200,
    ppo_max_steps: int = 1000,
    ppo_warmup_steps: int = 0,          # set to 5 if you want the same warmup as train_agent
    reppo_total_steps: int = 10000,
    reppo_num_step: int = 128,
    reppo_num_epoch: int = 16,
    reppo_num_mini_batch: int = 8,
    seed: int = 0,
    save_dir: str | None = None,
    show_plots: bool = True,
):
    """
    Train PPO (NO BUFFER) and RePPO on the same env and visualize:
      • Episode reward curves (raw + moving average)
      • Loss curves (PPO: policy/value; RePPO: actor/critic)

    Notes:
      - PPO loop mirrors your train_agent(): get_action → map to env range → step →
        store_transition(state, mapped_action, ...) → update() once per episode.
      - RePPO loop is mini-batch SGD per update with actor/critic loss tracking.
    """
    import numpy as np
    import torch
    import random
    import matplotlib.pyplot as plt
    import gymnasium as gym
    from tensordict import TensorDict

    # --- helpers ---
    def set_seed(sd: int):
        random.seed(sd); np.random.seed(sd)
        torch.manual_seed(sd); torch.cuda.manual_seed_all(sd)

    def moving_avg(x, k=10):
        if len(x) == 0: return np.array([])
        k = max(1, min(k, len(x)))
        c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
        return (c[k:] - c[:-k]) / k

    def _env_action_from_unit(action, space):
        # Map [-1, 1] -> env range if continuous Box
        if hasattr(space, "low") and isinstance(space, gym.spaces.Box):
            low, high = space.low, space.high
            return low + 0.5 * (action + 1.0) * (high - low)
        return action

    def _episode_stats_from_rollout_local(transition_td: TensorDict):
        rewards = transition_td['rewards'].squeeze(-1)
        dones   = transition_td['dones'].squeeze(-1).bool()
        truncs  = transition_td['truncations'].squeeze(-1).bool()
        finished = dones | truncs
        T, N = rewards.shape
        running_ret = torch.zeros(N, device=rewards.device)
        ep_returns = []
        for t in range(T):
            running_ret += rewards[t]
            fin = finished[t]
            if fin.any():
                ep_returns += running_ret[fin].tolist()
                running_ret[fin] = 0.0
        return ep_returns

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ================= PPO (NO BUFFER) =================
    ppo_env = gym.make(env_name)
    obs_space, act_space = ppo_env.observation_space, ppo_env.action_space
    state_dim, action_dim = obs_space.shape[0], act_space.shape[0]

    ppo_agent = PPOAgent(state_dim, action_dim, device)

    ppo_rewards: list[float] = []
    ppo_policy_losses: list[float] = []
    ppo_value_losses: list[float] = []

    for ep in range(ppo_episodes):
        state, _ = ppo_env.reset(seed=seed if ep == 0 else None)

        # optional warmup steps (mirrors your train_agent)
        for _ in range(ppo_warmup_steps):
            _, r, term, trunc, _ = ppo_env.step(ppo_env.action_space.sample())
            if term or trunc:
                state, _ = ppo_env.reset()

        ep_ret = 0.0
        for t in range(ppo_max_steps):
            action_unit, info = ppo_agent.get_action(state, training=True)  # in [-1,1]
            env_action = _env_action_from_unit(action_unit, ppo_env.action_space)
            next_state, reward, terminated, truncated, _ = ppo_env.step(env_action)
            done = bool(terminated or truncated)

            # IMPORTANT: keep consistent with your train_agent (store MAPPED action)
            ppo_agent.store_transition(state, env_action, reward, next_state, done, info)

            ep_ret += reward
            state = next_state
            if done:
                break

        logs = ppo_agent.update() or {}
        ppo_rewards.append(ep_ret)
        if "policy_loss" in logs: ppo_policy_losses.append(float(logs["policy_loss"]))
        if "value_loss"  in logs: ppo_value_losses.append(float(logs["value_loss"]))

        print(f"[PPO] Episode {ep+1}/{ppo_episodes}  Return: {ep_ret:.2f}  "
              f"pi_loss: {logs.get('policy_loss', float('nan')):.4f}  "
              f"v_loss: {logs.get('value_loss', float('nan')):.4f}")

    ppo_env.close()

    # ================= RePPO =================
    reppo_env = gym.make(env_name)
    sdim2, adim2 = reppo_env.observation_space.shape[0], reppo_env.action_space.shape[0]

    # from models.rl.agents.reppo import EmpiricalNormalizer, RePPOAgent, compute_gve
    obs_norm = EmpiricalNormalizer(shape=sdim2, device=device)
    critic_obs_norm = EmpiricalNormalizer(shape=sdim2, device=device)
    reppo_agent = RePPOAgent(
        observation_dim=sdim2, action_dim=adim2,
        obs_normalizer=obs_norm, critic_obs_normalizer=critic_obs_norm
    )

    reppo_rewards: list[float] = []
    reppo_actor_losses: list[float] = []
    reppo_critic_losses: list[float] = []

    N_envs = getattr(reppo_env, 'num_envs', 1)
    batch_size = (N_envs * reppo_num_step) // reppo_num_mini_batch
    total_updates = reppo_total_steps * 10 // (N_envs * reppo_num_step) + 1

    reset_return, _ = reppo_env.reset(seed=seed)
    observation = reset_return[0] if isinstance(reset_return, tuple) else reset_return
    critic_observation = None
    global_update = 0

    while global_update < total_updates:
        transition, observation, critic_observation, infos = reppo_agent.collect(
            reppo_env, observation, critic_observation, reppo_num_step
        )

        # episode returns from rollout
        ep_returns = _episode_stats_from_rollout_local(transition)
        reppo_rewards.extend(ep_returns)

        # targets
        gves = compute_gve(
            rewards=transition['rewards'],
            dones=transition['dones'],
            truncations=transition['truncations'],
            next_values=transition['next_values'],
            gamma=reppo_agent.gamma,
            lmbda=reppo_agent.lmbda,
        )

        data = TensorDict(
            {
                'observation': transition['observation'],
                'critic_observation': transition['critic_observation'],
                'actions': transition['actions'],
                'rewards': transition['rewards'],
                'raw_rewards': transition['raw_rewards'],
                'next_embedding': transition['next_embedding'],
                'next_values': transition['next_values'],
                'dones': transition['dones'],
                'truncations': transition['truncations'],
                'gve': torch.stack(gves, dim=0),
            },
            batch_size=(reppo_num_step, N_envs),
            device=reppo_agent.device
        ).float().flatten(0, 1).detach()

        # SGD epochs with mini-batches; avg losses per update
        actor_loss_accum, critic_loss_accum = [], []
        for _ in range(reppo_num_epoch):
            index = torch.randperm(reppo_num_step * N_envs, device=reppo_agent.device)
            data_shuffle = data[index].contiguous()
            for mb in range(reppo_num_mini_batch):
                batch = data_shuffle[mb * batch_size:(mb + 1) * batch_size]
                critic_logs = reppo_agent.update_critic(batch) or {}
                actor_logs  = reppo_agent.update_actor(batch)  or {}
                if "loss" in critic_logs: critic_loss_accum.append(float(critic_logs["loss"]))
                elif "critic_loss" in critic_logs: critic_loss_accum.append(float(critic_logs["critic_loss"]))
                if "loss" in actor_logs: actor_loss_accum.append(float(actor_logs["loss"]))
                elif "actor_loss" in actor_logs or "policy_loss" in actor_logs:
                    actor_loss_accum.append(float(actor_logs.get("actor_loss", actor_logs.get("policy_loss"))))

        with torch.no_grad():
            for p, q in zip(reppo_agent.actor.parameters(), reppo_agent.old_actor.parameters()):
                q.data.copy_(p.data)

        if actor_loss_accum:  reppo_actor_losses.append(float(np.mean(actor_loss_accum)))
        if critic_loss_accum: reppo_critic_losses.append(float(np.mean(critic_loss_accum)))

        global_update += 1
        if global_update * reppo_num_step >= reppo_total_steps:
            break

    reppo_env.close()

    # ================= Stats & Plots =================
    import os as _os
    print(f"PPO:   Mean={np.mean(ppo_rewards):.2f} ± {np.std(ppo_rewards):.2f}  n={len(ppo_rewards)}")
    print(f"RePPO: Mean={np.mean(reppo_rewards):.2f} ± {np.std(reppo_rewards):.2f}  n={len(reppo_rewards)}")

    plt.figure(figsize=(12, 10))

    # 1) Rewards
    plt.subplot(3, 1, 1)
    x1 = np.arange(1, len(ppo_rewards) + 1)
    x2 = np.arange(1, len(reppo_rewards) + 1)
    plt.plot(x1, ppo_rewards, alpha=0.35, label="PPO (episode rewards)")
    if len(ppo_rewards) > 1:
        y1 = moving_avg(ppo_rewards, 10)
        plt.plot(np.arange(len(y1)) + 10, y1, label="PPO (MA@10)")
    plt.plot(x2, reppo_rewards, alpha=0.35, label="RePPO (episode rewards)")
    if len(reppo_rewards) > 1:
        y2 = moving_avg(reppo_rewards, 10)
        plt.plot(np.arange(len(y2)) + 10, y2, label="RePPO (MA@10)")
    plt.title(f"Rewards on {env_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2) PPO losses
    plt.subplot(3, 1, 2)
    if len(ppo_policy_losses) > 0:
        plt.plot(np.arange(1, len(ppo_policy_losses) + 1), ppo_policy_losses, label="PPO policy loss")
    if len(ppo_value_losses) > 0:
        plt.plot(np.arange(1, len(ppo_value_losses) + 1), ppo_value_losses, label="PPO value loss")
    plt.title("PPO Losses (per episode update)")
    plt.xlabel("Updates (episodes)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3) RePPO losses
    plt.subplot(3, 1, 3)
    if len(reppo_actor_losses) > 0:
        plt.plot(np.arange(1, len(reppo_actor_losses) + 1), reppo_actor_losses, label="RePPO actor loss")
    if len(reppo_critic_losses) > 0:
        plt.plot(np.arange(1, len(reppo_critic_losses) + 1), reppo_critic_losses, label="RePPO critic loss")
    plt.title("RePPO Losses (per update)")
    plt.xlabel("Updates")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir is not None:
        _os.makedirs(save_dir, exist_ok=True)
        fig_path = _os.path.join(save_dir, f"ppo_vs_reppo_{env_name.replace('-', '_')}.png")
        plt.savefig(fig_path, dpi=200)
        print(f"Saved figure to {fig_path}")

    if show_plots:
        plt.show()

    return {
        "ppo_rewards": ppo_rewards,
        "ppo_policy_losses": ppo_policy_losses,
        "ppo_value_losses": ppo_value_losses,
        "reppo_rewards": reppo_rewards,
        "reppo_actor_losses": reppo_actor_losses,
        "reppo_critic_losses": reppo_critic_losses,
    }
