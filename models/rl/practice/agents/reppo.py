import os
import sys
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.optim as optim
import copy
import gymnasium as gym
import torch.nn.functional as F
from tensordict import TensorDict
from typing import Dict, Tuple, Optional
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.rl.utils.reppo_network import *

"""
This algorithm is Relative Entropy Pairwise Policy Optimization (RePPO)
This is on-policy and actor-critic alogrithm and train robust surrogate value function and effectively use pairwise policy gradient.
Build Maximum entropy framework and combines with a principled KL regularization, 
Categorical Q-Learning, Appropriately normalized neural network architecture and auxilary task
Show how a joint entropy and policy deviation tuning objective can address the twin problems of sufficient exploration and controlled policy update 
According to the paper, it does not utilize the replay buffer
"""

def _env_shape(env):
    num_envs = getattr(env, "num_envs", 1)
    max_steps = getattr(env, "max_episode_steps", 5000)
    asymmetric = getattr(env, "asymmetric_obs", False )
    return num_envs, max_steps, asymmetric

def _to_tensor(input, device, dtype=torch.float32):
    """
    Converting input to tensor with the given device
    """
    if torch.is_tensor(input):
        return input.to(device = device, dtype=dtype if input.dtype.is_floating_point else input.dtype)
    
    return torch.as_tensor(input, device=device, dtype=dtype)

def _split_step_return(returns):
    if len(returns) == 5:
        next_observation, reward, done, truncated, info = returns 
    else:
        next_observation, reward, done, info = returns 
        truncated = torch.zeros_like(done, dtype=torch.bool) if torch.is_tensor(done) else False
    return next_observation, reward, done, truncated, info

def compute_gve(rewards, dones, truncations, next_values, gamma:float, lmbda:float):
    """
    Compute Generalized Value Estimator for REPPO
    """
    gves = []
    last_gve = torch.zeros_like(next_values[-1])
    trunc = truncations.clone()
    trunc[-1] = 1.0

    for t in reversed(range(rewards.shape[0])):
        lmbda_sum = lmbda * last_gve + (1 - lmbda) * next_values[t]
        delta = gamma * torch.where(trunc[t].bool(), next_values[t], (1.0 - dones[t]) * lmbda_sum)
        last_gve = rewards[t] + delta
        gves.insert(0, last_gve)

    return gves

@dataclass(slots=True)
class TrainState:
    device: torch.device
    obs: torch.Tensor
    critic_obs: torch.Tensor
    actor: Actor
    old_actor: Actor
    critic: Critic
    # normalizer: EmpiricalNormalization
    # critic_normalizer: EmpiricalNormalization
    actor_optimizer: optim.Optimizer
    critic_optimizer: optim.Optimizer
    # scaler: GradScaler



class RePPOAgent:
    def __init__(self, observation_dim, action_dim, num_atoms = 101, 
                 vmin=-250, vmax=250, device='cuda',
                 lr = 3e-4, gamma = 0.99, kl_start = 0.1, entropy_start = 0.1,
                 lmbda = 0.95):
        super().__init__()
        
        self.critic = Critic(observation_dim, action_dim, num_atoms, 
                             vmin=vmin, vmax=vmax, device=device)
        self.vmin = vmin
        self.vmax = vmax
        self.gamma = gamma
        self.lmbda = lmbda
        
        self.num_atoms = num_atoms
        self.observation_dim = observation_dim
        self.action_dim = action_dim 
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.aux_loss_mult = 1.0
        self.max_grad_norm = 1.0
        self.kl_bound = 0.02
        # Actor(Policy) includes entropy and kl-regularization which are subject to optimization
        self.actor = Actor(observation_dim=observation_dim,
                            action_dim=action_dim, 
                           kl_start=kl_start, entropy_start=entropy_start, 
                           device=device)
        self.old_actor = copy.deepcopy(self.actor).to(self.device)
        self.critic = Critic(observation_dim=observation_dim, 
                             action_dim=action_dim, 
                             num_atoms=num_atoms, 
                             vmin=vmin, vmax=vmax, 
                             device=device)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr = lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr = lr)
        
    def _actor_forward(self, observation):
        """
        Forward pass for the actor network
        
        Returns:
            pi: Action probabilities with transformed distributions
            mean: Mean of the action distribution
            log_temp: Log of entropy temperature
            log_lagrange: Log of KL regularization parameter
            
        """
        pi, mean, log_temp, log_lagrange = self.actor(observation)
        return pi, mean, log_temp, log_lagrange
    
    def _old_actor_dist(self, observation):
        """
        Based on the observation with old actor network, getting old pi distribution
        """

        old_pi, _, _, _ = self.old_actor(observation)

        return old_pi
    
    
    def _critic_forward(self, observation, action):
        """
        Critic returns value, log, next_pred, and featuers. Critic network has three modules which are feature module, prediction module, critic module. 
        Firstly, concatenate observation and action and put those in feature FCNN network. Then, using features resulted from feature network, get next pred
        Embedded next pred is current state's prediction of the next embedding prediction and via feature network, we can store next state's target embedding 
        """
        value, logits, next_pred_feature, features = self.critic(observation, action)

        return value, logits, next_pred_feature, features 

    def update_critic(self, batch: TensorDict):
        """
        Update critic network in training mode
        """
        self.critic.train()

        observation_critic = batch['critic_observation'] # Shape: (Batch, observation_critic)
        actions = batch['actions'] # shape: (Batch, Action)
        targets = batch['gve'].squeeze(-1) # Shape: (Batch, )
        truncated = batch['truncation'].squeeze(-1) # Shape: (Batch, )
        target_next_feature = batch['next_embedding'] # Shape: (Batch, )

        trunc_mask = (1.0 - truncated).to(observation_critic.dtype)

        with torch.no_grad():
            q_target_dist = hl_gauss(targets, self.vmin, self.vmax, self.num_atoms)
        ## Getting Q-value from critic network 
        q_scalar, q_logit, next_pred, _features = self._critic_forward(observation_critic, actions)
        log_probs = F.log_softmax(q_logit, dim = -1)
        ce = -(q_target_dist * log_probs).sum(-1)

        emb_loss = F.mse_loss(next_pred, target_next_feature, reduction=None)
        loss = (trunc_mask * (ce +  self.aux_loss_mult * emb_loss)).mean()
        self.critic_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)    
        self.critic_optimizer.step()

        return {
            "qf_loss": loss.detach(),
            "qf_mean": targets.mean().detach(),
            "qf_max": targets.max().detach(),
            "qf_min": targets.min().detach(),
            "embedding_loss": emb_loss.detach()
        }
    
    def update_actor(self, batch: TensorDict):
        """
        Update actor network in training mode

        The main difference between actor and critic network is in actor network, 
        the agent got observation and observation critic from batch
        """
        self.actor.train()

        observation = batch['observation']
        observation_critic = batch['critic_observation']
        ## Get action distribution, mean, entropy, and kl-regulation from actor network 
        pi, mean, log_temp, log_lagrange = self._actor_forward(observation=observation) 
        # Then resample the action from the distribution(Transformed)
        actions = pi.rsample() # Shape: (Batch, Action)

        actions_for_log = torch.clamp(actions, -1 + 1e-6, 1 - 1e-6) 

        log_pi = pi.log_prob(actions_for_log).sum(-1)

        entropy = -log_pi

        q_scalar, _, _, _ = self._critic_forward(observation_critic, actions)
        # Actor objective 
        actor_obj = -q_scalar + log_temp.detach() * log_pi
        ## KL(new||old) using old policy sample 
        with torch.no_grad():
            # Sample from the old distribution
            old_pi = self._old_actor_dist(observation)
            old_sample = old_pi.sample((16, )).clamp(-1 + 1e-6, 1 - 1e-6)
            #### Should take a look the shape and how it looks like 
            old_log_prob = old_pi.log_prob(old_sample).sum(-1) # Estimate the log probability with the given old distribuiton
        # Estimate the old log probability from the given new action distribution
        new_log_prob = pi.log_prob(old_sample).sum(-1)
        kl_est = (old_log_prob - new_log_prob).mean(0)
        
        kl_bound = self.kl_bound
        clipped = torch.where(kl_est < kl_bound, actor_obj, kl_est * log_lagrange)

        ### entropy temperature and KL-Lagrangian terms and loss 
        target_entropy = observation.new_tensor(observation.shape[-1])

        entropy_loss = (target_entropy + entropy).detach().mean() * log_temp

        lagrangian = - log_lagrange * (kl_est - kl_bound).mean().detach()
        total_loss = clipped.mean() + entropy_loss + lagrangian
        self.actor_optimizer.zero_grad(set_to_none=True)

        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

        self.actor_optimizer.step()
        return {
            "actor_loss": total_loss.detach(),
            "kl": kl_est.mean().detach(),
            "entropy": entropy.mean().detach(),
            "temperature": log_temp.detach(),
            "lagrangian": log_lagrange.detach(),
            "entropy_loss": entropy_loss.detach(),
            "lagrangian_loss": lagrangian.detach(),
        }

    
    @torch.no_grad()
    def collect(self, env: gym.Env, observation: Optional[torch.Tensor], critic_observation: Optional[torch.Tensor], num_steps = 10000):
        """
        On-policy rollout and return (transition tensordict, final_obs, final_critic_obs, infos)
        """
        N, _, asymmetric = _env_shape(env)
        trajectory = []
        info_list = []
        ## initial reset
        if observation is None:
            reset_return = env.reset()
            observation = reset_return[0] if isinstance(reset_return, tuple) else reset_return

        if critic_observation is None:
            critic_observation = observation

        observation = _to_tensor(observation, self.device)
        critic_observation = _to_tensor(critic_observation, self.device)

        for _ in range(num_steps):
            pi, _, log_temp, log_lagrange = self._actor_forward(observation)
            action = pi.sample()
            step_return = env.step(action)

            next_observation, rewards, dones, truncated, infos = _split_step_return(step_return)
            next_critic_observation = next_observation

            _next_observation = _to_tensor(next_observation, self.device)
            _next_critic_obs = _to_tensor(next_critic_observation, self.device)

            next_pi, _, next_log_temp, next_log_lagran = self._actor_forward(_next_observation)

            next_action = next_pi.sample()
            ### Take sum over batch dimension
            next_log_prob = next_pi.log_prob(next_action.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1)

            next_value, _, next_pred_unused, next_features = self._critic_forward(next_critic_observation, next_action)
            rewards = _to_tensor(rewards, self.device).view(-1)

            shaped_reward = rewards - self.gamma * next_log_prob * (next_log_temp if torch.is_tensor(next_log_temp) else float(next_log_temp))
            
            td = TensorDict(
                {
                    'observation': observation,
                    'critic_observation': critic_observation,
                    'actions': action,
                    'log_prob': pi.log_prob(action.clamp(-1 + 1e-6, 1 - 1e-6)).sum(-1),
                    'rewards': shaped_reward.unsqueeze(-1),
                    'next_embedding': next_features, ### Target for aux loss is the next state encoder features
                    'next_values': next_value.unsqueeze(-1),
                    'dones': _to_tensor(dones, self.device, dtype=torch.float32).unsqueeze(-1),
                    'truncations': _to_tensor(truncated, self.device, dtype = torch.float32).unsqueeze(-1)
                },
                batch_size=(N, )
            )
            trajectory.append(td)
            info_list.append(infos)
            
            observation = _to_tensor(next_observation, self.device)
            critic_observation = _to_tensor(next_critic_observation, self.device)
            
        transition  = torch.stack(trajectory, dim = 0) ## Shape: (T, N)
            
        return transition, observation, critic_observation, info_list
    
    @torch.no_grad()
    def evaluate(
        self,
        env,
        episodes: int = 5,
        stochastic: bool = False,
        max_steps: int | None = None,
    ):
        """
        Roll out the current policy on `env` and return average/individual episode returns and lengths.
        Uses *environment reward* (not shaped reward).
        """
        self.actor.eval()
        device = self.device

        returns = []
        lengths = []

        for _ in range(episodes):
            reset_ret = env.reset()
            obs = reset_ret[0] if isinstance(reset_ret, tuple) else reset_ret
            obs = torch.as_tensor(obs, device=device, dtype=torch.float32)
            ep_ret = 0.0
            ep_len = 0

            horizon = max_steps or getattr(env, "max_episode_steps", 1000)

            for _t in range(horizon):
                pi, det, _, _ = self._actor_forward(obs.unsqueeze(0))
                if stochastic:
                    action = pi.sample()
                else:
                    action = det.unsqueeze(0)  # deterministic = tanh(mean)

                step_ret = env.step(action)
                next_obs, reward, done, trunc, info = (
                    step_ret if len(step_ret) == 5
                    else (*step_ret, False)  # gym classic (no truncation flag)
                )
                ep_ret += float(torch.as_tensor(reward).mean().item())
                ep_len += 1

                if bool(done) or bool(trunc):
                    break

                obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)

            returns.append(ep_ret)
            lengths.append(ep_len)

        self.actor.train()
        avg_ret = float(torch.tensor(returns).mean().item())
        avg_len = float(torch.tensor(lengths).float().mean().item())
        return {
            "avg_return": avg_ret,
            "avg_length": avg_len,
            "returns": returns,
            "lengths": lengths,
        }
    
    def save(
        self,
        file_path: str,
        step: int | None = None,
        include_optim: bool = True,
        extra: dict | None = None,
    ):
        """
        Save actor/critic and (optionally) optimizers. Includes temperature/lagrange via actor state_dict.
        """
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "step": step,
        }
        if include_optim:
            ckpt.update(
                {
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                }
            )
        if extra:
            ckpt.update(extra)

        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        torch.save(ckpt, file_path)

    def load(self, file_path: str, strict: bool = True, load_optim: bool = True):
        """
        Load actor/critic (and optimizers if present). Supports multiple key names for compatibility.
        """
        ckpt = torch.load(file_path, map_location=self.device)

        def _first_key(d, *names):
            for n in names:
                if n in d:
                    return n
            return None

        # Actor
        k = _first_key(ckpt, "actor", "actor_network", "actor_state_dict")
        if k is None:
            raise KeyError("No actor state_dict found in checkpoint.")
        self.actor.load_state_dict(ckpt[k], strict=strict)

        # Critic
        k = _first_key(ckpt, "critic", "critic_network", "critic_state_dict")
        if k is None:
            raise KeyError("No critic state_dict found in checkpoint.")
        self.critic.load_state_dict(ckpt[k], strict=strict)

        # Optimizers (optional)
        if load_optim:
            if "actor_optimizer" in ckpt:
                self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
            if "critic_optimizer" in ckpt:
                self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])

        # Keep old_actor in sync with loaded actor
        with torch.no_grad():
            for p, q in zip(self.actor.parameters(), self.old_actor.parameters()):
                q.data.copy_(p.data)

        return ckpt.get("step", None)