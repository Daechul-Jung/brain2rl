import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import List, Tuple, Dict
import gym

class DQNAgent:
    """Deep Q-Network Agent"""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update = target_update
        self.update_counter = 0
        
        # Create Q-Network and Target Network
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def _build_network(self) -> nn.Module:
        """Build the neural network for Q-learning"""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.argmax().item()
    
    def replay(self) -> float:
        """Train on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss and update
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()

class PPOAgent:
    """Proximal Policy Optimization Agent"""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Create actor-critic network
        self.actor_critic = self._build_network()
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
    
    def _build_network(self) -> nn.Module:
        """Build the actor-critic network"""
        class ActorCritic(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Linear(state_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 128),
                    nn.ReLU()
                )
                
                self.actor = nn.Sequential(
                    nn.Linear(128, action_size),
                    nn.Softmax(dim=-1)
                )
                
                self.critic = nn.Linear(128, 1)
            
            def forward(self, x):
                shared_features = self.shared(x)
                return self.actor(shared_features), self.critic(shared_features)
        
        return ActorCritic(self.state_size, self.action_size)
    
    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Choose action and get value estimate"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = self.actor_critic(state)
            
            # Sample action
            action = torch.multinomial(action_probs, 1).item()
            log_prob = torch.log(action_probs[0, action])
            
            return action, log_prob.item(), value.item()
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        running_return = 0
        previous_value = 0
        running_advantage = 0
        
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            running_return = r + self.gamma * running_return * (1 - d)
            returns.insert(0, running_return)
            
            td_error = r + self.gamma * previous_value * (1 - d) - v
            running_advantage = td_error + self.gamma * self.gae_lambda * running_advantage * (1 - d)
            advantages.insert(0, running_advantage)
            
            previous_value = v
        
        return np.array(advantages), np.array(returns)
    
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray
    ) -> Dict[str, float]:
        """Update policy and value function"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        total_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.update_epochs):
            # Create mini-batches
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                # Get current policy and value
                action_probs, values = self.actor_critic(states[idx])
                new_log_probs = torch.log(action_probs.gather(1, actions[idx].unsqueeze(1))).squeeze()
                
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = self.value_coef * self.criterion(values.squeeze(), returns[idx])
                
                entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()
                
                loss = actor_loss + value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
        
        return {
            'loss': total_loss / self.update_epochs,
            'value_loss': total_value_loss / self.update_epochs,
            'entropy': total_entropy / self.update_epochs
        }

class SACAgent:
    """Soft Actor-Critic Agent"""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        memory_size: int = 100000,
        batch_size: int = 256
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        # Create networks
        self.actor = self._build_actor()
        self.critic1 = self._build_critic()
        self.critic2 = self._build_critic()
        self.target_critic1 = self._build_critic()
        self.target_critic2 = self._build_critic()
        
        # Initialize target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
    
    def _build_actor(self) -> nn.Module:
        """Build the actor network"""
        return nn.Sequential(
            nn.Linear(self.state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_size * 2)  # Mean and log_std for each action
        )
    
    def _build_critic(self) -> nn.Module:
        """Build the critic network"""
        return nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def act(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Choose action using current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            mean, log_std = self.actor(state).chunk(2, dim=-1)
            
            if evaluate:
                return torch.tanh(mean).numpy()
            
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            
            return action.numpy()
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self) -> Dict[str, float]:
        """Update networks"""
        if len(self.memory) < self.batch_size:
            return {'actor_loss': 0, 'critic_loss': 0}
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_actions = self.act(next_states.numpy())
            next_actions = torch.FloatTensor(next_actions)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], dim=1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], dim=1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q1 = self.critic1(torch.cat([states, actions], dim=1))
        current_q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions = self.act(states.numpy())
        new_actions = torch.FloatTensor(new_actions)
        q1 = self.critic1(torch.cat([states, new_actions], dim=1))
        q2 = self.critic2(torch.cat([states, new_actions], dim=1))
        q = torch.min(q1, q2)
        
        actor_loss = -q.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._update_target_network(self.target_critic1, self.critic1)
        self._update_target_network(self.target_critic2, self.critic2)
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2
        }
    
    def _update_target_network(self, target: nn.Module, source: nn.Module):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

def train_agent(
    env: gym.Env,
    agent: Union[DQNAgent, PPOAgent, SACAgent],
    num_episodes: int,
    max_steps: int = 1000,
    render: bool = False
) -> List[float]:
    """Train an RL agent"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps):
            if render:
                env.render()
            
            if isinstance(agent, DQNAgent):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.replay()
            
            elif isinstance(agent, PPOAgent):
                action, log_prob, value = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                # Store transition for PPO update
                # (You'll need to implement a buffer for PPO)
            
            elif isinstance(agent, SACAgent):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.update()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    return episode_rewards

# Example usage
if __name__ == '__main__':
    from models.rl.sensor_gym_env import SensorGymEnv
    from models.pipelines.sensor_data_pipeline import SensorDataProcessor
    
    # Initialize environment
    processor = SensorDataProcessor()
    env = SensorGymEnv(processor)
    
    # Create and train DQN agent
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    dqn_agent = DQNAgent(state_size, action_size)
    dqn_rewards = train_agent(env, dqn_agent, num_episodes=100)
    
    # Create and train PPO agent
    ppo_agent = PPOAgent(state_size, action_size)
    ppo_rewards = train_agent(env, ppo_agent, num_episodes=100)
    
    # Create and train SAC agent
    sac_agent = SACAgent(state_size, action_size)
    sac_rewards = train_agent(env, sac_agent, num_episodes=100)
    
    env.close() 